#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_map, ssim_map
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
import math
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import cv2

import matplotlib.pyplot as plt

curr_file_path = os.path.abspath(__file__)
curr_dir_path = os.path.dirname(curr_file_path)
customized_odak_path = os.path.join(curr_dir_path, "../metamer")
sys.path.append(customized_odak_path)
from odak_perception import MetamerMSELoss, MetamericLoss, MetamericLossUniform

def log_message(message, file_path='training_log.log'):
    """
    Logs a message to the specified log file and flushes it immediately.
    
    Args:
    - message: The message to log.
    - file_path: The path to the log file.
    """
    with open(file_path, 'a') as log_file:
        log_file.write(message + '\n')
        log_file.flush()

def set_up_workdir(dataset, trial_name):
    work_dir = os.path.join(args.model_path, trial_name)
    dataset.model_path = work_dir
    os.makedirs(work_dir, exist_ok = True)

    log_path = os.path.join(work_dir, "log.txt")
    # clean the log file
    with open(log_path, 'w') as log_file:
        log_file.write("")
    
    # record the arguments
    log_message(str(args), log_path)

    return log_path


def metric_pruning(gaussians, prune_ratio=0.1, scene=None ,pipe=None, bg=None):
    view_stacks = scene.getTrainCameras().copy()
    pnum = gaussians.get_xyz.shape[0]
    # create gs_count and contibs as shape pnum,1
    metrics = torch.zeros((pnum, 1), device="cuda")
    # iterate the dataset and collect comp eff
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            if args.metric == "max_comp_efficiency":
                loss_map = torch.ones_like(viewpoint_cam.original_image.cuda()).float()
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, cuda_type="pcheck_obb_loss_weighted_max_count", loss_map=loss_map)
                contibs = render_pkg["contribs"].unsqueeze(1).float()
                gs_count = render_pkg["gs_count"].unsqueeze(1).float()
                cur_metric = contibs / (gs_count + 1e-7)
                cur_metric[gs_count < 1] = 0
                metrics[metrics < cur_metric] = cur_metric[metrics < cur_metric]
            elif args.metric == "surface":
                loss_map = torch.ones_like(viewpoint_cam.original_image.cuda()).float()
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, cuda_type="pcheck_obb_loss_weighted_max_count", loss_map=loss_map)
                contibs = render_pkg["contribs"].unsqueeze(1).float()
                cur_metric = contibs 
                metrics[metrics < cur_metric] = cur_metric[metrics < cur_metric]
            elif args.metric == "max_contrib":
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, only_train_shs_dc = True, cuda_type="pcheck_obb_max")
                gs_count = render_pkg["gs_count"].unsqueeze(1).float()
                contibs = render_pkg["contribs"].unsqueeze(1).float()
                cur_metric = contibs
                metrics[metrics < cur_metric] = cur_metric[metrics < cur_metric]
                

    _, indices = torch.sort(metrics, descending=False, dim=0)
    prune_num = int(pnum * prune_ratio)
    mask_index = indices[:prune_num]
    # make a mask vector that is 1 for the points to prune
    mask = torch.zeros((pnum, 1), device="cuda")
    mask[mask_index] = 1
    mask = mask.bool()
    gaussians.prune_points(mask.squeeze())
    
    return gaussians


def test_hvs_loss(gaussians, scene=None, pipe=None, bg=None, need_resize=False, size=(256, 256)):
    # test hvs loss in training set
    hvs_loss = MetamericLossUniform(n_pyramid_levels=args.n_pyramid_levels, n_orientations=args.n_orientations, 
                            pooling_size=args.pooling_size, device="cuda", loss_type="MSE", bilinear_downsampling=True) 
    if args.monitor_val:
        view_stacks = scene.getTestCameras().copy()
    else:
        view_stacks = scene.getTrainCameras().copy()
    hvs_loss_sum = 0.0
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, cuda_type="pcheck_obb")
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            
            if need_resize:
                gt_image = F.interpolate(gt_image.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
                image = F.interpolate(image.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
            else:
                gt_image = gt_image.unsqueeze(0)
                image = image.unsqueeze(0)
            hvs_loss_sum += hvs_loss(image, gt_image)
    return hvs_loss_sum / len(view_stacks)
            
def test_ssim_loss(gaussians, scene=None, pipe=None, bg=None):
    # test hvs loss in training set
    if args.monitor_val:
        view_stacks = scene.getTestCameras().copy()
    else:
        view_stacks = scene.getTrainCameras().copy()

    ssim_loss_sum = 0.0
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, cuda_type="pcheck_obb")
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            gt_image = gt_image.unsqueeze(0)
            image = image.unsqueeze(0)
            ssim_loss_sum += ssim(image, gt_image)
            # import ipdb; ipdb.set_trace()
    return ssim_loss_sum / len(view_stacks)

def test_psnr_loss(gaussians, scene=None, pipe=None, bg=None):
    # test hvs loss in training set
    if args.monitor_val:
        view_stacks = scene.getTestCameras().copy()
    else:
        view_stacks = scene.getTrainCameras().copy()

    psnr_loss_sum = 0.0
    for viewpoint_cam in view_stacks:
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, cuda_type="pcheck_obb")
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            gt_image = gt_image.unsqueeze(0)
            image = image.unsqueeze(0)

            psnr_loss_sum += psnr(image, gt_image).mean()
            # import ipdb; ipdb.set_trace()
    return psnr_loss_sum / len(view_stacks)


def training(dataset, opt, pipe, debug_from, trial_name):
    uniform_hvs_loss = MetamericLossUniform(n_pyramid_levels=args.n_pyramid_levels, n_orientations=args.n_orientations, 
                            pooling_size=args.pooling_size, device="cuda", loss_type="L1", bilinear_downsampling=True) 
    prune_interval = 1000
    prune_ratio = 0.02
    per_prune_iterative_times = 5
    scale_weight = 0.0 # initial weight for scale loss
    # Set up work directory
    log_path = set_up_workdir(dataset, trial_name)
    # write config args
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    ema_training_loss = None

    if args.pretrain_ply != None:
        gaussians.load_ply(args.pretrain_ply)
    else:
        raise "Pretrained 3DGS is not provided"
    
    gaussians.training_setup(opt)

    min_divisor = 2 ** args.n_pyramid_levels
    height = scene.getTrainCameras()[0].original_image.size(1)
    width = scene.getTrainCameras()[0].original_image.size(2)
    required_height = math.ceil(height / min_divisor) * min_divisor
    required_width = math.ceil(width / min_divisor) * min_divisor
    if required_height > height or required_width > width:
        need_resize = True
    else:
        need_resize = False


    for iteration in range(first_iter, opt.iterations + 1):   

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, cuda_type="pcheck_obb_sum")
        image = render_pkg["render"]
        gs_count = render_pkg["gs_count"]



        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        loss = 0.0
        Ll1 = l1_loss(image, gt_image)
        original_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss += original_loss


        if iteration < args.pruning_iters and args.use_scale_decay:
            scale = gaussians.get_scaling
            scale_max, _ = torch.max(scale, dim=1)
            scale_loss = ( scale_max  * (gs_count - 4) * (gs_count > 4)).mean() 
            loss += scale_loss * scale_weight


        if ema_training_loss is None:
            ema_training_loss = loss.item()
        else:   
            ema_training_loss = 0.6 * ema_training_loss + 0.4 * loss.item()


        loss.backward()
        
        with torch.no_grad():
            # Optimizer step
            if iteration <= opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Pruning
            if (iteration % prune_interval == 1) and iteration < args.pruning_iters:
                gaussians.prune(prune_method = "opacity", threshold = 0.005)
                # print point number
                log_message("Iteration = {}, Point Number Before Pruning: {}".format(iteration, gaussians.get_xyz.shape[0]), log_path)
                # perform metric pruning
                tested_ssim = test_ssim_loss(gaussians, scene=scene, pipe=pipe, bg=bg)
                tested_psnr = test_psnr_loss(gaussians, scene=scene, pipe=pipe, bg=bg)
                log_message("Iteration = {}, Testing SSIM loss, SSIM Loss in training/test set: {}".format(iteration, tested_ssim), log_path)
                log_message("Iteration = {}, Testing PSNR loss, PSNR Loss in training/test set: {}".format(iteration, tested_psnr), log_path)
                psnr_pass = tested_psnr >= args.target_psnr
                ssim_pass = tested_ssim >= args.target_ssim
                pass_test = psnr_pass & ssim_pass
                # import ipdb; ipdb.set_trace()
                if pass_test:
                    log_message("Iteration = {}, Pass Test, Saving Model Before Prune".format(iteration), log_path)
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/current-best.pth")
                    # perform metric pruning
                    for i in range(per_prune_iterative_times):
                        gaussians = metric_pruning(gaussians, prune_ratio=prune_ratio, scene=scene, pipe=pipe, bg=bg)

                    log_message("Iteration = {}, Metric Pruning Done, Point Number After Pruning: {}".format(iteration, gaussians.get_xyz.shape[0]), log_path)

                    scale_weight = scale_weight * 3
                    if scale_weight < 1e-4:
                        scale_weight = 1e-4

                    log_message("Iteration = {}, Scale Weight: {}".format(iteration, scale_weight), log_path)
                        
                    gaussians.reset_opacity_max(max=0.1)

                    torch.cuda.empty_cache()
                else:
                    scale_weight = scale_weight / 3
                    if scale_weight < 1e-4:
                        scale_weight = 0.0


                    log_message("Iteration = {}, Not Pass Test, Metric Pruning Skipped".format(iteration), log_path)
                    log_message("Iteration = {}, Scale Weight: {}".format(iteration, scale_weight), log_path)

                if not args.use_scale_decay:
                    scale_weight = 0.0
                    log_message("Iteration = {}, No Scale Weight: {}".format(iteration, scale_weight), log_path)
                    
                
            

            if iteration == args.pruning_iters:
                tested_ssim = test_ssim_loss(gaussians, scene=scene, pipe=pipe, bg=bg)
                tested_psnr = test_psnr_loss(gaussians, scene=scene, pipe=pipe, bg=bg)
                log_message("Iteration = {}, Testing SSIM loss, SSIM Loss in training/test set: {}".format(iteration, tested_ssim), log_path)
                log_message("Iteration = {}, Testing PSNR loss, PSNR Loss in training/test set: {}".format(iteration, tested_psnr), log_path)
                pass_test = (tested_ssim >= args.target_ssim) & (tested_psnr >= args.target_psnr)
                if not pass_test:
                    log_message("Iteration = {}, Tested Loss Not Pass, Reload Best Model and Perform Final Prune Before Adatation: ".format(iteration), log_path)
                    best_checkpoint = scene.model_path + "/current-best.pth"
                    (model_params, _) = torch.load(best_checkpoint)
                    gaussians.best_restore(model_params,opt)
                else:
                    log_message("Iteration = {}, Pruning Done, Tested Loss Pass".format(iteration), log_path)
                
                for i in range(per_prune_iterative_times):
                    log_message("Iteration = {}, Perform  {} th Final Prune Saving Model Before Prune ".format(iteration, i), log_path)
                    best_checkpoint = scene.model_path + "/current-best.pth"
                    torch.save((gaussians.capture(), iteration), best_checkpoint)
                    gaussians = metric_pruning(gaussians, prune_ratio=prune_ratio, scene=scene, pipe=pipe, bg=bg)
                    tested_ssim = test_ssim_loss(gaussians, scene=scene, pipe=pipe, bg=bg)
                    tested_psnr = test_psnr_loss(gaussians, scene=scene, pipe=pipe, bg=bg)
                    log_message("Iteration = {}, Testing SSIM loss, SSIM Loss in training/test set: {}".format(iteration, tested_ssim), log_path)
                    log_message("Iteration = {}, Testing PSNR loss, PSNR Loss in training/test set: {}".format(iteration, tested_psnr), log_path)
                    pass_test =  (tested_ssim >= args.target_ssim) & (tested_psnr >= args.target_psnr)
                    if not pass_test :
                        log_message("Iteration = {}, Final Pruning Meet target, Reload Best Model and leave for adaptation".format(iteration), log_path)
                        (model_params, _) = torch.load(best_checkpoint)
                        gaussians.best_restore(model_params,opt)
                        gaussians.prune(prune_method = "opacity", threshold = 0.005)
                        gaussians.reset_opacity_max(max=0.1)
                        break

            iter_end.record()


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = ema_training_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration == opt.iterations):
                gaussians.prune(prune_method = "opacity", threshold = 0.005)
                tested_ssim = test_ssim_loss(gaussians, scene=scene, pipe=pipe, bg=bg)
                tested_psnr = test_psnr_loss(gaussians, scene=scene, pipe=pipe, bg=bg)
                log_message("Iteration = {}, Final Testing SSIM loss, SSIM Loss: {}".format(iteration, tested_ssim), log_path)
                log_message("Iteration = {}, Final Testing PSNR loss, PSNR Loss: {}".format(iteration, tested_psnr), log_path)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                save_path = scene.get_save_path(iteration)
                gaussians.save_ply(save_path)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def update_opt(opt, pruned_iters, adaptation_iters):
    opt.iterations = pruned_iters + adaptation_iters
    opt.position_lr_max_steps = opt.iterations
    return opt


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # load the 3DGS model for pruning
    parser.add_argument("--pretrain_ply", type=str, default = None) # pretrained 3dgs for pruning
    # Set up the HVS loss
    parser.add_argument('--pooling_size', type=int, default=1)
    parser.add_argument('--n_pyramid_levels', type=int, default=5)
    parser.add_argument('--n_orientations', type=int, default=6)
    # Set up the pruning method
    parser.add_argument('--target_hvs', type=float, required=True) # targeted loss for adaptive pruning
    parser.add_argument('--target_ssim', type=float, required=True) # targeted loss for adaptive pruning
    parser.add_argument('--target_psnr', type=float, required=True) # targeted loss for adaptive pruning
    parser.add_argument('--pruning_iters', type=int, default = 25000) # iterations for pruning
    parser.add_argument('--final_adaptation_iters', type=int, default = 5000) # final adaptation iterations number after pruning
    parser.add_argument('--metric', type=str, default = "max_comp_efficiency", choices=["max_comp_efficiency", "max_contrib", "surface"])

    # Other arguments
    parser.add_argument('--trial_name', type=str, default = None) # working directory = dataset_path +/ trial_name
    parser.add_argument('--position_lr_init_scale', type=float, default=0.1) # scale the initial learning rate for position

    parser.add_argument("--monitor_val", action="store_true", default=False)
    parser.add_argument("--use_scale_decay", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])

    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(False)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Update the optimization parameters for pruning
    opt = op.extract(args)
    update_opt(opt, args.pruning_iters, args.final_adaptation_iters) 
    opt.position_lr_init = opt.position_lr_init * args.position_lr_init_scale

    training(lp.extract(args), opt, pp.extract(args), args.debug_from, args.trial_name)

    # All done
    print("\nTraining complete.")
