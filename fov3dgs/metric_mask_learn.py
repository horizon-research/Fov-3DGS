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
from torch import optim
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
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, cuda_type="pcheck_obb_max")
                gs_count = render_pkg["gs_count"].unsqueeze(1).float()
                contibs = render_pkg["contribs"].unsqueeze(1).float()
                cur_metric = contibs
                metrics[metrics < cur_metric] = cur_metric[metrics < cur_metric]
                
    # sort the metric
    _, indices = torch.sort(metrics, descending=False, dim=0)

    # prune the points
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
            
    

def training(dataset, opt, pipe, debug_from, trial_name):
    uniform_hvs_loss = MetamericLossUniform(n_pyramid_levels=args.n_pyramid_levels, n_orientations=args.n_orientations, 
                            pooling_size=args.pooling_size, device="cuda", loss_type=args.hvs_loss_type, bilinear_downsampling=True) 

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
        # import ipdb; ipdb.set_trace()
        if args.init_index:
            gaussians.load_ply(args.pretrain_ply)
            gaussians.init_index()
        else:
            gaussians.load_ply_index(args.pretrain_ply)
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

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, masking = True, cuda_type="pcheck_obb_sum")
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        loss = 0.0

        if need_resize:
            gt_image = F.interpolate(gt_image.unsqueeze(0), size=(required_height, required_width), mode='bilinear', align_corners=False)
            image = F.interpolate(image.unsqueeze(0), size=(required_height, required_width), mode='bilinear', align_corners=False)
        else:
            gt_image = gt_image.unsqueeze(0)
            image = image.unsqueeze(0)
        loss += uniform_hvs_loss(image, gt_image)

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

            # Masking
            if (iteration % prune_interval == 1) and iteration < args.pruning_iters:
                gaussians.prune(prune_method = "opacity", threshold = 0.005)
                # print point number
                log_message("Iteration = {}, Point Number Before Masking: {}".format(iteration, gaussians.get_xyz.shape[0]), log_path)
                # perform metric Masking
                tested_loss = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=bg, need_resize=need_resize, size=(required_height, required_width))
                log_message("Iteration = {}, Testing HVS loss, HVS Loss in training set: {}".format(iteration, tested_loss), log_path)

                if tested_loss <= args.target_loss:
                    current_best_iter = iteration
                    log_message("Iteration = {}, Pass Test, Saving Model Before Masking".format(iteration), log_path)
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/current-best.pth")

                    for i in range(per_prune_iterative_times):
                        gaussians = metric_pruning(gaussians, prune_ratio=prune_ratio, scene=scene, pipe=pipe, bg=bg)

                    log_message("Iteration = {}, Metric Pruning Done, Point Number After Masking: {}".format(iteration, gaussians.get_xyz.shape[0]), log_path)
                    torch.cuda.empty_cache()
                    # if (iteration < reset_until):
                    gaussians.reset_opacity_max(max=0.1)
                else:
                    log_message("Iteration = {}, Not Pass Test, Metric Masking Skipped".format(iteration), log_path)
                
            

            if iteration == args.pruning_iters:
                tested_loss = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=bg, need_resize=need_resize, size=(required_height, required_width))
                log_message("Iteration = {}, Masking Done, Testing HVS loss, HVS Loss in training set: {}".format(iteration, tested_loss), log_path)

                if tested_loss > args.target_loss:
                    log_message("Iteration = {}, Tested Loss Not Pass, Reload Best Model Before Final Pruning: ".format(iteration), log_path)
                    best_checkpoint = scene.model_path + "/current-best.pth"
                    (model_params, _) = torch.load(best_checkpoint)
                    gaussians.best_restore(model_params,opt)
                else:
                    log_message("Iteration = {}, Pruning Done, Tested Loss Pass".format(iteration), log_path)

                # final pruning
                for i in range(per_prune_iterative_times):
                    log_message("Iteration = {}, Final Prune {} Before Adaptation, Saving Model Before Prune".format(iteration, i), log_path)
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/current-best.pth")
                    gaussians = metric_pruning(gaussians, prune_ratio=prune_ratio, scene=scene, pipe=pipe, bg=bg)
                    tested_loss = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=bg, need_resize=need_resize, size=(required_height, required_width))
                    if tested_loss > args.target_loss:
                        log_message("Iteration = {}, Final Prune {}  Meet Target, Reload Best Model Before Adatation: ".format(iteration, i), log_path)
                        best_checkpoint = scene.model_path + "/current-best.pth"
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
                tested_loss = test_hvs_loss(gaussians, scene=scene, pipe=pipe, bg=bg, need_resize=need_resize, size=(required_height, required_width))
                log_message("Iteration = {}, Final Testing HVS loss, HVS Loss in training set: {}".format(iteration, tested_loss), log_path)

                print("\n[ITER {}] Saving Gaussians".format(iteration))
                gaussians.prune(prune_method = "opacity", threshold = 0.005)
                save_path = scene.get_save_path(iteration)
 
                gaussians.save_ply_index(save_path)
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def update_opt(opt, pruned_iters, adaptation_iters):
    opt.iterations = pruned_iters + adaptation_iters
    opt.position_lr_max_steps = opt.iterations
    return opt


if __name__ == "__main__":
    prune_interval = 500
    prune_ratio = 0.02
    # reset_until = 6000
    per_prune_iterative_times = 5
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
    parser.add_argument("--hvs_loss_type", type=str, default = "L1") # what loss to use during pruning
    parser.add_argument('--pooling_size', type=int, default=1)
    parser.add_argument('--n_pyramid_levels', type=int, default=5)
    parser.add_argument('--n_orientations', type=int, default=6)
    # Set up the pruning method
    parser.add_argument('--target_loss', type=float, default=0.5) # targeted loss for adaptive pruning
    parser.add_argument('--pruning_iters', type=int, default = 25000) # iterations for pruning
    parser.add_argument('--final_adaptation_iters', type=int, default = 5000) # final adaptation iterations number after pruning
    parser.add_argument('--metric', type=str, default = "max_comp_efficiency", choices=["max_comp_efficiency", "max_contrib", "surface"])

    # Other arguments
    parser.add_argument('--trial_name', type=str, default = None) # working directory = dataset_path +/ trial_name
    parser.add_argument("--init_index", action="store_true", default=False)  # whether to initialize the index for the 3DGS, it is for FOV masking
    
    parser.add_argument("--monitor_val", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])

    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(False)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Update the optimization parameters for pruning
    opt = op.extract(args)
    update_opt(opt, args.pruning_iters, args.final_adaptation_iters) 

    training(lp.extract(args), opt, pp.extract(args), args.debug_from, args.trial_name)

    # All done
    print("\nTraining complete.")
