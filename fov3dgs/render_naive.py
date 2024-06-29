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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from utils.sh_utils import RGB2SH



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene=None):
    render_path = os.path.join(args.out_folder, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(args.out_folder, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fps_list = []

    # gaussians.training_setup(op)
    # gaussians = metric_resize(gaussians, scene=scene, pipe=pipeline, bg=background)
    # selected_pts_mask = torch.max(gaussians.get_scaling, dim=1).values > extent * 0.05
    # gaussians.prune_points(selected_pts_mask)

    # import ipdb; ipdb.set_trace()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        time = 0
        frame_num = 1
        for i in range(frame_num):
            rendering = render(view, gaussians, pipeline, background, starter=starter, ender=ender, cuda_type="pcheck_obb")["render"]
            torch.cuda.synchronize()
            time += starter.elapsed_time(ender)
        
        fps = frame_num / (time / 1000)
        print("FPS: ", frame_num / (time / 1000))
        
        fps_list.append(fps)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    print("Average FPS: ", sum(fps_list) / len(fps_list))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # load mask info
        highest_levels = torch.load(args.highest_levels_path)

        selected = highest_levels >= args.target_layer
        
        gaussians.training_setup(op)
        gaussians.prune_points(~selected)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--target_layer", default=0, type=int)
    parser.add_argument("--highest_levels_path", "-hl", type=str, required=True)
    parser.add_argument("--out_folder", "-of", type=str, required=True)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    op = OptimizationParams(parser)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)