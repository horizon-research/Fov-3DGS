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
from gaussian_renderer_fov_naive import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer_fov_naive import GaussianModel
import sys

def render_set(out_path, name, iteration, views, gaussians, pipeline, background, highest_levels):
    gaze_samples = [(0.25 * i, 0.25 * j) for i in range(1, 4) for j in range(1, 4)]
    print("Gaze samples:", gaze_samples)
    
    highest_levels = highest_levels.cuda()


    # import ipdb; ipdb.set_trace()   

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    avg_fpss = []
    
    first_view = views[0]

    for gi, gaze in enumerate(gaze_samples):
        gazeArray = torch.tensor([gaze[0], gaze[1]]).float().cuda()
        
        
        for i in range(10):
            blending_rendering = render(first_view, gaussians, background, alpha=0.05, gazeArray = gazeArray, blending=True, starter= starter, ender= ender, highest_levels=highest_levels)["render"]
            torch.cuda.synchronize()
        
        fpss = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            time = 0
            for i in range(5):
                blending_rendering = render(view, gaussians, background, alpha=0.05, gazeArray = gazeArray, blending=False, starter= starter, ender= ender, highest_levels=highest_levels)["render"]
                torch.cuda.synchronize()
                time += starter.elapsed_time(ender)
            
            fps = 5 / (time / 1000)
            print("FPS: ", fps)
            fpss.append(fps)
            
            # save the rendering
            # torchvision.utils.save_image(blending_rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            

            
        avg_fps = sum(fpss) / len(fpss)
        print(f"Average FPS of gaze {gaze[0]}, {gaze[1]}: {avg_fps}")
        avg_fpss.append(avg_fps)
        
        
    print("Average FPS: ", sum(avg_fpss) / len(avg_fpss))



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False, fps_mode=True)
        finest_gs_path = args.base_folder + f"1_PS1_{args.layer_num}_{args.max_pooling_size}/point_cloud/iteration_55000/point_cloud.ply"
        gaussians.load_ply(finest_gs_path)

        # load mask info
        mask_path = os.path.join(args.base_folder,f"all_shared_fr_{args.layer_num}_{args.max_pooling_size}", "highest_levels.pth")
        highest_levels = torch.load(mask_path)


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out_path = os.path.join(args.base_folder, f"all_shared_fr_{args.layer_num}_{args.max_pooling_size}")

        # if not skip_train:
        #      render_set(out_path, "train", 0, scene.getTrainCameras(), gaussians, pipeline, background, highest_levels)

        if not skip_test:
             render_set(out_path, "test", 0, scene.getTestCameras(), gaussians, pipeline, background, highest_levels)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--layer_num", type=int, help="Layer number")
    parser.add_argument("--max_pooling_size", type=int, help="Max pooling size")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)