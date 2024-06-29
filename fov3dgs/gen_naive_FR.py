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
from utils.loss_utils import l1_loss, ssim
# from gaussian_renderer import render, network_gui
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

def compose(ply_paths, trial_name):
    os.makedirs(trial_name, exist_ok=True)
    sh_degree = 3

    for i, path in enumerate(ply_paths):
        print ("Processing: ", i)
        if (i == 0):
            gaussian_finest = GaussianModel(sh_degree)
            gaussian_finest.load_ply(path)
            gaussian_finest.init_index()

            pnum = gaussian_finest.get_xyz.shape[0]
            current_sampled_idx = torch.randperm(pnum)
            highest_levels = torch.zeros(pnum, dtype=torch.float)
        else:
            gaussian = GaussianModel(sh_degree)
            gaussian.load_ply_index(path)
            snum = gaussian.get_xyz.shape[0]

            sampled_idx = current_sampled_idx[:snum]    
            highest_levels[sampled_idx] = i
            current_sampled_idx = sampled_idx



    # import ipdb; ipdb.set_trace()
    torch.save(highest_levels, os.path.join(trial_name, "highest_levels.pth"))

    

def generate_ply_path(base, arg1, arg2, arg3, arg4):
    return f"{base}{arg1}_{arg2}-{arg3}_{args.layer_num}_{args.metric}/point_cloud/iteration_{arg4}/point_cloud.ply"

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--folder_base", type=str, default=None)
    parser.add_argument("--max_pooling_size", type=int, help="Max pooling size")
    parser.add_argument("--layer_num", type=int, help="Layer number")
    parser.add_argument("--metric", type=str, default="surface")

    args = parser.parse_args()

    # get sqrt of max pooling size
    sqrt_max_pooling_size = args.max_pooling_size**0.5
    # get interval according to layernum
    interval = float( (sqrt_max_pooling_size-1) / (args.layer_num -1))

    # generate layernum psizes
    psizes = []
    for i in range(args.layer_num):
        pooling_size = 1 + interval * i
        pooling_size = round(pooling_size**2)
        psizes.append(pooling_size)

    PS1_path = args.folder_base + f"1_PS1_{args.layer_num}_{args.max_pooling_size}/point_cloud/iteration_55000/point_cloud.ply"
    arg2_values = [6000] * (args.layer_num-1)
    arg3_values = [1500] * (args.layer_num-1)
    arg4_values = [7500] * (args.layer_num-1)
    # geneate 10 ply paths
    ply_paths = []
    ply_paths.append(PS1_path)
    for i in range(args.layer_num-1):
        ply_paths.append(generate_ply_path(args.folder_base, psizes[i+1], arg2_values[i], arg3_values[i], arg4_values[i]))




    compose(ply_paths, args.folder_base + f"all_shared_fr_{args.layer_num}_{args.max_pooling_size}")

    # All done
    print("\nTraining complete.")
