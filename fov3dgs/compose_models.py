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

curr_file_path = os.path.abspath(__file__)
curr_dir_path = os.path.dirname(curr_file_path)
customized_odak_path = os.path.join(curr_dir_path, "../metamer")
sys.path.append(customized_odak_path)
from odak_perception import MetamerMSELoss, MetamericLoss, MetamericLossUniform

def compose(ply_paths, trial_name):

    os.system(f"cp {args.folder_base}/1_PS1_{args.layer_num}_{args.max_pooling_size}/cfg_args {args.folder_base}/cfg_args")
    # import ipdb; ipdb.set_trace()

    sh_degree = 3
    for i, path in enumerate(ply_paths):
        print ("Processing: ", i)
        if (i == 0):
            gaussian_finest = GaussianModel(sh_degree)
            gaussian_finest.load_ply(path)
            gaussian_finest.init_index()
            shs_dcs = torch.zeros((gaussian_finest.get_xyz.shape[0], args.layer_num, 3))
            highest_levels = torch.zeros((gaussian_finest.get_xyz.shape[0], 1))
            shs_dcs[:, 0, :] = gaussian_finest.get_features[:, 0, :].cpu()

            opacities = torch.ones((gaussian_finest.get_xyz.shape[0], args.layer_num))
            opacities[:, 0] = gaussian_finest.get_opacity[:, 0].cpu()
        else:
            gaussian = GaussianModel(sh_degree)
            gaussian.load_ply_index(path)
            shs_dcs[:, i, :] = shs_dcs[:, i-1, :] 
            features = gaussian.get_features[:, 0, :].cpu().unsqueeze(1)
            shs_dcs[gaussian.indexes.cpu().long(), i, :]  = features

            # import ipdb; ipdb.set_trace()
            
            opacities[:, i] = opacities[:, i-1]
            opacity = gaussian.get_opacity[:, 0].cpu().unsqueeze(1)
            opacities[gaussian.indexes.cpu().long(), i] = opacity

            # import ipdb; ipdb.set_trace()

            highest_levels[gaussian.indexes.cpu().long()] = i
            # import ipdb; ipdb.set_trace()
    
    # save higest levels and shs_dcs   
    # mkdir, exist ok
    os.makedirs(trial_name, exist_ok=True)
    torch.save(highest_levels, os.path.join(trial_name, "highest_levels.pt"))
    torch.save(shs_dcs, os.path.join(trial_name, "shs_dcs.pt"))
    torch.save(opacities, os.path.join(trial_name, "opacities.pt"))

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




    compose(ply_paths, args.folder_base + f"composed_{args.layer_num}_{args.max_pooling_size}")

    # All done
    print("\nTraining complete.")
