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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from hvs_loss_calc import HVSLoss

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate():

    full_dict = {}
    per_view_dict = {}
    print("")

    hvs_calc = HVSLoss(uniform_pooling_size=args.ps)

    torch.backends.cudnn.benchmark = True


    scene_dir = Path(args.result_folder)
    print("Test Dir:", scene_dir)
    method = f"ps={args.ps}"
    print("Method:", method)

    full_dict[method] = {}
    per_view_dict[method] = {}

    method_dir = scene_dir 
    gt_dir = method_dir/ "gt"
    renders_dir = method_dir / "renders"
    renders, gts, image_names = readImages(renders_dir, gt_dir)

    hvss = []
    for idx in tqdm(range(len(renders)), desc=f"PS {args.ps} Metric evaluation progress"):
        # import ipdb; ipdb.set_trace()
        hvss.append(hvs_calc.calc_uniform_loss(renders[idx], gts[idx], pooling_size=args.ps))
    print(" HVS  : {:>12.7f}".format(torch.tensor(hvss).mean(), ".5"))
    print("")

    full_dict[method].update({"HVS": torch.tensor(hvss).mean().item()})
    

    per_view_dict[method].update({"Per HVS": {name: hvs for hvs, name in zip(torch.tensor(hvss).tolist(), image_names)}})

    with open(args.output, 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(args.output2, 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # gaze smaple = (0.25, 0.25), (0.25, 0.5), (0.25, 0.75), (0.5, 0.25) ...

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--result_folder', "-r", required=True, type=str)
    parser.add_argument('--output', "-o", required=True, type=str)
    parser.add_argument('--output2', "-o2", required=True, type=str)
    parser.add_argument('--ps', "-ps", required=True, type=int)
    args = parser.parse_args()
    evaluate()
