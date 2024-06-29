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

    hvs_calc = HVSLoss()

    torch.backends.cudnn.benchmark = True


    scene_dir = Path(args.ps1_folder)
    print("Test Dir:", scene_dir)
    method = "ps1"
    print("Method:", method)

    full_dict[method] = {}
    per_view_dict[method] = {}

    method_dir = scene_dir 
    gt_dir = method_dir/ "gt"
    renders_dir = method_dir / "renders"
    renders, gts, image_names = readImages(renders_dir, gt_dir)

    ssims = []
    psnrs = []
    lpipss = []
    hvss = []
    for idx in tqdm(range(len(renders)), desc="PS1 Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        # import ipdb; ipdb.set_trace()
        hvss.append(hvs_calc.calc_uniform_loss(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))



    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print(" HVS  : {:>12.7f}".format(torch.tensor(hvss).mean(), ".5"))
    print("")

    full_dict[method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item(),
                                            "HVS": torch.tensor(hvss).mean().item()})
    

    per_view_dict[method].update({"Per SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                "Per PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                "Per LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                "Per HVS": {name: hvs for hvs, name in zip(torch.tensor(hvss).tolist(), image_names)},
                                                })

    with open(args.output, 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(args.output2, 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # gaze smaple = (0.25, 0.25), (0.25, 0.5), (0.25, 0.75), (0.5, 0.25) ...
    gaze_samples = [(0.25 * i, 0.25 * j) for i in range(1, 4) for j in range(1, 4)]
    print("Gaze samples:", gaze_samples)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ps1_folder', "-ps1", required=True, type=str)
    parser.add_argument('--output', "-o", required=True, type=str)
    parser.add_argument('--output2', "-o2", required=True, type=str)
    args = parser.parse_args()
    evaluate()
