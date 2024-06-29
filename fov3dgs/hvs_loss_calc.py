import argparse
import glob
import cv2

import os
import torch
import sys

import torch.nn.functional as F
from tqdm import tqdm
import math

curr_file_path = os.path.abspath(__file__)
curr_dir_path = os.path.dirname(curr_file_path)
customized_odak_path = os.path.join(curr_dir_path, "../metamer")
sys.path.append(customized_odak_path)
from odak_perception import MetamerMSELoss, MetamericLoss, MetamericLossUniform
from utils.loss_utils import l1_loss, ssim


class HVSLoss:
    def __init__(self, n_pyramid_levels=5, n_orientations=6, device="cuda", loss_type="MSE", bilinear_downsampling=True, alpha=0.05, 
                 real_image_width=1.0, real_viewing_distance=0.5, uniform_pooling_size=1):
        self.n_pyramid_levels = n_pyramid_levels
        self.n_orientations = n_orientations
        self.device = device
        self.bilinear_downsampling = bilinear_downsampling
        self.loss_type = loss_type
        self.uniform_loss = MetamericLossUniform(n_pyramid_levels=self.n_pyramid_levels, n_orientations=self.n_orientations, 
                            pooling_size=uniform_pooling_size, device=self.device, loss_type=self.loss_type, bilinear_downsampling=self.bilinear_downsampling)
        
        self.alpha = alpha
        # Initialize self.fov_loss
        self.fov_loss = MetamericLoss(
            device=self.device,
            alpha=self.alpha,
            real_image_width=real_image_width,
            real_viewing_distance=real_viewing_distance,
            n_pyramid_levels=self.n_pyramid_levels,
            mode="quadratic",
            n_orientations=self.n_orientations,
            use_l2_foveal_loss=False,
            fovea_weight=False,
            use_radial_weight=False,
            use_fullres_l0=False,
            equi=False,
            loss_type=self.loss_type,
            use_bilinear_downup=bilinear_downsampling
        )
    
    def resize_img(self, img):
        min_divisor = 2 ** self.n_pyramid_levels
        height = img.size(2)
        width = img.size(3)
        required_height = math.ceil(height / min_divisor) * min_divisor
        required_width = math.ceil(width / min_divisor) * min_divisor
        if required_height > height or required_width > width:
            need_resize = True
        else:
            need_resize = False
        if need_resize:
            return F.interpolate(img, size=(required_height, required_width), mode='bilinear', align_corners=False)
        else:
            return img
        
    def calc_uniform_loss(self, img1, img2, pooling_size=1):
        self.uniform_loss.pooling_size = pooling_size
        img1 = self.resize_img(img1)
        img2 = self.resize_img(img2)
        return self.uniform_loss(img1, img2)
    
    def calc_fov_loss(self, img1, img2, gaze=[0.5, 0.5]):
        img1 = self.resize_img(img1)
        img2 = self.resize_img(img2)
        return self.fov_loss(img1, img2, gaze=gaze)
