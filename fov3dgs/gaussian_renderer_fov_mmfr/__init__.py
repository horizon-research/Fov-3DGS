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
import math
# from diff_gaussian_rasterization_fov import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_mmfr_pcheck_obb import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, bg_color : torch.Tensor, scaling_modifier = 1.0, alpha=None, gazeArray=None, blending = None, starter=None, ender=None, multi_gs = None, layer_num = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(multi_gs[0].get_xyz, dtype=multi_gs[0].get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=multi_gs[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    means3D_list = []
    opacity_list = []
    scales_list = []
    rotations_list = []
    shs_list = []


    for gi in range(len(multi_gs)):
        means3D_list.append(multi_gs[gi].get_xyz)
        opacity_list.append(multi_gs[gi].get_opacity)
        scales_list.append(multi_gs[gi].get_scaling)
        rotations_list.append(multi_gs[gi].get_rotation)
        shs_list.append(multi_gs[gi].get_features)
        


    if starter is not None:
        starter.record()


    i = 0
    means3D = means3D_list[i]
    opacity = opacity_list[i]
    scales = scales_list[i]
    rotations = rotations_list[i]
    shs = shs_list[i]
    rendered_image0, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        cur_level =  i,
        gazeArray = gazeArray,
        alpha = alpha, 
        blending = blending)
    
    

    i = 1
    means3D = means3D_list[i]
    opacity = opacity_list[i]
    scales = scales_list[i]
    rotations = rotations_list[i]
    shs = shs_list[i]
    rendered_image1, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        cur_level =  i,
        gazeArray = gazeArray,
        alpha = alpha, 
        blending = blending)
    

    i = 2
    means3D = means3D_list[i]
    opacity = opacity_list[i]
    scales = scales_list[i]
    rotations = rotations_list[i]
    shs = shs_list[i]
    rendered_image2, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        cur_level =  i,
        gazeArray = gazeArray,
        alpha = alpha, 
        blending = blending)
    

    i = 3
    means3D = means3D_list[i]
    opacity = opacity_list[i]
    scales = scales_list[i]
    rotations = rotations_list[i]
    shs = shs_list[i]
    rendered_image3, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        cur_level =  i,
        gazeArray = gazeArray,
        alpha = alpha, 
        blending = blending)
    


    all_rendered_image = rendered_image0 + rendered_image1 + rendered_image2 + rendered_image3

    # import ipdb; ipdb.set_trace()
    

    if ender is not None:
        ender.record()


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": all_rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
