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
from gaussian_wrapper import get_gs_rasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_pcheck_obb_sum import GaussianRasterizationSettings

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, masking = False,
           starter=None, ender=None, cuda_type = "", loss_map = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
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
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = get_gs_rasterizer(cuda_type, raster_settings)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    cov3D_precomp = None

    scales = pc.get_scaling
    rotations = pc.get_rotation

    


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None

    if masking:
        shs = pc.get_features_detach_rest
    else:
        shs = pc.get_features


    if masking:
        # detach all other tensors except shs and opacity
        scales = scales.detach()
        means3D = means3D.detach()
        rotations = rotations.detach()
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if starter is not None:
        starter.record()

    
    if cuda_type == "pcheck_obb_loss_weighted_max_count":
        rendered_image, radii, gs_count, contribs = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            loss_map = loss_map)
    elif cuda_type != "pcheck_obb_max" and cuda_type != "pcheck_obb_sum" and cuda_type != "pcheck_obb_for_mask":
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, gs_count, contribs = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    
    if ender is not None:
        ender.record()

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if cuda_type == "pcheck_obb_loss_weighted_max_count":
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "gs_count": gs_count,
                "contribs": contribs
                }
    elif cuda_type != "pcheck_obb_max" and cuda_type != "pcheck_obb_sum" and cuda_type != "pcheck_obb_for_mask":
        return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "gs_count": gs_count,
                "contribs": contribs
                }
