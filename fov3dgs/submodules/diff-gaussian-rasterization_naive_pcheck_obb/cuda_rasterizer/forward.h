/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	void preprocess(int P, int D, int M,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* shs,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* means2D,
		float* depths,
		float* cov3Ds,
		float* rgb,
		float3* conics,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		float* eigen_lengths,
		float* eigen_vecs);

	void render(
		const int P,
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float3* __restrict__ colors,
		//
		const float* __restrict__ tile_levels,
		const bool* __restrict__ tile_blendings,
		const float* __restrict__ opacities,
		//
		const float3* conics,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color);

	void render_blending(
		const int P,
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float3* __restrict__ colors,
		//
		const float* __restrict__ tile_levels,
		const float* __restrict__ tile_level_gradient_ys,
		const float* __restrict__ tile_level_gradient_xs,
		const bool* __restrict__ tile_blendings,
		const float* __restrict__ highest_levels,
		const float* __restrict__ opacities,
		//
		const float3* conics,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color);
}


#endif