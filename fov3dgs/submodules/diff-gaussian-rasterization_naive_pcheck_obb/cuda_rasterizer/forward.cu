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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}




// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* shs_rest,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float3* conics,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float* eigen_lengths,
	float* eigen_vecs
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	uint potential_tnum = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	if (potential_tnum == 0)
		return;

	float len1 = 0;
	float len2 = 0;
	float2 eigenvector1 = { 0, 0 };
	float2 eigenvector2 = { 0, 0 };

	if 	(potential_tnum > 1) {
		// compute eigenvectors
		float a1 = cov.x - lambda1;
		float b1 = cov.y;
		float a2 = cov.x - lambda2;
		float b2 = cov.y;

		eigenvector1 = {-b1, a1};
		eigenvector2 = {-b2, a2};

		// Normalize eigenvectors
		normalize(eigenvector1);
		normalize(eigenvector2);

		len1 = 3.0f * sqrtf(lambda1);
		len2 = 3.0f * sqrtf(lambda2);
	}


	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// if (colors_precomp == nullptr)
	// {
	// 	glm::vec3 result = computeRestColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs_rest);
	// 	rgb[idx * C + 0] = result.x;
	// 	rgb[idx * C + 1] = result.y;
	// 	rgb[idx * C + 2] = result.z;
	// }

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conics[idx] = conic;
	tiles_touched[idx] = potential_tnum;

	eigen_lengths[idx * 2] = len1;
	eigen_lengths[idx * 2 + 1] = len2;
	eigen_vecs[idx * 4] = float(eigenvector1.x);
	eigen_vecs[idx * 4 + 1] = float(eigenvector1.y);
	eigen_vecs[idx * 4 + 2] = float(eigenvector2.x);
	eigen_vecs[idx * 4 + 3] = float(eigenvector2.y);
}


struct alignas(16) EarlySkipInfo1 {
    float2 xy;
    float3 conic;
	float opacity1;
};

struct alignas(16) EarlySkipInfo2 {
	float highest_level;
};

struct alignas(16) RenderInfo1 {
    float3 features;
};



template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_blending(const int P,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float3* __restrict__ features,
	//
	const float* __restrict__ tile_levels,
	const float* __restrict__ tile_level_gradient_ys,
	const float* __restrict__ tile_level_gradient_xs,
	const bool* __restrict__ tile_blendings,
	const float* __restrict__ highest_levels,
	const float* __restrict__ opacities,
	//
	const float3* __restrict__ conics,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	int tile_width_num = (W + 15) / BLOCK_X;
	uint current_tile_idx = block.group_index().x + tile_width_num  * block.group_index().y ;
	bool blending = tile_blendings[current_tile_idx];
	if (!blending){
		return;
	}

	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ EarlySkipInfo1 skip_infos1[BLOCK_SIZE];
	__shared__ EarlySkipInfo2 skip_infos2[BLOCK_SIZE];
	__shared__ RenderInfo1 render_infos1[BLOCK_SIZE];

	// Initialize helper variables
	float T1 = 1.0f;
	float T2 = 1.0f;
	float C1[CHANNELS] = { 0 };
	float C2[CHANNELS] = { 0 };
	float estimated_pix_level;
	float L1_f;
	float tile_level_f = tile_levels[current_tile_idx];
	int tile_level_i = (int)tile_level_f;


	float dx = float(block.thread_index().x);
	float dy = float(block.thread_index().y);
	estimated_pix_level = tile_level_f + (dx * tile_level_gradient_xs[current_tile_idx] + dy * tile_level_gradient_ys[current_tile_idx]) / (float)BLOCK_X;

	// Decide the levels to blend
	int L1_i, L2_i;
	float L2_f;
	L1_i = tile_level_i;
	L1_f = tile_level_f;
	L2_i = L1_i + 1;
	L2_f = L1_f + 1.0f;

	// summarize what to render
	bool L1_done = false;
	if (estimated_pix_level > float(L2_i)){
		L1_done = true;
	}
	bool L2_done = false; // If only one level, L2 is done

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			EarlySkipInfo1 skip_info1;
			skip_info1.xy = points_xy_image[coll_id];
			skip_info1.conic = conics[coll_id];
			skip_info1.opacity1 = opacities[coll_id];
			skip_infos1[block.thread_rank()] = skip_info1;

			RenderInfo1 render_info1;
			render_info1.features = features[coll_id];
			render_infos1[block.thread_rank()] = render_info1;

			EarlySkipInfo2 skip_info2;
			skip_info2.highest_level = highest_levels[coll_id];
			skip_infos2[block.thread_rank()] = skip_info2;

		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{

			// Stage1 ======== Perform early skip test for both pix (shared part) ===============
			EarlySkipInfo1 skip_info1 = skip_infos1[j];
			float2 xy = skip_info1.xy;
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float3 con_o = skip_info1.conic;
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f || power < -4.5f)
				continue;
			float exp_val = exp(power);

			float alpha1 = min(0.99f, skip_info1.opacity1 *exp_val);
			bool alpha_skip_1 = alpha1 < 1.0f / 255.0f;

			// Stage2 ======== Deal with L1 ===============
			if (!L1_done)
			{
				if (!alpha_skip_1){
					float test_T1 = T1 * (1 - alpha1);
					L1_done = (test_T1 < 0.0001f);
					if (!L1_done)
					{
						RenderInfo1 render_info1 = render_infos1[j];
						float w = alpha1 * T1;
						C1[0] += render_info1.features.x * w;
						C1[1] += render_info1.features.y * w;
						C1[2] += render_info1.features.z * w;
						T1 = test_T1;
					}

				}
				else{
					continue;
				}
			}

			// Stage3 ======== Deal With L2 ==============
			if (!L2_done)
			{
				EarlySkipInfo2 skip_info2 = skip_infos2[j];
				bool skip2 = ( (skip_info2.highest_level + 1) < L2_f);
				if (!skip2){
					float test_T2 = T2 * (1 - alpha1);
					L2_done = (test_T2 < 0.0001f);
					if (!L2_done)
					{
						RenderInfo1 render_info1 = render_infos1[j];
						float w = alpha1 * T2;
						C2[0] += render_info1.features.x * w;
						C2[1] += render_info1.features.y * w;
						C2[2] += render_info1.features.z * w;
						T2 = test_T2;
					}
				}
			}

			if (L1_done && L2_done){
				done = true;
				continue;
			}

		}
	}

	

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// first calculate C1
		for (int ch = 0; ch < CHANNELS; ch++)
			C1[ch] = C1[ch] + bg_color[ch] * T1;


		for (int ch = 0; ch < CHANNELS; ch++)
			C2[ch] = C2[ch] + bg_color[ch] * T2;


		float x = fabs( estimated_pix_level - (float(L1_i) + start_blend) ) / blend_width;

		x = max(0.0f, min(1.0f, x));
		float L1_w;
		float blend_T = 3 * x * x - 2 * x * x * x;
		L1_w = 1 - blend_T;

		float C[CHANNELS];
		for (int ch = 0; ch < CHANNELS; ch++)
		{
			C[ch] = C1[ch] * L1_w + C2[ch] * (1.f - L1_w);
		}
		
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch];
	}
}


struct alignas(16) EarlySkipInfo {
    float2 xy;
    float3 conic;
	float opacity1;
};

struct alignas(16) RenderInfo {
    float3 features;
};


template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(const int P,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float3* __restrict__ features,
	//
	const float* __restrict__ tile_levels,
	const bool* __restrict__ tile_blendings,
	const float* __restrict__ opacities,
	//
	const float3* __restrict__ conics,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	int tile_width_num = (W + 15) / BLOCK_X;
	uint current_tile_idx = block.group_index().x + tile_width_num  * block.group_index().y ;
	bool blending = tile_blendings[current_tile_idx];
	if (blending){
		return;
	}

	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ EarlySkipInfo skip_infos[BLOCK_SIZE];
	__shared__ RenderInfo render_infos[BLOCK_SIZE];

	// Initialize helper variables
	float T1 = 1.0f;
	float C1[CHANNELS] = { 0 };
	float tile_level_f = tile_levels[current_tile_idx];
	int tile_level_i = (int)tile_level_f;


	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			EarlySkipInfo skip_info;
			skip_info.xy = points_xy_image[coll_id];
			skip_info.conic = conics[coll_id];
			skip_info.opacity1 = opacities[coll_id];
			skip_infos[block.thread_rank()] = skip_info;

			RenderInfo render_info;
			render_info.features = features[coll_id];
			render_infos[block.thread_rank()] = render_info;
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			EarlySkipInfo skip_info = skip_infos[j];
			float2 xy = skip_info.xy;
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float3 con_o = skip_info.conic;
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f || power < -4.5f)
				continue;

			float alpha = min(0.99f, skip_info.opacity1 * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T1 = T1 * (1 - alpha);
			if (test_T1 < 0.0001f)
			{
				done = true;
				continue;
			}
			RenderInfo render_info = render_infos[j];

			// Stage3 ======== Perform C1 integration ===============
			float w = alpha * T1;
			C1[0] += render_info.features.x * w;
			C1[1] += render_info.features.y * w;
			C1[2] += render_info.features.z * w;
			T1 = test_T1;
		}
	}


	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] =  C1[ch] + bg_color[ch] * T1;
	}
}


void FORWARD::render( const int P,
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float3* colors,
	//
	const float* __restrict__ tile_levels,
	const bool* __restrict__ tile_blendings,
	const float* __restrict__ opacities,
	//
	const float3* conics,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		P,
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		//
		tile_levels,
		tile_blendings,
		opacities,
		//
		conics,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}


void FORWARD::render_blending( const int P,
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float3* colors,
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
	float* out_color)
{
	renderCUDA_blending<NUM_CHANNELS> << <grid, block >> > (
		P,
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		//
		tile_levels,
		tile_level_gradient_ys,
		tile_level_gradient_xs,
		tile_blendings,
		highest_levels,
		opacities,
		//
		conics,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}


void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* shs_rest,
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
	float* eigen_vecs)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		shs_rest,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conics,
		grid,
		tiles_touched,
		prefiltered,
		eigen_lengths,
		eigen_vecs
		);
}





