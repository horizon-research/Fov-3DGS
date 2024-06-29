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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	return glm::max(result, 0.0f);
}

__forceinline__ __device__ float calculate_distance(float location_x, float location_y, float location_z) {
    // Compute the Euclidean distance in 3D space.
    return sqrtf(location_x * location_x + location_y * location_y + location_z * location_z);
}

__forceinline__ __device__ void norm_vector(float3 &vec) {
	// Normalize the vector.
	float distance = calculate_distance(vec.x, vec.y, vec.z);
	vec.x /= distance;
	vec.y /= distance;
	vec.z /= distance;
}

__forceinline__ __device__ float3 ncd2dir(const float2 ncd, const float real_width, const float real_height) {
	// Convert a 2D vector to a 3D vector.
	float3 vec3;
	vec3.x = (ncd.x - 0.5f) * real_width;
	vec3.y = (ncd.y - 0.5f) * real_height;
	vec3.z = real_viewing_distance;

	norm_vector(vec3);

	return vec3;

}

__forceinline__ __device__ float dot(const float3 dir1, const float3 dir2) {
	// Compute the dot product of two vectors.
	return dir1.x * dir2.x + dir1.y * dir2.y + dir1.z * dir2.z;

}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void compute_tile_levels_cuda(
	int T,
	float* __restrict__ tile_levels,
	const float2 gaze,
	const int W,
	const int H, 
	const int tile_width_num,
	const float alpha
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;

	// reverse idx to get tile W and H
	int tile_y = idx / tile_width_num;
	int tile_x = idx % tile_width_num;
	// get the pixel of the tile
	float p_x = tile_x * BLOCK_X + BLOCK_X / 2;
	float p_y = tile_y * BLOCK_Y + BLOCK_Y / 2;
	float2 p = make_float2(p_x, p_y);
	float real_image_height = float(H) / float(W) * real_image_width;

	float2 tile_ncd;
	tile_ncd.x = (p.x / W);
	tile_ncd.y = (p.y / H);
	float3 tile_dir = ncd2dir(tile_ncd, real_image_width, real_image_height);

	float3 gaze_dir = ncd2dir(gaze, real_image_width, real_image_height);

	const float2 center_ncd = make_float2(0.5, 0.5);
	float3 center_dir = ncd2dir(center_ncd, real_image_width, real_image_height);

	float ecc = acosf(dot(gaze_dir, tile_dir));
	float ecc_center = acosf(dot(tile_dir, center_dir));

	float pooling_rad = alpha * ecc * ecc;
	float angle_min = ecc_center - pooling_rad * 0.5;
	float angle_max = ecc_center + pooling_rad * 0.5;

    // Calculate major and minor axes
	float distance_to_pixel = calculate_distance( (tile_ncd.x-0.5) * real_image_width,  (tile_ncd.y-0.5) * real_image_height, real_viewing_distance);
    float major_axis = (tanf(angle_max) - tanf(angle_min)) * real_viewing_distance;
    float minor_axis = 2.0f * distance_to_pixel * tanf(pooling_rad * 0.5f);
    
    // Calculate area, ensure it's positive, and take its square root
    float area = M_PI * major_axis * minor_axis * 0.25f;
	float real2pix_factor = W / real_image_width;
	float pooling_size = sqrtf(area) * real2pix_factor;

	// Store the result
	float level;
	ps2level(pooling_size, level);
	if (level > (float(fov_num)-0.1)) {
		level = (float(fov_num)-0.1);
	}
	tile_levels[idx] = level;
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void compute_tile_level_infos_cuda(
	int T,
	float* __restrict__ tile_levels,
	const int tile_width_num,
	const int tile_height_num, 
	float* __restrict__ tile_level_gradient_ys,
	float* __restrict__ tile_level_gradient_xs, 
	float* __restrict__ tile_level_min,
	bool* __restrict__ tile_blendings
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;

	// reverse idx to get tile W and H
	int tile_y = idx / tile_width_num;
	int tile_x = idx % tile_width_num;
	float tile_level_f = tile_levels[idx];



	// first, estimate the level gradient using nearby tiles' level
	// then, use the gradient to estimate current pixel's level
	float right_level = -1, left_level = -1, up_level = -1, down_level = -1;
	if (tile_x + 1 < tile_width_num) {
		uint tile_idx = (tile_x + 1) + tile_width_num * tile_y;
		right_level = tile_levels[tile_idx];
	}
	if (tile_x - 1 >= 0) {
		uint tile_idx = (tile_x - 1) + tile_width_num * tile_y;
		left_level = tile_levels[tile_idx];
	}
	if (tile_y + 1 < tile_height_num) {
		uint tile_idx = tile_x + tile_width_num * (tile_y + 1);
		up_level = tile_levels[tile_idx];
	}
	if (tile_y - 1 >= 0) {
		uint tile_idx = tile_x + tile_width_num * (tile_y - 1);
		down_level = tile_levels[tile_idx];
	}

	float level_gradient_x = 0, level_gradient_y = 0;
	if (right_level != -1 && left_level != -1) {
		level_gradient_x = (right_level - left_level) / 2.0f;
	}
	else if (right_level != -1) {
		level_gradient_x = right_level - tile_level_f;
	}
	else if (left_level != -1) {
		level_gradient_x = tile_level_f - left_level;
	}

	if (up_level != -1 && down_level != -1) {
		level_gradient_y = (up_level - down_level) / 2.0f;
	}
	else if (up_level != -1) {
		level_gradient_y = up_level - tile_level_f;
	}
	else if (down_level != -1) {
		level_gradient_y = tile_level_f - down_level;
	}


	float max_delta = 0.5 * (abs(level_gradient_x) + abs(level_gradient_y));

	float tile_min = tile_level_f - max_delta;
	// float tile_max = tile_level_f + max_delta;

	if (tile_min < 0) {
		tile_min = 0;
	}

	// if (tile_max >= float(fov_num) - 0.1) {
	// 	tile_max = float(fov_num) - 0.1;
	// }


	tile_level_min[idx] = tile_min;
	float tile_min_i = float(int(tile_min));


	if( (tile_min - tile_min_i) > start_blend && (tile_min_i < (fov_num - 1))) {
		tile_blendings[idx] = true;
	}
	else{
		tile_blendings[idx] = false;
	}

	tile_level_gradient_ys[idx] = level_gradient_y;
	tile_level_gradient_xs[idx] = level_gradient_x;
}



// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void compute_tile_skips_cuda(
	int T,
	const float* __restrict__ tile_level_min,
	bool* __restrict__ tile_skips,
	const float cur_level
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;

	float tile_min = tile_level_min[idx];

	float lb, hb;

	lb = cur_level - blend_width;
	hb = cur_level + 1;


	bool min_in = tile_min > lb && tile_min < hb;
	// bool max_in = tile_max > lb && tile_max < hb;
	if (!min_in) {
		tile_skips[idx] = true;
	}
	else {
		tile_skips[idx] = false;
	}
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void filter(
	int P,
	const float2* points_xy,
	const uint32_t* offsets,
	const dim3 grid,
	int* radii,
	const float* eigen_lengths, // for OBB
	const float* eigen_vecs, // for OBB
	const bool* tile_skips, //for FOV
	uint32_t* tiles_touched,
	bool* filter_result,
	const int tile_width_num)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		float2 point_image = points_xy[idx];
		getRect(point_image, radii[idx], rect_min, rect_max, grid);

		uint potential_tnum = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

		uint count=0;
		if (potential_tnum == 1)
		{
			uint tile_idx =  rect_min.y  * tile_width_num + rect_min.x;
			bool skip = tile_skips[tile_idx];
			if ( !skip ) 
			{
				count = 1;
			}
		}
		else
		{
			float2 eigenvector1 = { eigen_vecs[idx * 4], eigen_vecs[idx * 4 + 1] };
			float2 eigenvector2 = { eigen_vecs[idx * 4 + 2], eigen_vecs[idx * 4 + 3] };
			float len1 = eigen_lengths[idx * 2];
			float len2 = eigen_lengths[idx * 2 + 1];
			float2 center = point_image;

			// get 4 vertex from center and len1, len2, eigenvector1, eigenvector2
			float d1x = len1 * eigenvector1.x;
			float d1y = len1 * eigenvector1.y;
			float d2x = len2 * eigenvector2.x;
			float d2y = len2 * eigenvector2.y;
			float2 vertexs[4] = { 
				{center.x + d1x + d2x , center.y + d1y + d2y},
				{center.x - d1x + d2x , center.y - d1y + d2y},
				{center.x - d1x - d2x , center.y - d1y - d2y},
				{center.x + d1x - d2x , center.y + d1y - d2y}
			};

			for (int y = rect_min.y; y < rect_max.y; y++)
			{
				for (int x = rect_min.x; x < rect_max.x; x++)
				{
					bool inside;
					// check if the pixel is inside the FOV
					uint tile_idx = y * tile_width_num + x;
					bool skip = tile_skips[tile_idx];
					if ( !skip) {
						inside = true;
					}
					else {
						inside = false;
					}
					if (inside)
					{
						// check if the pixel is inside the OBB
						float px = float(x) * float(BLOCK_X) +  float(BLOCK_X) / 2.0f;
						float py = float(y) * float(BLOCK_Y) + float(BLOCK_Y) / 2.0f;
						OBB_check(px, py, vertexs, center, eigenvector1, eigenvector2, len1, len2, inside);
						if (inside)
						{
							count++;
						}
					}
					filter_result[off] = inside;
					off++;
				}
			}
		}
		tiles_touched[idx] = count;
		if (count == 0)
		{
			radii[idx] = 0;
		}
	}
}


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets_old,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid,
	bool* filter_result)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint32_t off_old = (idx == 0) ? 0 : offsets_old[idx - 1];
		uint2 rect_min, rect_max;
		float2 point_image = points_xy[idx];
		getRect(point_image, radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		uint potential_tnum = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

		if (potential_tnum == 1)
		{
			uint64_t key = rect_min.y * grid.x + rect_min.x;
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = idx;
			off++;
		}
		else
		{
			for (int y = rect_min.y; y < rect_max.y; y++)
			{
				for (int x = rect_min.x; x < rect_max.x; x++)
				{
					bool inside = filter_result[off_old];
					off_old++;
					if (inside)
					{
						uint64_t key = y * grid.x + x;
						key <<= 32;
						key |= *((uint32_t*)&depths[idx]);
						gaussian_keys_unsorted[off] = key;
						gaussian_values_unsorted[off] = idx;
						off++;
					}
				}
			}
		}
	}
}

__global__ void compute_fov_colors(
	const int P, const int D, const int M,
	const float* orig_points,
	const glm::vec3* cam_pos,
	const float* shs,
	const int* __restrict__ radii,
	float3* __restrict__ rgb)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs);
		rgb[idx].x = result.x;
		rgb[idx].y  = result.y;
		rgb[idx].z  = result.z;
	}
}
// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conics, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	obtain(chunk, geom.point_offsets_old, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> OBBFunc,
	std::function<char* (size_t)>  FOVColorFunc,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	//
	const float cur_level,
	const float2 gaze,
	const float alpha,
	bool blending,
	//
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	float* eigen_lengths,
	float* eigen_vecs,
	bool debug)
{

	// --- Start: before preprocessing --- //
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}
	// --- End: before preprocessing --- //

	
	// --- Start: preprocessing --- //
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		shs,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conics,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		eigen_lengths,
		eigen_vecs
	), debug)
	// --- End: preprocessing --- //
	


	// --- Start: PreFiltering --- //
	// Define static pointers
	static float* tile_levels = nullptr;
	static float* tile_level_gradient_ys = nullptr;
	static float* tile_level_gradient_xs = nullptr;
	static float* tile_level_min = nullptr;
	static bool* tile_blendings = nullptr;
	static bool* tile_skips = nullptr;
	static int tile_num = 0;


	// collect the FOV data
	int T = ( (width+15) / 16) * ( (height+15) / 16); // Calculate Tile Number T
	int tile_width_num = (width + 15) / BLOCK_X;
	int tile_height_num = (height + 15) / 16;
    // Allocate memory on the device only if the pointers are null
	if (tile_num != T) {
		tile_num = T;
		// Free the memory if it is not null
		if (tile_levels != nullptr) {
			cudaFree(tile_levels);
		}
		if (tile_level_gradient_ys != nullptr) {
			cudaFree(tile_level_gradient_ys);
		}
		if (tile_level_gradient_xs != nullptr) {
			cudaFree(tile_level_gradient_xs);
		}
		if (tile_level_min != nullptr) {
			cudaFree(tile_level_min);
		}
		if (tile_blendings != nullptr) {
			cudaFree(tile_blendings);
		}
		if (tile_skips != nullptr) {
			cudaFree(tile_skips);
		}
		cudaMalloc(&tile_levels, T * sizeof(float));
		cudaMalloc(&tile_level_gradient_ys, T * sizeof(float));
		cudaMalloc(&tile_level_gradient_xs, T * sizeof(float));
		cudaMalloc(&tile_level_min, T * sizeof(float));
		cudaMalloc(&tile_blendings, T * sizeof(bool));
		cudaMalloc(&tile_skips, T * sizeof(bool));
	}

	if (cur_level == 0) {
		compute_tile_levels_cuda << <(T + 255) / 256, 256 >> > (
			T,
			tile_levels,
			gaze,
			width,
			height,
			tile_width_num,
			alpha
			);
		compute_tile_level_infos_cuda << <(T + 255) / 256, 256 >> > (
				T,
				tile_levels,
				tile_width_num,
				tile_height_num,
				tile_level_gradient_ys,
				tile_level_gradient_xs,
				tile_level_min, 
				tile_blendings
			);
	}




	compute_tile_skips_cuda << <(T + 255) / 256, 256 >> > (
		T,
		tile_level_min,
		tile_skips,
		cur_level
	);

	// Perform FOV OBB Filtering
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets_old, P), debug)
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets_old + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	size_t filter_result_size = num_rendered;
	char* filter_result_chunkptr = OBBFunc(filter_result_size);
	bool* filter_result = reinterpret_cast<bool*>(filter_result_chunkptr);
	// --- End: PreOBB --- //


	// --- Start: Filter --- //
	filter << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.point_offsets_old,
		tile_grid,
		radii,
		eigen_lengths,
		eigen_vecs,
		tile_skips,
		geomState.tiles_touched,
		filter_result,
		tile_width_num);
	CHECK_CUDA(, debug)
	// --- End: Filter --- //



	// Pre Duplicating
	// real num_rendered
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
	// End Pre Duplicating


	// --- Start: Duplicating --- //
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets_old,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		filter_result);
	CHECK_CUDA(, debug)
	// --- End: Duplicating --- //


	// --- Start: Pre Sorting --- //
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	// --- End: Pre Sorting --- //


	// --- Start: Sorting --- //
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)
	// --- End: Sorting --- //



	// --- Start: Pre Identifying tile ranges --- //
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
	// --- End: Pre Identifying tile ranges --- //


	// --- Start: Identifying tile ranges --- //
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)
	// --- End: Identifying tile ranges --- //



	// --- Start: Compute fov color --- //
	size_t fov_color_size =  P * 1 * sizeof(float3);
	// std::cout << "fov_color_size: " << fov_color_size << std::endl;
	// std::cout << "P: " << P << std::endl;
	char* fov_color_chunkptr = FOVColorFunc(fov_color_size);
	float3* fov_colors = reinterpret_cast<float3*>(fov_color_chunkptr);
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	compute_fov_colors << <(P + 255) / 256, 256 >> >  (
		P, D, M, means3D, (glm::vec3*)cam_pos, shs,
		radii,
		fov_colors);
	CHECK_CUDA(, debug)
	// --- End: Compute fov color --- //


	// --- Start: Rendering Blending --- //
	// Let each tile blend its range of Gaussians independently in parallel
	// const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render_blending(
		P,
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		fov_colors,
		//
		tile_level_min,
		tile_level_gradient_ys,
		tile_level_gradient_xs,
		tile_blendings,
		tile_skips,
		cur_level,
		opacities,
		//
		geomState.conics,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color), debug)

	// --- Start: Rendering --- //
	// Let each tile blend its range of Gaussians independently in parallel
	// const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		P,
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		fov_colors,
		//
		tile_level_min,
		tile_blendings,
		tile_skips,
		opacities,
		//
		geomState.conics,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color), debug)

	return num_rendered;
}
