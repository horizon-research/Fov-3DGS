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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <math_constants.h> // For CUDART_NAN
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__  __device__ void atomicMaxFloat(float* address, float value) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
}




__forceinline__ __device__ void normalize(float2& vec) {
    float norm = rsqrtf(vec.x * vec.x + vec.y * vec.y);
    vec.x *= norm;
    vec.y *= norm;
}


__forceinline__ __device__ float dotProduct(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__forceinline__ __device__ void OBB_check(float tile_px, float tile_py, float2* vertexs, float2 center, float2 norm_v1, float2 norm_v2, float len1, float len2, bool &inside) {
	// check x axis
	// normalize vertexs to tile center
	float2 relative_point_vertexs[4] = {
		{vertexs[0].x - tile_px, vertexs[0].y - tile_py},
		{vertexs[1].x - tile_px, vertexs[1].y - tile_py},
		{vertexs[2].x - tile_px, vertexs[2].y - tile_py},
		{vertexs[3].x - tile_px, vertexs[3].y - tile_py}
	};
	// get min and max on x
	float px_min = relative_point_vertexs[0].x;
	float px_max = px_min;
	float x = 0;

	for (int i = 1; i < 4; i++) {
		x = relative_point_vertexs[i].x;
		px_min = fminf(px_min, x);
		px_max = fmaxf(px_max, x);
	}

	// // check if the tile is outside the OBB
	if (px_max < -8.0f || px_min > 8.0f) {
		inside =  false;
		return;
	}

	// // check y axis
	float py_min = relative_point_vertexs[0].y;
	float py_max = py_min;
	float y = 0;

	for (int i = 1; i < 4; i++) {
		y = relative_point_vertexs[i].y;
		py_min = fminf(py_min, y);
		py_max = fmaxf(py_max, y);
	}
	// check if the tile is outside the OBB
	if (py_max < -8.0f || py_min > 8.0f) {
		inside = false;
		return;
	}

	// // make 4 points 
    // // Precomputed tile vertices (relative to the center)
 	float2 relative_tile_vertexs[4] = {
        {tile_px + 8.0f - center.x, tile_py + 8.0f - center.y},  // Top right corner
        {tile_px - 8.0f - center.x, tile_py + 8.0f - center.y},  // Top left corner
        {tile_px - 8.0f - center.x, tile_py - 8.0f - center.y},  // Bottom left corner
        {tile_px + 8.0f - center.x, tile_py - 8.0f - center.y}   // Bottom right corner
    };

	// // check along v1
    float tile_min1 = dotProduct(relative_tile_vertexs[0], norm_v1);
    float tile_max1 = tile_min1;
	float dot1 = 0;

    for (int i = 1; i < 4; i++) {
        dot1 = dotProduct(relative_tile_vertexs[i], norm_v1);
        tile_min1 = fminf(tile_min1, dot1);
        tile_max1 = fmaxf(tile_max1, dot1);
    }

    // Point's min and max along v1 are known to be len1 and -len1
    if (len1 < tile_min1 || -len1 > tile_max1) {
		inside = false;
		return;
    }


    // // Check along v2
    float tile_min2 = dotProduct(relative_tile_vertexs[0], norm_v2);
    float tile_max2 = tile_min2;
	float dot2 = 0;


    for (int i = 1; i < 4; i++) {
        dot2 = dotProduct(relative_tile_vertexs[i], norm_v2);
        tile_min2 = fminf(tile_min2, dot2);
        tile_max2 = fmaxf(tile_max2, dot2);
    }

    if (len2 < tile_min2 || -len2 > tile_max2) {
		inside = false;
		return;
    }

	inside = true;
	return;
}



__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif