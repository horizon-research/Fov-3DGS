
from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer_original
from diff_gaussian_rasterization_pcheck_obb import GaussianRasterizer as GaussianRasterizer_pcheck_obb
from diff_gaussian_rasterization_pcheck_obb_max import GaussianRasterizer as GaussianRasterizer_pcheck_obb_max
from diff_gaussian_rasterization_pcheck_obb_sum import GaussianRasterizer as GaussianRasterizer_pcheck_obb_sum

from diff_gaussian_rasterization_pcheck_obb_loss_weighted_max_count import GaussianRasterizer as GaussianRasterizer_pcheck_obb_loss_weighted_max_count



def get_gs_rasterizer(cuda_type, raster_settings):
    if cuda_type == "original":
        return GaussianRasterizer_original(raster_settings=raster_settings)
    elif cuda_type == "pcheck_obb":
        return GaussianRasterizer_pcheck_obb(raster_settings=raster_settings)
    elif cuda_type == "pcheck_obb_max":
        return GaussianRasterizer_pcheck_obb_max(raster_settings=raster_settings)
    elif cuda_type == "pcheck_obb_sum":
        return GaussianRasterizer_pcheck_obb_sum(raster_settings=raster_settings) 
    elif cuda_type == "pcheck_obb_loss_weighted_max_count":
        return GaussianRasterizer_pcheck_obb_loss_weighted_max_count(raster_settings=raster_settings)
    else:
        raise ValueError("Invalid cuda type: {}".format(cuda_type))