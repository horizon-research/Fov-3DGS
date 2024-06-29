# HVS functions for Foveated Training and Blending

We use code from https://github.com/kaanaksit/odak/tree/0.1.9/odak/learn/perception, and make our changes.

## Usages in Our Usecase
- Create and use the Uniform Loss For Training
```python
# Create the Loss 
uniform_hvs_loss = MetamericLossUniform(n_pyramid_levels=5, n_orientations=4, pooling_size=64, device="cuda", loss_type="L1") 
# Use the Loss
if need_resize:
    gt_image = F.interpolate(gt_image.unsqueeze(0), size=(required_height, required_width), mode='bilinear', align_corners=False).squeeze(0)
    image = F.interpolate(image.unsqueeze(0), size=(required_height, required_width), mode='bilinear', align_corners=False).squeeze(0)
else:
    gt_image = gt_image.unsqueeze(0)
    image = image.unsqueeze(0)
```

- Create and use the foveated Loss for evaluation
```python
# Create the Loss 
fov_hvs_loss = MetamericLoss(real_image_width=1.4, real_viewing_distance=0.7, equi=False, alpha=0.2, mode="quadratic", loss_type="L1", use_l2_foveal_loss=False, n_pyramid_levels=5, n_orientations=4)
# Use the Loss
hvs_loss_term = fov_hvs_loss(my_image, gt_image, gaze=[0.5, 0.5])
```

- Create and use the metamer generator for blending
```python
# Create the Metamer Generator
gen_metamer = MetamerMSELoss(real_image_width=1.4, real_viewing_distance=0.7, equi=False, alpha=0.2, mode="quadratic",
                          n_pyramid_levels=5, n_orientations=4, device="cuda")
# Get the Metamer
# note that input image is srgb between (0-1) 
metamer = gen_metamer.gen_metamer(my_image, [0.5, 0.5])
metamer = (metamer * 255)
metamer = np.clip(metamer, 0, 255)
metamer = metamer.astype(np.uint8)
```

## Modifications
- We add L1 for optimization of HVS Loss, and use it in training, it give us better result. 
See [odak_perception/metameric_loss_uniform.py#L93](odak_perception/metameric_loss_uniform.py#L93) for example.
- We forbid the padding inside the function, forcing user feed in image with valid size, to prevent potential image geometry changing issue. See [odak_perception/spatial_steerable_pyramid.py#L27](odak_perception/spatial_steerable_pyramid.py#L27)


