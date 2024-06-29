#!/bin/bash

method="ours-Q"
mkdir -p ./MMFR/"$method"




scenes=(
    "bicycle"
    # "flowers"
    # "garden"
    # "stump"
    # "treehill"
    # "counter"
    # "kitchen"
    # "bonsai"
    # "room"
)


for scene in "${scenes[@]}"; do
    python3 get_multimodel.py --pnum_folder ../fov3dgs/pnum/"$method" --output_dir ./MMFR/"$method"/"$scene"/ --scene "$scene" --ps1 ../dataset/"$scene"/4_12_0.01_1.0/1_PS1_4_12
done
