#!/bin/bash

method="MMFR-Q"
mkdir -p ./fps/"$method"


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
    python3 render_compose_gazes_fps_mmfr.py --base_folder ../LightGaussian/MMFR/ours-Q/"$scene"/ --layer_num 4 --skip_train -m  ../LightGaussian/MMFR/ours-Q/"$scene"/L0/1_PS1_4_12/  -s ../dataset/"$scene" > ./fps/"$method"/"$scene".txt
done

