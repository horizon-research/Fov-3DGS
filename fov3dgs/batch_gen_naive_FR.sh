#!/bin/bash

method="ours-Q-1gaze"
mkdir -p ../origin_3dgs/fps/"$method"


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
    python3 gen_naive_FR.py --folder_base ../dataset/"$scene"/4_12_0.01_1.0/ --max_pooling_size 12 --layer_num 4 --metric surface
done


