#!/bin/bash

method="naiveFR"
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
    python3 render_compose_gazes_fps_naive.py --base_folder ../dataset/"$scene"/4_12_0.01_1.0/  --layer_num 4 --max_pooling_size 12 --skip_train --eval -m ../dataset/"$scene"/4_12_0.01_1.0/ -s ../dataset/"$scene" > ./fps/"$method"/"$scene".txt
done

