#!/bin/bash

method="ours-Q"
mkdir -p "./pnum/$method"


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
    python3 pnum_analyzer.py --folder_base ../dataset/"$scene"/4_12_0.01_1.0/ --max_pooling_size 12 --layer_num 4 --metric surface > ./pnum/"$method"/"$scene".txt
done
