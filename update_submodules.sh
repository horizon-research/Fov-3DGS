#!/bin/bash

# Directory containing the submodules
SUBMODULE_DIR="fov3dgs/submodules"

# Change to the submodule directory
cd $SUBMODULE_DIR

# Loop through each submodule and initialize and update it
for dir in */ ; do
    if [ -d "$dir" ]; then
        echo "Updating submodule in $dir"
        cd "$dir"
        pip install .
        cd ..
    fi
done

echo "All submodules have been updated."
