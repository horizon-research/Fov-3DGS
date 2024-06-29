workdir=$1
pooling_size=$2
iters=$3

# check if /train exists
if [ ! -d "$workdir"/train ]; then
    python3 render.py --iteration "$iters" -s ../m360/bicycle --eval -m "$workdir"
fi

python3 hvs_loss_calc.py \
    --folder1 "$workdir"/train/ours_"$iters"/gt \
    --folder2 "$workdir"/train/ours_"$iters"/renders \
    --log_file "$workdir"/train/ours_"$iters"/train_loss_eval.txt \
    --n_pyramid_levels 5 \
    --n_orientations 6\
    --avg_pooling_hvs_pyramid \
    --pooling_size "$pooling_size" 
