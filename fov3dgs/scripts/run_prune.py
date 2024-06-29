import subprocess
# use argparser
import argparse
import os
import json

# Example usage within a larger script context
def main():
    # take arguments, scene, target_loss, initial_pretrain_ply, max_pooling_size from command line, layernum from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, help="Scene path")
    # target loss
    parser.add_argument("--relax_ratio", type=float, help="Target loss relaxation ratio (100% + / - relax_ratio)")
    # path of pretrain trial
    parser.add_argument('--path_to_pretrain_trial', type=str, default = None)
    # pruning settings
    parser.add_argument('--pruning_metric', type=str, default = "max_comp_efficiency")
    # FOV parameters
    parser.add_argument("--iterations", type=int, help="Budget for pruning, total iteration number")

    parser.add_argument("--monitor_val", action="store_true", default=False)

    parser.add_argument('--images_set', type=str, default = "")

    parser.add_argument("--use_scale_decay", action="store_true", default=False)

    args = parser.parse_args()

    if args.monitor_val:
        args.original_train_loss = f"{args.path_to_pretrain_trial}/test_results.json"
    else:
        args.original_train_loss = f"{args.path_to_pretrain_trial}/train_results.json"

    args.pretrained_pc = f"{args.path_to_pretrain_trial}/point_cloud/iteration_35000/point_cloud.ply"

    # check if the original training loss txt exists
    if not os.path.exists(args.original_train_loss):
        if args.monitor_val:
            cmd = f"python3 render.py -s {args.scene} -m {args.path_to_pretrain_trial}  --eval --iteration 35000 --skip_train"
        else:
            cmd = f"python3 render.py -s {args.scene} -m {args.path_to_pretrain_trial}  --eval --iteration 35000 --skip_test"
        print("Original training loss json not found, running render.py to generate images.")
        print(cmd)
        subprocess.run(cmd, shell=True)
        print("Original training loss json not found, running hvs_metric to generate it.")
        if args.monitor_val:
            cmd = f"python3 hvs_metrics.py -m {args.path_to_pretrain_trial} -s test"
        else:
            cmd = f"python3 hvs_metrics.py -m {args.path_to_pretrain_trial} -s train"
        subprocess.run(cmd, shell=True)


    # find the "Average L1 HVS Loss" in the text
    with open(args.original_train_loss, 'r') as file:
        text = file.read()
        data = json.loads(text)
    hvs_uniform_value = data["ours_35000"]["HVS Uniform"]
    ssim = data["ours_35000"]["SSIM"]
    psnr = data["ours_35000"]["PSNR"]

    # print 
    print(f"Average HVS Loss: {hvs_uniform_value}")
    print(f"Average SSIM: {ssim}")

    args.target_hvs = float(hvs_uniform_value) * (1 + args.relax_ratio)
    args.target_ssim = float(ssim)  * (1 - args.relax_ratio)
    args.target_psnr = float(psnr)  * (1 - args.relax_ratio) 


    prune_iterations = round(args.iterations * 0.9)
    adaptation_iters = args.iterations - prune_iterations
    
    base_command = f"""
    python3 prune.py -s "{args.scene}" -i {args.images_set} \
        -m "{args.scene}" --eval \
        --pretrain_ply "{args.pretrained_pc}" \
        --pooling_size {1} \
        --target_hvs {args.target_hvs} \
        --target_ssim {args.target_ssim} \
        --target_psnr {args.target_psnr} \
        --pruning_iters {prune_iterations} \
        --final_adaptation_iters {adaptation_iters} \
        --trial_name "mon2_montest={args.monitor_val}_sd={args.use_scale_decay}_{args.pruning_metric}_{args.relax_ratio}" \
        --position_lr_init_scale 0.1 \
        --metric {args.pruning_metric} \
    """
    additional_flags = ""
    if args.monitor_val:
        additional_flags += " --monitor_val"

    if args.use_scale_decay:
        additional_flags += " --use_scale_decay"

    # Complete command with additional flags
    command = f"{base_command} {additional_flags}"
    result = subprocess.run(command, shell=True)
    print(result)

   



if __name__ == "__main__":
    main()