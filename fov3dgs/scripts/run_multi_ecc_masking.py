import subprocess
# use argparser
import argparse
import os
import json

def run_masking_command(pooling_size, scene, target_loss, pretrain_ply, trial_name, is_first, prune_iters, adaptation_iters, args, init_idx):
    # Build the base command with dynamic parts depending on whether it's the first run
    # import ipdb; ipdb.set_trace()   
    base_command = f"""
    python3 metric_mask_learn.py -s "{scene}" -i {args.images_set} \
        -m "{scene}" --eval \
        --pretrain_ply "{pretrain_ply}" \
        --hvs_loss_type L1 \
        --pooling_size {pooling_size} \
        --target_loss {target_loss} \
        --pruning_iters {prune_iters} \
        --final_adaptation_iters {adaptation_iters} \
        --trial_name "{trial_name}" \
        --metric {args.masking_metric} \
    """
    # Add --only_train_shs_dc for subsequent runs
    additional_flags = ""
        
    # import ipdb; ipdb.set_trace()
    if init_idx:
        additional_flags += " --init_index "

    if args.monitor_val:
        additional_flags += " --monitor_val"

    # Complete command with additional flags
    command = f"{base_command} {additional_flags}"

    print(command)
    # import ipdb; ipdb.set_trace()
    result = subprocess.run(command, shell=True)
    print(result)


def create_folder(scene, layernum, max_pooling_size, target_loss_scale, ps1_loss_scale):
    folder_name = f"{layernum}_{max_pooling_size}_{ps1_loss_scale}_{target_loss_scale}"
    full_path = os.path.join(scene, folder_name)
    
    # Create the folder exist ok
    os.makedirs(full_path, exist_ok=True)

    return full_path, folder_name

# Example usage within a larger script context
def main():
    # take arguments, scene, target_loss, initial_pretrain_ply, max_pooling_size from command line, layernum from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, help="Scene path")
    parser.add_argument("--target_loss_scale", type=float, help="Target loss scale")
    parser.add_argument('--path_to_pretrain_trial', type=str, default = None)

    parser.add_argument('--path_to_ps1_trial', type=str, default = None)

    # pruning settings
    parser.add_argument('--masking_metric', type=str, default = "surface")

    # FOV parameters
    parser.add_argument("--max_pooling_size", type=int, help="Max pooling size")
    parser.add_argument("--layernum", type=int, help="Layer number")
    parser.add_argument("--budget", type=int, help="Budget for pruning, total iteration number")

    # add a store true arg
    parser.add_argument("--monitor_val", action="store_true", default=False)
    parser.add_argument("--ps1_iterations", type=int)

    parser.add_argument('--images_set', type=str, default = "")
    # ps1_loss_scale
    parser.add_argument("--ps1_loss_scale", type=str, help="")


    
    args = parser.parse_args()


    if args.monitor_val:
        args.original_train_loss = f"{args.path_to_pretrain_trial}/test_results.json"
    else:
        args.original_train_loss = f"{args.path_to_pretrain_trial}/train_results.json"

    # check if the original training loss txt exists
    if not os.path.exists(args.original_train_loss):
        # if not, run:
        if args.monitor_val:
            cmd = f"python3 render.py -s {args.scene} -m {args.path_to_pretrain_trial}  --eval --iteration {args.ps1_iterations} --skip_train"
        else:
            cmd = f"python3 render.py -s {args.scene} -m {args.path_to_pretrain_trial}  --eval --iteration {args.ps1_iterations} --skip_test"
        print("Original training loss txt not found, running render.py to generate it.")
        print(cmd)
        subprocess.run(cmd, shell=True)
        print("Original training loss txt not found, running hvs_metric to generate it.")
        if args.monitor_val:
            cmd = f"python3 hvs_metrics.py -m {args.path_to_pretrain_trial} -s test"
        else:
            cmd = f"python3 hvs_metrics.py -m {args.path_to_pretrain_trial} -s train"
        subprocess.run(cmd, shell=True)


    # Use a regular expression to find the "Average L1 HVS Loss" in the text
    with open(args.original_train_loss, 'r') as file:
        text = file.read()
        data = json.loads(text)
    hvs_uniform_value = data[f"ours_{args.ps1_iterations}"]["HVS Uniform"]

    # print 
    print(f"Average HVS Loss: {hvs_uniform_value}")
    args.target_loss = float(hvs_uniform_value) * args.target_loss_scale

    full_folder_path, folder_name= create_folder(args.scene, args.layernum, args.max_pooling_size, args.target_loss_scale, args.ps1_loss_scale)

    # generate iternum according to layernum and budget
    assert args.layernum > 1    
    iternum = args.budget / (args.layernum -1)

    # get sqrt of max pooling size
    sqrt_max_pooling_size = args.max_pooling_size**0.5
    # get interval according to layernum
    interval = float( (sqrt_max_pooling_size-1) / (args.layernum -1))


    # Assuming args is already defined and properly configured
    # import ipdb; ipdb.set_trace()   
    for i in range(args.layernum):
        pooling_size = 1 + interval * i
        pooling_size = round(pooling_size**2)
        is_first = (i == 0)  # Check if it's the first iteration
        if (is_first):
            src_trial_name = f"{args.path_to_ps1_trial}"
            target_trial_name = f"{full_folder_path}/{pooling_size}_PS1_{args.layernum}_{args.max_pooling_size}"
            if not os.path.exists(target_trial_name):
                cmd = f"cp -r {src_trial_name} {target_trial_name}"
                subprocess.run(cmd, shell=True)
            pretrain_ply = f"{target_trial_name}/point_cloud/iteration_{args.ps1_iterations}/point_cloud.ply"
        else:
            # Run the pruning process with the current parameters
            prune_iters = round(iternum * 0.8)
            adaptation_iters = round(iternum * 0.2)
            trial_name = f"{folder_name}/{pooling_size}_{prune_iters}-{adaptation_iters}_{args.layernum}_{args.masking_metric}"
            if i==1:
                init_idx = True
            else:
                init_idx = False
            # check if trial_name exists
            if not os.path.exists(f"{args.scene}/{trial_name}/point_cloud/iteration_{prune_iters + adaptation_iters}/point_cloud.ply"):
                run_masking_command(pooling_size, args.scene, args.target_loss, pretrain_ply, trial_name, is_first, prune_iters, adaptation_iters, args, init_idx)
            # For simplicity, assuming model paths are stored in a specific format
            pretrain_ply = f"{args.scene}/{trial_name}/point_cloud/iteration_{prune_iters + adaptation_iters}/point_cloud.ply"



if __name__ == "__main__":
    main()