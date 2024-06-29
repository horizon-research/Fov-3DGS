import argparse
import os
import subprocess

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Process some folders.")
    parser.add_argument("--output_dir", type=str, help="output folder for the model")
    parser.add_argument("--ps1", type=str, help = "path to ps = 1 model")
    parser.add_argument("--scene", type=str, help="The path to scene")
    args = parser.parse_args()

    

    print(f"Processing scene: {args.scene}")
    

    
    ratios = [1.0, 0.21, 0.17, 0.09, 0.03]
    
    
    # mkdir {args.output_dir}/L0
    if not os.path.exists(f"{args.output_dir}/L0"):
        os.makedirs(f"{args.output_dir}/L0")
        
    
    
    print(f"copy ps1 to dest: ")
    cp_command = f"cp -r {args.ps1} {args.output_dir}/L0"
    print(f"Running command: {cp_command}")
    subprocess.run(cp_command, shell=True)
    
    
    
    ps1_path = os.path.join(args.ps1, "chkpnt30000.pth")

    for i in range(1, 5):
        print("Pruning Layer ", i)
        target_ratio = ratios[i]
        pruned_ratio = 1 - target_ratio
        print("Target Ratio: ", target_ratio)
        model_path = os.path.join(args.output_dir, f"L{i}")
        # mkdir model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        prune_cmd = f"bash scripts/run_prune_finetune.sh {pruned_ratio} {ps1_path} {args.scene} {model_path}"
        
        subprocess.run(prune_cmd, shell=True)


if __name__ == "__main__":
    main()
