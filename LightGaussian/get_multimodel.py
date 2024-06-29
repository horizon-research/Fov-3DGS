import argparse
import os
import subprocess


def load_txt(filepath):
    """Load level data from a text file."""
    levels = {"Level 0": None, "Level 1": None, "Level 2": None, "Level 3": None}
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("Level 0:"):
                levels["Level 0"] = int(line.split("Level 0:")[1].strip())
            elif line.startswith("Level 1:"):
                levels["Level 1"] = int(line.split("Level 1:")[1].strip())
            elif line.startswith("Level 2:"):
                levels["Level 2"] = int(line.split("Level 2:")[1].strip())
            elif line.startswith("Level 3:"):
                levels["Level 3"] = int(line.split("Level 3:")[1].strip())
    return levels

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Process some folders.")
    parser.add_argument("--pnum_folder", type=str, help="The path to the folder containing pnum")
    parser.add_argument("--output_dir", type=str, help="output folder for the model")
    parser.add_argument("--ps1", type=str, help = "path to ps = 1 model")
    parser.add_argument("--scene", type=str, help="The path to scene")
    args = parser.parse_args()

    # check if the given folder exists
    if not os.path.exists(args.pnum_folder):
        print(f"Error: The pnum_folder {args.pnum_folder} does not exist.")
        return
    

    print(f"Processing scene: {args.scene}")
    
    pnum_txt = os.path.join(args.pnum_folder, f"{args.scene}.txt")
    
    pnums = []
    
    levels = load_txt(pnum_txt)
    pnums.append(levels.get("Level 0", None))
    pnums.append(levels.get("Level 1", None))
    pnums.append(levels.get("Level 2", None))
    pnums.append(levels.get("Level 3", None))
    
    
    # mkdir {args.output_dir}/L0
    if not os.path.exists(f"{args.output_dir}/L0"):
        os.makedirs(f"{args.output_dir}/L0")
        
    
    
    print(f"copy ps1 to dest: ")
    cp_command = f"cp -r {args.ps1} {args.output_dir}/L0"
    print(f"Running command: {cp_command}")
    subprocess.run(cp_command, shell=True)
    
    
    
    ps1_path = os.path.join(args.ps1, "chkpnt55000.pth")

    for i in range(1, 4):
        target_pnum = pnums[i]
        print("Pruning Layer ", i)
        target_ratio = target_pnum / pnums[0]
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
