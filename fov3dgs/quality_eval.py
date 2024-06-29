import os
import subprocess
import json
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
os.environ['MKL_THREADING_LAYER'] = 'GNU'


def run_command(command):
    """Run a shell command and wait for it to complete."""
    subprocess.run(command, shell=True, check=True)
def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as file:
        return json.load(file)


eval_folder = "./full_eval_results"
m360_base = "../dataset"
db_base = "../dataset"
tat_base = "../dataset"

# ours-Q
ps1_method_dir = "4_12_0.01_1.0/1_PS1_4_12"
fov_method_dir = "4_12_0.01_1.0/"
method = "ours-Q"
iterations = 55000

# mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
# mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
# tanks_and_temples_scenes = ["truck", "train"]
# deep_blending_scenes = ["drjohnson", "playroom"]
mipnerf360_outdoor_scenes = ["bicycle"]
mipnerf360_indoor_scenes = []
tanks_and_temples_scenes = []
deep_blending_scenes = []
all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

# make output directory
output_folder = os.path.join(eval_folder, method)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


all_metrics = dict()
all_metrics["m360"] = dict()
all_metrics["tat"] = dict()
all_metrics["db"] = dict()

metrics = ["SSIM", "PSNR", "LPIPS", "HVS FOV"]
# init SSIM, PSNR, LPIPS, HVS FOV with empty list
for metric in metrics:
    all_metrics["m360"][metric] = []
    all_metrics["tat"][metric] = []
    all_metrics["db"][metric] = []

def main():
    for scene in all_scenes:
        if scene in mipnerf360_outdoor_scenes:
            base = m360_base
            scene_type = "m360"
        elif scene in mipnerf360_indoor_scenes:
            base = m360_base
            scene_type = "m360"
        elif scene in tanks_and_temples_scenes:
            base = tat_base
            scene_type = "tat"
        elif scene in deep_blending_scenes:
            base = db_base
            scene_type = "db"
        else:
            raise ValueError(f"Unknown scene {scene}")
        
        scene_dir = os.path.join(base, scene)
        output_json = os.path.join(output_folder, scene + "_quality.json")
        output_json_per = os.path.join(output_folder, scene + "_quality_per.json")
        ps1_method_path = os.path.join(scene_dir, ps1_method_dir)
    
        render_command = f"python3 render.py -s {scene_dir} -m {ps1_method_path} --eval --skip_train --iteration {iterations}"
        run_command(render_command)


        # second, measure the ps1  and fov quality
        ps1_imgs_path = os.path.join(ps1_method_path, "test", f"ours_{iterations}")

        metrics_command = f"python3 quality_metrics.py -ps1 {ps1_imgs_path} -o {output_json} -o2 {output_json_per}"
        run_command(metrics_command)

        # third, read from the jetson and add to m360_all, tat_all, db_all

if __name__ == "__main__":

    main()

