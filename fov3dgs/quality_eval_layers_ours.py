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


eval_folder = "./layers_eval_results"
m360_base = "../m360"
db_base = "../db"
tat_base = "../tandt"

# ours-Q
method_dir_list = ["4_12_0.01_1.0/1_PS1_4_12",
                   "4_12_0.01_1.0/3_6000-1500_4_surface",
                   "4_12_0.01_1.0/7_6000-1500_4_surface",
                   "4_12_0.01_1.0/12_6000-1500_4_surface"
]
method = "ours-Q"
iterations_list = [ 55000, 
               7500,
               7500,
               7500
            ]

ps_list = [1, 3, 7, 12]

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]
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

def main():
    for scene in all_scenes:
        if scene in mipnerf360_outdoor_scenes:
            base = m360_base
        elif scene in mipnerf360_indoor_scenes:
            base = m360_base
        elif scene in tanks_and_temples_scenes:
            base = tat_base
        elif scene in deep_blending_scenes:
            base = db_base
        else:
            raise ValueError(f"Unknown scene {scene}")
        
        scene_dir = os.path.join(base, scene)


        for i in range(len(method_dir_list)):
            if scene in ["bicycle"]:
                continue
            method_path = os.path.join(scene_dir, method_dir_list[i])
            iterations = iterations_list[i]
            render_command = f"python3 render.py -s {scene_dir} -m {method_path} --eval --skip_train --iteration {iterations}"
            run_command(render_command)


        for i in range(len(method_dir_list)):
            method_path = os.path.join(scene_dir, method_dir_list[i])
            iterations = iterations_list[i]
            ps = ps_list[i]
            result_path = os.path.join(method_path, "test", f"ours_{iterations}")
            
            output_json = os.path.join(output_folder, scene + f"_{ps}.json")
            output_json_per = os.path.join(output_folder, scene + f"_{ps}_per.json")
            metrics_command = f"python3 quality_metrics_layer.py --result_folder {result_path} -ps {ps} -o {output_json} -o2 {output_json_per}"
            run_command(metrics_command)

        # third, read from the jetson and add to m360_all, tat_all, db_all
    



if __name__ == "__main__":

    main()

