import os
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--skip_masking", action="store_true")
parser.add_argument("--skip_ps1", action="store_true")
args, _ = parser.parse_known_args()

# mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
# mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
# tanks_and_temples_scenes = ["truck", "train"]
# deep_blending_scenes = ["drjohnson", "playroom"]

mipnerf360_outdoor_scenes = ["bicycle"]
mipnerf360_indoor_scenes = []
tanks_and_temples_scenes = []
deep_blending_scenes = []

scenes = []
scenes.extend(mipnerf360_outdoor_scenes)
scenes.extend(mipnerf360_indoor_scenes)
scenes.extend(tanks_and_temples_scenes)
scenes.extend(deep_blending_scenes)


script_dir = os.path.dirname(os.path.abspath(__file__))
m360_base_dir = os.path.abspath(os.path.join(script_dir, "../dataset/"))
tat_base_dir = os.path.abspath(os.path.join(script_dir, "../dataset/"))
db_base_dir = os.path.abspath(os.path.join(script_dir, "../dataset/"))

pruning_relax_ratios = [0.01]
pruning_metrics = ["max_comp_efficiency"]
masking_metrics = ["surface"]
obb_finetune_iterations = 35000 # 30000~35000


prune_iters = 50000
hvs_finetune_iterations = prune_iters + 5000
layer_num = 4
max_pooling_size = 12
masking_budget = 22500

use_scale_decays = [True]

# Script loop
for scene in scenes:
    for use_scale_decay in use_scale_decays:
        if scene in mipnerf360_outdoor_scenes:
            image_set = "images_4"
        elif scene in mipnerf360_indoor_scenes:
            image_set = "images_2"
        else:
            image_set = "images"

        if scene in mipnerf360_outdoor_scenes:
            base_scene_path = m360_base_dir
        elif scene in mipnerf360_indoor_scenes:
            base_scene_path = m360_base_dir
        elif scene in tanks_and_temples_scenes:
            base_scene_path = tat_base_dir
        else:
            base_scene_path = db_base_dir

        path_to_scene = os.path.join(base_scene_path, scene)

        if not args.skip_ps1:
            # OBB finetune
            print(f"Processing scene for eff_finetune: {scene}")
            eff_finetune_command = [
                "python3", "eff_finetune.py",
                "-s", f"{path_to_scene}/",
                "-m", f"{path_to_scene}/ms_d_ft",
                "--checkpoint_iterations", "35000",
                "--start_checkpoint", f"{path_to_scene}/ms_d/chkpnt30000.pth",
                "-i", image_set,
                "--iteration", "35000",
                "--checkpoint_iterations", "35000",
                "--eval"
            ]
            eff_finetune_path = os.path.join(path_to_scene, "ms_d_ft", "chkpnt35000.pth")
            if not os.path.exists(eff_finetune_path):
                print(f"Running command: {eff_finetune_command}")
                subprocess.run(eff_finetune_command)

        if not args.skip_ps1:
            # Pruning
            for relax_ratio in pruning_relax_ratios:
                for pruning_metric in pruning_metrics:
                    path_to_pretrain = os.path.join(path_to_scene, "ms_d_ft")
                    
                    pruning_command = (
                        f"python3 scripts/run_prune.py "
                        f"--pruning_metric \"{pruning_metric}\" "
                        f"--scene {path_to_scene} --relax_ratio \"{relax_ratio}\" "
                        f"--iterations {prune_iters} --monitor_val "
                        f"--path_to_pretrain_trial \"{path_to_pretrain}\" --images_set {image_set} "
                    )
                    if use_scale_decay:
                        pruning_command += "--use_scale_decay "
                    
                    print(f"Running command: {pruning_command}")

                    check_point = os.path.join(path_to_scene, f"mon2_montest=True_sd={use_scale_decay}_{pruning_metric}_{relax_ratio}/chkpnt{prune_iters}.pth")
                    if not os.path.exists(check_point):
                        os.system(pruning_command)

        if not args.skip_ps1:
            # hvs_finetune
            for pruning_metric in pruning_metrics:
                for relax_ratio in pruning_relax_ratios:
                    pruned_folder = f"mon2_montest=True_sd={use_scale_decay}_{pruning_metric}_{relax_ratio}"
                    hvs_path = os.path.join(path_to_scene, f"{pruned_folder}/hvs_reshape_ps1")
                    prune_path = os.path.join(path_to_scene, pruned_folder)

                    hvs_finetune_command = [
                        "python3", "eff_finetune.py",
                        "-s", f"{path_to_scene}/",
                        "-m", f"{hvs_path}",
                        "--checkpoint_iterations",f"{hvs_finetune_iterations}",
                        "--start_checkpoint", f"{prune_path}/chkpnt{prune_iters}.pth",
                        "-i", image_set,
                        "--iteration",f"{hvs_finetune_iterations}",
                        "--eval", "--hvs_ft"
                    ]
                            
                    print(f"Running command: {hvs_finetune_command}")
                    check_point = os.path.join(hvs_path, f"chkpnt{hvs_finetune_iterations}.pth")
                    if not os.path.exists(check_point):
                        subprocess.run(hvs_finetune_command)

        if not args.skip_masking and use_scale_decay == True:
            for relax_ratio in pruning_relax_ratios:
                for masking_metric in masking_metrics:
                    pruned_folder = f"mon2_montest=True_sd={use_scale_decay}_max_comp_efficiency_{relax_ratio}"
                    path_to_ps1 = f"{pruned_folder}/hvs_reshape_ps1"
                    path_to_ps1 = os.path.join(path_to_scene, path_to_ps1)
                    path_to_pretrain = path_to_ps1

                    command = (
                        f"python3 scripts/run_multi_ecc_masking.py "
                        f"--masking_metric \"{masking_metric}\" "
                        f"--scene {path_to_scene} --target_loss_scale \"{1.0}\"  --ps1_loss_scale \"{relax_ratio}\" "
                        f"--max_pooling_size {max_pooling_size} --layernum {layer_num} --budget {masking_budget} "
                        f"--path_to_pretrain_trial \"{path_to_pretrain}\" "
                        f"--path_to_ps1_trial \"{path_to_ps1}\" --monitor_val --ps1_iterations {hvs_finetune_iterations} --images_set {image_set}"
                    )
                    print(f"Running command: {command}")
                    os.system(command)

print("All scenes processed.")
