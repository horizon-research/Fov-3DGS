#!/bin/bash
pruned_ratio=$1
ps1_path=$2
scene=$3
model_path=$4
# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=10000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
   $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=6049


function string_in_list {
  local string="$1"
  shift
  local list=("$@")

  for item in "${list[@]}"; do
    if [[ "$item" == "$string" ]]; then
      return 0
    fi
  done
  return 1
}

# Define the lists
mipnerf360_outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill")
mipnerf360_indoor_scenes=("room" "counter" "kitchen" "bonsai")
tanks_and_temples_scenes=("truck" "train")
deep_blending_scenes=("drjohnson" "playroom")
# Only one dataset specified here, but you could run multiple
declare -a run_args=(
    $scene
  )


# Prune percentages and corresponding decays, volume power
declare -a prune_percents=($pruned_ratio)
# decay rate for the following prune. The 2nd prune would prune out 0.5 x 0.6 = 0.3 of the remaining gaussian
declare -a prune_decays=(1)
# The volumetric importance power. The higher it is the more weight the volume is in the Global significant
declare -a v_pow=(0.1)

# prune type, by default the Global significant listed in the paper, but there are other option that you can play with
declare -a prune_types=(
  "v_important_score"
  # "important_score"
  # "count"
  )


# Check that prune_percents, prune_decays, and v_pow arrays have the same length
if [ "${#prune_percents[@]}" -ne "${#prune_decays[@]}" ] || [ "${#prune_percents[@]}" -ne "${#v_pow[@]}" ]; then
  echo "The lengths of prune_percents, prune_decays, and v_pow arrays do not match."
  exit 1
fi

# Loop over the arguments array
for arg in "${run_args[@]}"; do
# Check in each list
  search_string="$arg"
  if string_in_list "$search_string" "${mipnerf360_outdoor_scenes[@]}"; then
    base="../dataset"
    img_set="images_4"
  elif string_in_list "$search_string" "${mipnerf360_indoor_scenes[@]}"; then
    base="../dataset"
    img_set="images_2"
  elif string_in_list "$search_string" "${tanks_and_temples_scenes[@]}"; then
    base="../dataset"
    img_set="images"
  elif string_in_list "$search_string" "${deep_blending_scenes[@]}"; then
    base="../dataset"
    img_set="images"
  else
    echo "$search_string not found in any list"
  fi

  for i in "${!prune_percents[@]}"; do
    prune_percent="${prune_percents[i]}"
    prune_decay="${prune_decays[i]}"
    vp="${v_pow[i]}"

    for prune_type in "${prune_types[@]}"; do
      # Wait for an available GPU
      while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting prune_finetune.py with dataset '$arg', prune_percent '$prune_percent', prune_type '$prune_type', prune_decay '$prune_decay', and v_pow '$vp' on port $port"
          
          CUDA_VISIBLE_DEVICES=$gpu_id nohup python prune_finetune.py \
            -s ""$base"/$arg" \
            -m "./${model_path}" \
            --eval \
            --port $port \
            --start_checkpoint "$ps1_path" \
            --iteration 35000 \
            --prune_percent $prune_percent \
            --prune_type $prune_type \
            --prune_decay $prune_decay \
            --position_lr_max_steps 35000 \
            -i $img_set \
            --v_pow $vp > "${model_path}/[${prune_percent}_prunned.log]" 2>&1 

          # Increment the port number for the next run
          ((port++))
          # Allow some time for the process to initialize and potentially use GPU memory
          sleep 60
          break
        else
          echo "No GPU available at the moment. Retrying in 1 minute."
          sleep 60
        fi
      done
    done
  done
done
wait
echo "All prune_finetune.py runs completed."
