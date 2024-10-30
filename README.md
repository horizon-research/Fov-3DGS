# FoV-3DGS
Official Implementation of **MetaSapiens: Real-Time Neural Rendering with Efficiency-Aware Pruning and Accelerated Foveated Rendering.** [[Project Page](https://horizon-lab.org/metasapiens/)]
(ASPLOS 2025)

## (1) Setup
- Clone the repo
```bash
git clone https://github.com/horizon-research/FoV-3DGS.git
```
- Prepare Dataset
    - Download from: [Mip360](https://jonbarron.info/mipnerf360/)
    - Download the official preprocessed data from 3D Gaussian Splatting (3DGS):  [Tank&Temple and DeepBlending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
    - put the unzipped scene folders in [FoV-3DGS/dataset](./dataset)

- Prepare Dense 3DGS for pruning
    - Original 3DGS: download from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
    - Mini-Splatting-D: We provide our reprodeuced model of m360 bicycle [here](https://drive.google.com/file/d/1H2XhS1Jh-Pd-W8NvA4Z0bLlUADcJhAj5/view), for other scenes, you can reproduce using [their code](https://github.com/fatPeter/mini-splatting/tree/main).
    - move the dense model to the scene folder and name it "ms_d", structure under the folder should be like:
    ```bash
    |-- cameras.json
    |-- cfg_args
    |-- chkpnt30000.pth
    |-- input.ply
    `-- point_cloud
        `-- iteration_30000
            `-- point_cloud.ply
    ```

- Prepare environment:
    - We use docker: 
    ```bash
    # pull the docker (this is for x86 machine, for jetson you will need other prbuilt, see https://github.com/dusty-nv/jetson-containers/tree/master and find one that suitable for tour jetpack.)
    docker pull pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
    # run docker
    bash ./run_docker.sh
    # go in to docker and install all submodules
    bash update_submodules.sh
    ```
    - install some packages
    ```bash 
        pip install plyfile opencv-python matplotlib icecream
        apt-get update
        apt-get install libgl1-mesa-glx libglib2.0-0 -y
    ```

## (2) Run the pruning & FR (Foveated Rendering) Masking pipeline
```bash
# we only leave bicycle, uncomment other scenes for batch test
 python3 combined_training_script.py 
```
The result will be stored in the scene folder.


## (3) Measure Objective Metric for PS=1 model.
```bash
python3 quality_eval.py 
```
The result will be in ./full_eval_results/ours-Q

## (4) Generate the FoV model & Measure its FPS
```bash
bash batch_ours_fps.sh 
```


## Baselines
- SM (Shared Model) FR: Generate + Measure the Layer-Wise Quality & FPS
```bash
# we only leave bicycle, uncomment other scenes for batch test
bash batch_gen_naive_FR.sh # generate SMFR
python3 quality_eval_layers_naive.py # measure qulaity in each layer, result will be in ./layers_eval_results/naiveFR
bash batch_naive_fps.sh #measure fps, result will be in ./fps
```


- MM (Multi-Model) FR : Generate + Measure the Layer-Wise Quality & FPS
this one need [LightGS](https://github.com/VITA-Group/LightGaussian) for Pruning Multiple Models, we already include it in our repo
```bash
bash batch_pnum_analyzer.sh  # analyze pnum of our model in each layer
cd ../LightGaussian
bash ./batch_gen_mmFR.sh # the result will be in ./MMFR/ours-Q
cd ../fov3dgs
python3 quality_eval_layers_mmfr.py # measure qulaity in each layer, result will be in ./layers_eval_results/MMFR
bash ./batch_mmfr_fps.sh #measure fps, result will be in ./fps
```



## Acknowledgements
- Our 3D Gaussian Splatting (3DGS) related code is based on the work from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).
- Our Human Visual System (HVS) model code is adapted from the [Perception](https://github.com/kaanaksit/odak/tree/master/odak/learn/perception) library in [Odak](https://github.com/kaanaksit/odak/tree/master).