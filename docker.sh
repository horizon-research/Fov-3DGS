docker run --gpus all \
           --name rtgs \
           -d -it \
           --network=host \
           --volume $PATH_TO_REPO/FoV-3DGS/:/workspace \
           pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

