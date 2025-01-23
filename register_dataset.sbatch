#!/bin/bash

#SBATCH --job-name=open_clip_exp
#SBATCH --output=./output.log
#SBATCH --error=./output.err
#SBATCH --partition=normal
#SBATCH --environment=open_clip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=01:00:00

export DATASET_NAME=CC12M
export DATASET_PATH=/iopsstor/scratch/cscs/tkerimog/datasets/cc12m-wds

export MIXTERA_SERVER_ADDR=$(hostname)
export MIXTERA_SERVER_DIR=/iopsstor/scratch/cscs/tkerimog/open_clip/mixtera_server
export MIXTERA_SERVER_PORT=12345

export PYTHON_EXEC=/iopsstor/scratch/cscs/tkerimog/open_clip/open_clip_env/bin/python
export PYTHON_SCRIPT=/iopsstor/scratch/cscs/tkerimog/open_clip/openclip-mixtera/register_dataset.py

$PYTHON_EXEC -u -m mixtera.network.server.entrypoint \
    $MIXTERA_SERVER_DIR \
    --host $MIXTERA_SERVER_ADDR\
    --port $MIXTERA_SERVER_PORT &

sleep 5

srun -ul --container-writable --environment=open_clip bash -c "\$PYTHON_EXEC \$PYTHON_SCRIPT" > /dev/null 2>&1

sleep 300