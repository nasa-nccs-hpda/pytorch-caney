#!/bin/bash
#SBATCH -G 1
#SBATCH --time 2:00:00
#SBATCH -N 1
#SBATCH -J sv-inference-giant

module load singularity

srun singularity exec --nv --env PYTHONPATH=$PWD:$PWD/pytorch-caney -B /explore,/panfs /explore/nobackup/projects/ilab/containers/pytorch-caney-2024-08.dev python pytorch-caney/examples/inference/satvision-toa-reconstruction_giant.py --pretrained_model_dir /explore/nobackup/people/cssprad1/projects/satvision-toa/models/satvision-toa-giant-patch8-window8-128 --output_dir .