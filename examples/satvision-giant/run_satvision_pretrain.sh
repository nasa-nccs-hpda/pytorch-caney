#!/bin/bash

#SBATCH -J deepspeed-satvision-giant
#SBATCH -t 3-00:00:00
#SBATCH -G 4
#SBATCH -N 1

module load singularity

srun -n 1 singularity exec \
    --env PYTHONPATH="$PWD:$PWD/pytorch-caney" \
	--nv -B /lscratch,/explore,/panfs \
	$1 \
	deepspeed \
	pytorch-caney/pytorch_caney/pipelines/pretraining/mim_deepspeed.py \
	--cfg $2 \
	--dataset MODIS \
	--data-paths /explore/nobackup/projects/ilab/data/satvision/pretraining/training_* \
	--batch-size 32 \
	--output . \
	--enable-amp



