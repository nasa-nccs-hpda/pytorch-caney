#!/bin/bash

#SBATCH -J pretrain_satvision_swinv2
#SBATCH -t 3-00:00:00
#SBATCH -G 4
#SBATCH -N 1


export PYTHONPATH=$PWD:$PWD/pytorch-caney
export NGPUS=4

torchrun --nproc_per_node $NGPUS \
    pytorch-caney/pytorch_caney/pipelines/pretraining/satmae_temporal.py \
	--cfg $1 \
	--data-paths /explore/nobackup/projects/ilab/data/satvision/pretraining/training_* \
	--batch-size 32 \
	--output output \
	--disable-amp
	# --enable-amp