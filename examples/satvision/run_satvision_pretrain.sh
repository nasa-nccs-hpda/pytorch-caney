#!/bin/bash

#SBATCH -J pretrain_satvision_swinv2
#SBATCH -t 3-00:00:00
#SBATCH -G 4
#SBATCH -N 1


export PYTHONPATH=$PWD:../../../:../../../pytorch-caney
export NGPUS=4

torchrun --nproc_per_node $NGPUS \
    ../../../pytorch-caney/pytorch_caney/pipelines/pretraining/simmim.py \
	--cfg mim_pretrain_swinv2_satvision_base_192_window12_800ep.yaml \
    --dataset MODIS \
	--data-paths /explore/nobackup/projects/ilab/data/satvision/pretraining/training_* \
	--batch-size 128 \
	--output /explore/nobackup/people/cssprad1/projects/satnet/code/development/cleanup/trf/transformer/models \
	--enable-amp