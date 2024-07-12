#!/bin/bash

#SBATCH -J finetune_satvision_lc9
#SBATCH -t 3-00:00:00
#SBATCH -G 4
#SBATCH -N 1

export PYTHONPATH=$PWD:$PWD/pytorch-caney
export NGPUS=4

torchrun --nproc_per_node $NGPUS \
	pytorch-caney/pytorch_caney/pipelines/finetuning/finetune.py \
	--cfg $1 \
	--pretrained $2 \
	--dataset MODISLC9 \
	--data-paths /explore/nobackup/projects/ilab/data/satvision/finetuning/h18v04/labels_5classes_224 \
	--batch-size 4 \
	--output . \
	--enable-amp
