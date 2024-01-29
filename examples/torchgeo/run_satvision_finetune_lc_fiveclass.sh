#!/bin/bash

#SBATCH -J finetune_satvision_lc5
#SBATCH -t 3-00:00:00
#SBATCH -G 4
#SBATCH -N 1

module load anaconda

conda activate rapids-23.02

export PYTHONPATH=$PWD:$PWD/pytorch-caney
export NGPUS=4

# torchrun --nproc_per_node 4 \
python \
	pytorch-caney/pytorch_caney/pipelines/finetuning/finetune.py \
	--cfg finetune_satvision_base_landcover5class_128_window16_100ep.yaml \
	--pretrained $1 \
    --dataset MODISLC5 \
	--data-paths /panfs/ccds02/nobackup/projects/ilab/data/satvision/finetuning/h09v05/h09v05_5_class \
	--batch-size 32 \
	--output . \
	--enable-amp
