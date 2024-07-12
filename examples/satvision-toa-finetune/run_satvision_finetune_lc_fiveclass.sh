#!/bin/bash

#SBATCH -J finetune_satvision_lc5
#SBATCH -t 3-00:00:00
#SBATCH -G 4
#SBATCH -N 1


export PYTHONPATH=$PWD:../../../:../../../pytorch-caney
export NGPUS=8

torchrun --nproc_per_node $NGPUS \
    ../../../pytorch-caney/pytorch_caney/pipelines/finetuning/finetune.py \
	--cfg finetune_satvision_base_landcover5class_192_window12_100ep.yaml \
	--pretrained /explore/nobackup/people/cssprad1/projects/satnet/code/development/masked_image_modeling/development/models/simmim_satnet_pretrain_pretrain/simmim_pretrain__satnet_swinv2_base__img192_window12__800ep_v3_no_norm/ckpt_epoch_800.pth \
    --dataset MODISLC9 \
	--data-paths /explore/nobackup/projects/ilab/data/satvision/finetuning/h18v04/labels_9classes_224 \
	--batch-size 4 \
	--output /explore/nobackup/people/cssprad1/projects/satnet/code/development/cleanup/finetune/models \
	--enable-amp