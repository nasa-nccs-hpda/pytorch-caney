#!/bin/bash

#SBATCH -J 1000_unet-train-modislc5-ls-weights 
#SBATCH -t 3-00:00:00
#SBATCH -G 4
#SBATCH -N 1

module load anaconda

conda activate rapids-23.02

export PYTHONPATH=$PWD:$PWD/pytorch-caney

# srun -n 1 python \
python \
    pytorch-caney/pytorch_caney/pipelines/torchgeo_trainers/modis_segmentation_torchgeo_training_ls.py \
    --data_path /panfs/ccds02/nobackup/projects/ilab/data/satvision/finetuning/h09v05/h09v05_5_class \
    --ngpus 8 \
    --n_classes 5 \
    --batch_size 64 \
    --use_weights \
    --n_samples 1000 \
    --patience 30 \
    --min_epochs 90

