#!/bin/bash

#SBATCH -J finetune_satvision_lc5
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=40       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=300G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=12:00:00       # total run time limit (HH:MM:SS)
#SBATCH --mail-type=ALL        # send email when job begins
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --qos=8n_a100
#SBATCH --reservation=jcv

#export PATH="/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin:$PATH"
#eval "$(conda shell.bash hook)"
#conda activate ilab-pytorch

#export PYTHONPATH=$PWD:../../../:../../../pytorch-caney
export NGPUS=4

#singularity exec --nv -B $NOBACKUP,/explore/nobackup/projects --env "PYTHONPATH=$PWD:../../../:/explore/nobackup/people/jacaraba/development/pytorch-caney" /explore/nobackup/projects/ilab/containers/pytorch-caney-2024.07.prod torchrun --nproc_per_node $NGPUS \
/discover/nobackup/jacaraba/spack/opt/spack/linux-sles15-zen/gcc-7.5.0/singularityce-3.11.3-o5pnooghlq7cgiv5zh5qnmyhmbltcynu/bin/singularity exec --nv -B /discover,/gpfsm --env "PYTHONPATH=$PWD:/discover/nobackup/jacaraba/development/downstream/pytorch-caney" /discover/nobackup/projects/akmosaic/container/pytorch-caney_latest torchrun --nproc_per_node $NGPUS \
    ../../../pytorch-caney/pytorch_caney/pipelines/finetuning/finetune.py \
	--cfg finetune_satvision_base_landcover5class_192_window12_3b-26m_50ep.yaml \
        --pretrained /discover/nobackup/projects/calebtest/3dclouds.runs/reconstruction/frontier_3b_100m_128_ckpt_50_pull/mp_rank_00_model_states.pt \
        --dataset MODISLC5 \
	--data-paths /discover/nobackup/projects/akmosaic/3dclouds.data/fixed-data-reproject/dataset-v2 \
	--batch-size 4 \
	--output /discover/nobackup/projects/akmosaic/3dclouds.data/fixed-data-reproject/dataset-v2/models \
	--enable-amp

        #--pretrained /discover/nobackup/projects/calebtest/3dclouds.runs/3b_26m_minmax/mim_satvision_pretrain-giant/mim_pretrain_swinv2_g_satvision_128_26m_window08_mpatch8_scaled_bt_minmax_100ep/ckpt_epoch_58_step_2000/mp_rank_00_model_states.pt \
# --pretrained /discover/nobackup/projects/calebtest/3dclouds.runs/reconstruction/frontier_3b_100m_128_ckpt_50_pull/mp_rank_00_model_states.pt \
