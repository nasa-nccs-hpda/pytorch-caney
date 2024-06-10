#!/bin/bash
#SBATCH --job-name=sv-toa-deepspeed   # create a short name for your job
#SBATCH --nodes=7                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=40       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=400G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=12:00:00       # total run time limit (HH:MM:SS)
#SBATCH --mail-type=ALL        # send email when job begins
#SBATCH --partition=gpu_a100
#SBATCH --reservation=warpsles15
#SBATCH --constraint=rome
#SBATCH --qos=8n_a100
#SBATCH --mail-user=caleb.s.spradlin@nasa.gov


# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_PORT=6000
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=ib

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo "MASTER_ADDR="$MASTER_ADDR


echo "$MASTER_ADDR:$MASTER_PORT"

export PYTHONPATH=$PWD:pytorch-caney
export NCCL_DEBUG=INFO

# do not remove or the training will hang and nodes will be lost w/o this workaround
#export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
#export TORCHELASTIC_ERROR_FILE=torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
#export NCCL_ASYNC_ERROR_HANDLING=1

#export NCCL_P2P_DISABLE=1

# cublas bug solve?
# export DISABLE_ADDMM_CUDA_LT=1

echo $SLURM_JOB_NUM_NODES
echo $SLURM_PROCID
echo $MASTER_ADDR
echo $MASTER_PORT

nnodes=$SLURM_JOB_NUM_NODES
validation_path="/discover/nobackup/projects/calebtest/3dclouds.runs/development/validation_test/data/sv_toa_128_chip_validation_04_24.npy"

launcher="/discover/nobackup/jacaraba/spack/opt/spack/linux-sles15-zen/gcc-7.5.0/singularityce-3.11.3-o5pnooghlq7cgiv5zh5qnmyhmbltcynu/bin/singularity exec --nv -B /discover,/gpfsm /discover/nobackup/projects/akmosaic/container/nvpt-24.01 python -u -m torch.distributed.run --nnodes=${nnodes} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=4" 
echo $launcher 

cmd=" pytorch-caney/pytorch_caney/pipelines/pretraining/mim_deepspeed.py --cfg $1 --dataset MODIS --data-paths /discover/nobackup/projects/calebtest/3dclouds/v3 --output . --batch-size 512 --validation-path ${validation_path} --resume $2"
echo $cmd

srun --jobid $SLURM_JOBID bash -c "$launcher --node_rank \$SLURM_PROCID $cmd" 

pkill -9 python

echo "END TIME: $(date)"
