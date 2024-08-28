#!/bin/bash
#SBATCH -A geo160
#SBATCH --job-name=hackathon-675m-batch-size-test   # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:8             # number of allocated gpus per node
#SBATCH -q debug
#SBATCH --time=00:30:00       # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=56
#SBATCH -C nvme

##### Setup modules
module load cpe/23.05         # recommended cpe version with cray-mpich/8.1.26
module load cray-mpich/8.1.26 # for better GPU-aware MPI w/ ROCm 5.7.1
module load PrgEnv-gnu/8.4.0
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH # because using a non-default cray-mpich
export LD_LIBRARY_PATH=/lustre/orion/geo160/proj-shared/testing/aws-olf-rccl-plugin/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
module load amd-mixed/5.7.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH="/mnt/bb/${USER}/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

##### sbcast env to local nvme
echo "copying torch_env to each node in the job"
conda_env_name='rocm-torch-test-full-0.1.0'

sbcast -pf /lustre/orion/geo160/proj-shared/envs/${conda_env_name}.tar.gz.hackathon /mnt/bb/${USER}/${conda_env_name}.tar.gz
echo /lustre/orion/geo160/proj-shared/envs/${conda_env_name}.tar.gz.hackathon
echo /mnt/bb/${USER}/${conda_env_name}.tar.gz
ls -l /mnt/bb/${USER}
ls -l /lustre/orion/geo160/proj-shared/envs 

if [ ! "$?" == "0" ]; then
	# CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
	# your application may pick up partially complete shared library files, which would give you confusing errors.
	echo "SBCAST failed!"
	exit 1
fi

srun --ntasks-per-node 1 mkdir /mnt/bb/${USER}/${conda_env_name}
echo "untaring torchenv"
srun --ntasks-per-node 1 tar -xzf /mnt/bb/${USER}/${conda_env_name}.tar.gz -C /mnt/bb/${USER}/${conda_env_name}
echo "Done untarring torchenv"

source activate /mnt/bb/${USER}/${conda_env_name}
echo "Activated ${conda_env_name}"

srun --ntasks-per-node 1 conda-unpack

# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_PORT=6000
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

# export NCCL_SOCKET_IFNAME=ib

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
datapaths=/lustre/orion/geo160/proj-shared/data/satvision-toa/50m
validationpath=/lustre/orion/geo160/proj-shared/data/satvision-toa/validation/sv_toa_128_chip_validation_04_24.npy
tensorboard_dir=/lustre/orion/geo160/proj-shared/data/tensorboard/hackathon_2024
batchsize=256
nprocpernode=8

launcher="python -u -m torch.distributed.run --nnodes=${nnodes} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=${nprocpernode}" 
echo $launcher 

cmd=" pytorch-caney/pytorch_caney/pipelines/pretraining/mim_deepspeed.py --cfg $1 --dataset MODIS --data-paths ${datapaths} --output . --batch-size ${batchsize} --validation-path ${validationpath} --tensorboard-dir ${tensorboard_dir}"
echo $cmd

srun -l -c56 --gpus-per-task=${nprocpernode} --gpu-bind=closest --jobid $SLURM_JOBID bash -c "$launcher --node_rank \$SLURM_PROCID $cmd" 

echo "END TIME: $(date)"

