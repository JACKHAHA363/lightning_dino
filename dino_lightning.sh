#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --nodes=4
#SBATCH --time=30:00:00
#SBATCH --signal=SIGUSR1@120

export NCCL_NET_SHARED_BUFFERS=0
PYTHONBIN=/data/home/lyuchen/miniconda/envs/vilt/bin/python
echo "Running on `hostname`"
srun $PYTHONBIN run_dino.py with \
	data_root=/data/home/lyuchen/scratch2/vilt_dataset \
	num_workers=12 \
    exp_name=dino num_gpus=2 num_nodes=4 \
	seed=1234 max_epoch=50

#$PYTHONBIN run_dino.py with data_root=/data/home/lyuchen/scratch2/vilt_dataset \
#	num_workers=12 \
#	exp_name=dino_imagenet/debug \
#	num_gpus=1 num_nodes=1 fast_dev_run=5