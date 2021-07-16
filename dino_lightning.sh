#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@120

export NCCL_NET_SHARED_BUFFERS=0
PYTHONBIN=/data/home/lyuchen/miniconda/envs/vilt/bin/python
echo "Running on `hostname`"
$PYTHONBIN run_dino.py with \
	num_workers=12 \
	exp_name=repro_8gpu num_gpus=8 num_nodes=1 \
	seed=0

$PYTHONBIN run_dino.py with \
	num_workers=12 \
	exp_name=dino_imagenet/debug num_gpus=1 num_nodes=1 fast_dev_run=5