#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --nodes=4
#SBATCH --time=48:00:00
#SBATCH --signal=SIGUSR1@120
#SBATCH --exclude=a100-st-p4d24xlarge-35

export NCCL_NET_SHARED_BUFFERS=0
PYTHONBIN=/fsx/lyuchen/base_fsx/bin/python
echo "Running on `hostname`"
srun $PYTHONBIN run_dino.py with \
	data_root=/data/home/lyuchen/scratch2/vilt_dataset \
	num_workers=12 \
	exp_name=dino/80k_512_clip_init_reg_0.01 num_gpus=4 num_nodes=4 \
  	seed=6666 max_epoch=100 \
	nmb_centroids=80000 \
    bottleneck_dim=512 \
    init_word_emb=clip \
	word_emb_reg_coef=0.01 \

#$PYTHONBIN run_dino.py with \
#	data_root=/data/home/lyuchen/scratch2/vilt_dataset \
#	num_workers=12 \
#	exp_name=dino_80k_512 num_gpus=1 num_nodes=1 \
#  	seed=6666 max_epoch=100 \
#	nmb_centroids=80000 \
#    bottleneck_dim=512 \
#    init_word_emb=clip \
#	fast_dev_run=5
