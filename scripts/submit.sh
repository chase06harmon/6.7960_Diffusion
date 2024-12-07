#!/bin/bash

#SBATCH -c 16
#SBATCH --gres=gpu:volta:2
#SBATCH -o ../training_logs/EnglishSlangTest4.txt

# Loading Modules
source /etc/profile
module load anaconda/2023a 
module load cuda/11.6
module load nccl/2.11.4-cuda11.6

source activate

python print_hello.py

echo STARTING TRAINING

# Run the script 
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12231 --use_env run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 20000 \
--save_interval 5000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 425 \
--microbatch 425 \
--dataset EnglishSlang \
--data_dir ../datasets/EnglishSlang \
--learned_mean_embed True \
--denoise True \
--vocab bert \
--seq_len 128 \
--use_fp16 \
--denoise_rate 0.5 \
--schedule_sampler lossaware \
--notes learned_mask_fp16_denoise_0.5_reproduce