#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./experiment_DHSGAN/ \
    --summary_dir ./experiment_DHSGAN/log/ \
    --mode train \
    --is_training True \
    --task SRGAN \
    --batch_size 4 \
    --flip True \
    --random_crop True \
    --crop_size 96 \
    --tmap_beta 2.0 \
    --input_dir_LR ./train/train_global_reside/image_hazed/ \
    --input_dir_HR ./train/train_global_reside/image_real/ \
    --vgg_ckpt ./train/vgg_19.ckpt\
    --num_resblock 8 \
    --perceptual_mode VGG54 \
    --ratio 0.001 \
    --learning_rate 0.00001 \
    --decay_step 100000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
    --max_iter 200000 \
    --queue_thread 12 \
    --vgg_scaling 0.0061 \
    --pre_trained_model_type SRGAN \
    --pre_trained_model True \
    --checkpoint ./experiment_generator/model-170000

