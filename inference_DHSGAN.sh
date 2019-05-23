#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./fog_exp4/ \
    --summary_dir ./log/ \
    --mode inference \
    --is_training False \
    --task SRGAN \
    --input_dir_LR ./test/image_hazed \
    --tmap_beta 2.0  \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint ./checkpoint/model-170000