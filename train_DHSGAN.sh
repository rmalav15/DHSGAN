#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/experiment_net_slimming_by2_res12_SRGAN/ \
    --summary_dir /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/experiment_net_slimming_by2_res12_SRGAN/log/ \
    --mode train \
    --is_training True \
    --task SRGAN \
    --batch_size 4 \
    --flip True \
    --random_crop True \
    --crop_size 96 \
    --input_dir_LR /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/train/train_global_reside/image_hazed/ \
    --input_dir_HR /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/train/train_global_reside/image_real/ \
    --vgg_ckpt /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/vgg_19.ckpt\
    --num_resblock 8 \
    --perceptual_mode VGG54 \
    --ratio 0.001 \
    --learning_rate 0.0001 \
    --decay_step 100000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
    --max_iter 200000 \
    --queue_thread 12 \
    --vgg_scaling 0.0061 \
    --pre_trained_model_type SRResnet \
    --pre_trained_model True \
    --checkpoint /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/experiment_net_slimming_by2_res12_SRResNet/model-200000

