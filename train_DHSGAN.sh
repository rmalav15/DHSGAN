#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /mnt/069A453E9A452B8D/Ram/DHSGAN_data/fog_exp3/ \
    --summary_dir /mnt/069A453E9A452B8D/Ram/DHSGAN_data/fog_exp3/log/ \
    --mode train \
    --is_training True \
    --task SRGAN \
    --batch_size 4 \
    --flip True \
    --random_crop True \
    --crop_size 96 \
    --tmap_beta 2.0 \
    --input_dir_LR /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/train/train_global_reside/image_hazed/ \
    --input_dir_HR /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/train/train_global_reside/image_real/ \
    --vgg_ckpt /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/vgg_19.ckpt\
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
    --checkpoint /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/experiment_clean_reside_pred_g20_SRGAN/model-170000

