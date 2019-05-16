#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /mnt/069A453E9A452B8D/Ram/DHSGAN_data/fog_exp1/ \
    --summary_dir /mnt/069A453E9A452B8D/Ram/DHSGAN_data/fog_exp1/log/ \
    --mode inference \
    --is_training False \
    --task SRGAN \
    --input_dir_LR /mnt/069A453E9A452B8D/Ram/KAIST/Fog_Videos_Real/test_CAP/image_real \
    --tmap_beta 1.0  \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/experiment_clean_reside_pred_g20_SRGAN/model-170000