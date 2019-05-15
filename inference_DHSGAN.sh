#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/result_icl_nuim_final/normal/result_reb_clean130_13k/ \
    --summary_dir /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/result_icl_nuim_final/normal/result_reb_clean130_13k/log/ \
    --mode inference \
    --is_training False \
    --task SRGAN \
    --input_dir_LR /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/test_icl_nuim_final/image_hazed \
    --tmap_beta 1.0  \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/experiment_rebutal_clean_130_SRGAN/model-130000