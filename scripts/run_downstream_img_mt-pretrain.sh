#!/bin/bash


DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

python downstream_main_img.py \
    --finetune_type pretrain --epochs 200 --batch_size 300 \
    --max_lr 5e-4 --exp_name cnn_one-kidney_txt-label-multi-task_A_3D_kc_100%_v3 \
    --model_type student --downstream_task multi_task \
    --pre_processed True --pre_processed_type one_kidney_kc_v3 \
    --modalities A --split_file_name /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json \
    --cropsize_3d 128 --crop_slices 32 --in_features 512 \
    --multiple_3d True