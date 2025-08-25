#!/bin/bash

# seed参数的集合
seeds=(42)

DEVICE_ID=0


DOWNSTREAM_TASK='RENAL_AIN'
NUM_CLASSES=3
EXTERNAL_HOSPITALS='厦门,连云,瑞金,山东,TCIA,external'

best_lr=5e-5
best_bs=100

# ours

for seed in "${seeds[@]}"; do
  # Dynamically set exp_name including seed, lr, and batch_size
  exp_name="cnn_one-kidney_3D_${DOWNSTREAM_TASK}_tc_res18_llm2vec-renal-mt-pretrain-v1_xiong-size_seed-${seed}"

  # Set the environment variable
  export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

  # Execute the Python script with the current hyperparameters
  python downstream_main_img.py \
    --finetune_type finetune --epochs 100 \
    --batch_size ${best_bs} --max_lr ${best_lr} --exp_name ${exp_name} \
    --model_type student --num_classes ${NUM_CLASSES} --downstream_task ${DOWNSTREAM_TASK} \
    --pre_processed True --pre_processed_type one_kidney_tc_v3 \
    --modalities A --split_file_name /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json \
    --cropsize_3d 128 --crop_slices 32 --in_features 512 \
    --seed ${seed} \
    --external_hospitals ${EXTERNAL_HOSPITALS} \
    --fusion False \
    --pretrained_exp_name RenalCLIP-image-encoder --pretrained_metric acc \
    --hidden_dim 128
done