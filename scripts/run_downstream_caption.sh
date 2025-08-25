#!/bin/bash


seeds=(42)

DEVICE_ID=0
DOWNSTREAM_TASK='caption'

# ours
adapter_lr=5e-4
lora_lr=2e-6

for seed in "${seeds[@]}"; do

  exp_name="cnn_one-kidney_3D_${DOWNSTREAM_TASK}_tc_res18_llm2vec-renal-mt-pretrain-v1_xiong-size_seed-${seed}"

  export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

  python downstream_main_caption_twostage.py \
  --finetune_type finetune \
  --stage_A_epochs 20 --stage_B_epochs 2 --batch_size 30 \
  --exp_name ${exp_name} \
  --pre_processed True \
  --modalities A --split_file_name /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json \
  --cropsize_3d 128 --crop_slices 32 --image_feature_dim 512 \
  --seed ${seed} \
  --pretrained_exp_name RenalCLIP-image-encoder --pretrained_metric acc \
  --stage_A_adapter_lr 1e-3 \
  --stage_B_adapter_lr ${adapter_lr} --lora_lr ${lora_lr}
done