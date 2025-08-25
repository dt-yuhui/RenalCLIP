#!/bin/bash

EXTERNAL_HOSPITALS='TCIA,external'
DEVICE_ID=0
ZY_IC='v2'


avg_before_align=False
use_max_similarity=False

exp_name="llm2vec_mt-pretrain_zeroshot_ensemble-${avg_before_align}_use_max_similarity-${use_max_similarity}"
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
python downstream_main_img_zeroshot.py \
  --finetune_type zeroshot --epochs 100 --batch_size 150 \
  --max_lr 2e-4 --exp_name ${exp_name} \
  --model_type student \
  --pretrained_exp_name RenalCLIP-image-encoder --pretrained_metric acc \
  --pre_processed True --pre_processed_type one_kidney_tc_v3 \
  --modalities A --split_file_name /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json \
  --cropsize_3d 128 --crop_slices 32 --in_features 512 \
  --zeroshot_attributes BMC,IC \
  --external_hospitals ${EXTERNAL_HOSPITALS} \
  --clip_output_dim 4096 \
  --avg_before_align ${avg_before_align} \
  --use_max_similarity ${use_max_similarity} \
  --ZY_IC ${ZY_IC}



avg_before_align=False
use_max_similarity=True

exp_name="llm2vec_mt-pretrain_zeroshot_ensemble-${avg_before_align}_use_max_similarity-${use_max_similarity}"
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
python downstream_main_img_zeroshot.py \
  --finetune_type zeroshot --epochs 100 --batch_size 150 \
  --max_lr 2e-4 --exp_name ${exp_name} \
  --model_type student \
  --pretrained_exp_name RenalCLIP-image-encoder --pretrained_metric acc \
  --pre_processed True --pre_processed_type one_kidney_tc_v3 \
  --modalities A --split_file_name /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json \
  --cropsize_3d 128 --crop_slices 32 --in_features 512 \
  --zeroshot_attributes BMC,IC \
  --external_hospitals ${EXTERNAL_HOSPITALS} \
  --clip_output_dim 4096 \
  --avg_before_align ${avg_before_align} \
  --use_max_similarity ${use_max_similarity} \
  --ZY_IC ${ZY_IC}