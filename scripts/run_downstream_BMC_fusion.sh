#!/bin/bash

seeds=(42)
DEVICE_ID=0

DOWNSTREAM_TASK='BMC'
NUM_CLASSES=2
EXTERNAL_HOSPITALS='厦门,连云,张掖,瑞金,山东,TCIA,external'

best_lr=1e-5
best_bs=150

# llm2vec-renal, mt-pretrain, NAVD train
for seed in "${seeds[@]}"; do
  # 动态设置exp_name，包含seed, fold_index 和 lr
  exp_name="cnn_one-kidney_3D_${DOWNSTREAM_TASK}_tc_res18_llm2vec-renal-mt-pretrain-v1_xiong-size_4Dto3D_seed-${seed}"

  # 设置环境变量
  export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

  python downstream_main_img.py \
    --finetune_type finetune --epochs 100 \
    --batch_size ${best_bs} --max_lr ${best_lr} \
    --exp_name ${exp_name} \
    --model_type student --num_classes ${NUM_CLASSES} --downstream_task ${DOWNSTREAM_TASK} \
    --pre_processed True --pre_processed_type one_kidney_tc_v3 \
    --modalities A --split_file_name /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json \
    --cropsize_3d 128 --crop_slices 32 --in_features 512 \
    --seed ${seed} \
    --external_hospitals ${EXTERNAL_HOSPITALS} \
    --fusion False \
    --pretrained_exp_name RenalCLIP-image-encoder --pretrained_metric acc \
    --hidden_dim 128 \
    --multiple_3d True
done




# llm2vec-renal, mt-pretrain, NAVD train, voting ensemble
for seed in "${seeds[@]}"; do
  # 动态设置exp_name，包含seed, fold_index 和 lr
  exp_name="cnn_one-kidney_3D_${DOWNSTREAM_TASK}_tc_res18_llm2vec-renal-mt-pretrain-v1_xiong-size_seed-${seed}_NAVD-ensemble-test-all"

  # 设置环境变量
  export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

  # 动态构造 split_pretrained_path JSON 字符串
  split_pretrained_path=$(
    echo -n "{"
    echo -n "\"N\": \"cnn_one-kidney_3D_${DOWNSTREAM_TASK}_tc_res18_llm2vec-renal-mt-pretrain-v1_xiong-size_4Dto3D_seed-${seed}\", "
    echo -n "\"A\": \"cnn_one-kidney_3D_${DOWNSTREAM_TASK}_tc_res18_llm2vec-renal-mt-pretrain-v1_xiong-size_4Dto3D_seed-${seed}\", "
    echo -n "\"V\": \"cnn_one-kidney_3D_${DOWNSTREAM_TASK}_tc_res18_llm2vec-renal-mt-pretrain-v1_xiong-size_4Dto3D_seed-${seed}\", "
    echo -n "\"D\": \"cnn_one-kidney_3D_${DOWNSTREAM_TASK}_tc_res18_llm2vec-renal-mt-pretrain-v1_xiong-size_4Dto3D_seed-${seed}\""
    echo -n "}"
  )

  python downstream_main_img.py \
    --finetune_type finetune --epochs 100 \
    --batch_size ${best_bs} --max_lr ${best_lr} \
    --exp_name ${exp_name} \
    --model_type student --num_classes ${NUM_CLASSES} --downstream_task ${DOWNSTREAM_TASK} \
    --pre_processed True --pre_processed_type one_kidney_tc_v3 \
    --modalities N,A,V,D --split_file_name /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json \
    --cropsize_3d 128 --crop_slices 32 --in_features 512 \
    --seed ${seed} \
    --external_hospitals ${EXTERNAL_HOSPITALS} \
    --hidden_dim 128 \
    --test_only True --split_pretrained_path "${split_pretrained_path}" --hidden_dim 128
done
