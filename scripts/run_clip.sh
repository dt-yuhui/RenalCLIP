#!/bin/bash

exp_name="RenalCLIP"

python -m torch.distributed.run --nproc_per_node=4 main.py \
    --epochs 100 --batch_size_per_gpu 512 --freeze_bert True \
    --modalities A --max_words 224 \
    --use_grad_checkpointing False --use_fp16 True --bert_type emilyalsentzer/Bio_ClinicalBERT \
    --base_lr_img 1e-5 --base_lr_text 2e-5 \
    --base_lr_scratch 5e-4 \
    --beta2 0.98 \
    --exp_name ${exp_name} \
    --txt_transform True \
    --pre_processed_type one_kidney_kc_v3 \
    --split_file_name /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json \
    --text_encoder_freeze_layers 12 \
    --num_workers 12 --multiple_3d True \
    --llm2vec True --llm2vec_type radiology \
    --text_proj False --clip_output_dim 4096 \
    --img_encoder_pretrained_weight /cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/downstream_output/pretrain/cnn_one-kidney_txt-label-multi-task_A_3D_kc_100%_v3