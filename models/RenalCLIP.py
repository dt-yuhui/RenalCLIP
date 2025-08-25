import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from utils.data_util import *
from models.resnet import resnet18
from collections import OrderedDict
import numpy as np
from utils.logger import print_log
from transformers import AutoModel, AutoTokenizer
from models.resnet import GlobalEmbedding, IdentityModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = fr'/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP'
TEXT_PRETRAINED_DIR = fr"/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/pretrained_models/language_family"


class RenalCLIP(nn.Module):
    def __init__(self,
                 use_grad_checkpointing=False,
                 bert_type='emilyalsentzer/Bio_ClinicalBERT',
                 clip_hidden_dim=None,
                 clip_output_dim=128,
                 mode='pretrain',
                 modalities=('N', 'A', 'V', 'D'),
                 pretrained_img_encoder_weight=None,
                 logit_scale_init_value=0.07,
                 max_words=224,
                 text_proj = False,
                 llm2vec = False,
                 ):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.bert_type = bert_type
        self.mode = mode
        self.modalities = modalities
        self.pretrained_img_encoder_weight = pretrained_img_encoder_weight
        self.text_proj = text_proj
        self.clip_output_dim = clip_output_dim
        self.clip_hidden_dim = clip_hidden_dim

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

        # tokenizer params
        self.max_words = max_words

        self.llm2vec = llm2vec
        
        self.init_tokenizer()
        self.init_models()

    def init_tokenizer(self):
        TEXT_MODEL_NAME = os.path.join(TEXT_PRETRAINED_DIR, self.bert_type)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, local_files_only=True, trust_remote_code=True)

    def init_models(self):
        self.image_encoder_q_student = resnet18(num_classes=2,
                                                shortcut_type='A',
                                                clip_output_dim=self.clip_output_dim,
                                                clip_hidden_dim=self.clip_hidden_dim,
                                                use_grad_checkpointing=self.use_grad_checkpointing,
                                                )

        if not self.llm2vec:
            self.text_encoder_q_student = AutoModel.from_pretrained(os.path.join(TEXT_PRETRAINED_DIR, self.bert_type), local_files_only=True, trust_remote_code=True)
        else:
            self.text_encoder_q_student = IdentityModel()

        if not self.text_proj:
            self.text_encoder_q_student.global_embedding = nn.Identity()
        else:
            if self.llm2vec:
                input_dim = 4096
            else:
                input_dim = 768
            self.text_encoder_q_student.global_embedding = GlobalEmbedding(input_dim, self.clip_hidden_dim, self.clip_output_dim)
        
        if self.pretrained_img_encoder_weight is not None:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        checkpoint_path = os.path.join(self.pretrained_img_encoder_weight, 'best-checkpoint.pth')
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        image_encoder_new_state_dict = OrderedDict()

        model_state_dict = self.image_encoder_q_student.state_dict()

        for k, v in state_dict.items():

            if k.startswith('module.'):
                k = k.removeprefix('module.')

            if k.startswith('backbone.image_encoder.'):
                k = k.removeprefix('backbone.image_encoder.')
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    image_encoder_new_state_dict[k] = v

        msg = self.image_encoder_q_student.load_state_dict(image_encoder_new_state_dict, strict=False)
        print(f"image encoder from {self.pretrained_img_encoder_weight}: {msg}")            

    def _process_single_report(self, report, mode):
        report = preprocess_text(report)
        if mode == 'train':
            report = sentence_shuffling(report)
        report = restore_special_cases(report)

        return report
    
    def _get_caption(self, raw_reports, mode='train'):
        batch_size = len(raw_reports['left_kidney_findings'])
        left_reports, right_reports = [], []

        for i in range(batch_size):
            left_kidney_findings = self._process_single_report(raw_reports['left_kidney_findings'][i], mode)
            left_kidney_impression = self._process_single_report(raw_reports['left_kidney_impression'][i], mode)
            right_kidney_findings = self._process_single_report(raw_reports['right_kidney_findings'][i], mode)
            right_kidney_impression = self._process_single_report(raw_reports['right_kidney_impression'][i], mode)

            left_report_list = ['Findings: ', left_kidney_findings, '\n', 'Impression:', left_kidney_impression]
            left_report_full = ''.join(left_report_list)

            right_report_list = ['Findings: ', right_kidney_findings, '\n', 'Impression:', right_kidney_impression]
            right_report_full = ''.join(right_report_list)
            left_reports.append(left_report_full)
            right_reports.append(right_report_full)

        return {
            'left_reports': left_reports,
            'right_reports': right_reports
        }

    def _tokenize_report(self, batch_raw_reports, mode='train'):
        results = self._get_caption(batch_raw_reports, mode)
        left_reports = results['left_reports']
        right_reports = results['right_reports']

        tokenized_left_reports = self.tokenizer(
            left_reports,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_words
        )

        tokenized_right_reports = self.tokenizer(
            right_reports,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_words
        )

        tokenized_batch = {
            'left_reports': left_reports,
            'left_report_tokens_ids': tokenized_left_reports['input_ids'],
            'left_attention_mask': tokenized_left_reports['attention_mask'],

            'right_reports': right_reports,
            'right_report_tokens_ids': tokenized_right_reports['input_ids'],
            'right_attention_mask': tokenized_right_reports['attention_mask'],
        }

        return tokenized_batch

    def forward_text_one_kidney(self, batch, side, stage='train'):
        if self.llm2vec:
            report_emb_q_student = None
            report_feat_q_student = batch[f"{side}_llm2vec_features"]
        else:
            report_tokens_ids = batch[f"{side}_report_tokens_ids"]
            attention_mask = batch[f"{side}_attention_mask"]
            report_emb_q_student = None

            student_input = report_tokens_ids

            report_feat_q_student = self.text_encoder_q_student(input_ids=student_input, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        report_emb_q_student = self.text_encoder_q_student.global_embedding(report_feat_q_student)
        report_emb_q_student = F.normalize(report_emb_q_student, dim=-1)

        res_dict = {
            'report_emb_q_student': report_emb_q_student
        }

        return res_dict

    def forward_text(self, batch, stage='train'):
        l_txt_res_dict = self.forward_text_one_kidney(batch=batch, side='left', stage=stage)
        r_txt_res_dict = self.forward_text_one_kidney(batch=batch, side='right', stage=stage)

        return {
            'l_txt_res_dict': l_txt_res_dict,
            'r_txt_res_dict': r_txt_res_dict,
        }

    def forward_image_3D_one_kidney(self, batch, side, stage='train'):
        kidney, exist_modalities = batch[f"{side}_imgs"], batch['exist_modalities']
        img_emb_q_student_3D, img_feat_q_student_3D = None, None

        student_input = kidney

        img_feat_q_student_3D = self.image_encoder_q_student.forward_features(student_input)
        cls_feat_q_student_3D = self.image_encoder_q_student.avgpool(img_feat_q_student_3D).flatten(1)

        cls_emb_q_student_3D = self.image_encoder_q_student.global_embedding(cls_feat_q_student_3D)
        img_emb_q_student_3D = F.normalize(cls_emb_q_student_3D, dim=-1)

        res_dict = {
            'img_feat_q_student_3D': img_feat_q_student_3D,
            'img_emb_q_student_3D': img_emb_q_student_3D
        }

        return res_dict

    def forward_image_3D(self, batch, stage='train'):
        l_img_3D_res_dict = self.forward_image_3D_one_kidney(batch=batch, side='left', stage=stage)
        r_img_3D_res_dict = self.forward_image_3D_one_kidney(batch=batch, side='right', stage=stage)

        return {
            'l_img_3D_res_dict': l_img_3D_res_dict,
            'r_img_3D_res_dict': r_img_3D_res_dict,
        }

    def forward(self, batch, stage='train'):
        # volume-level forward
        # img_emb_q, report_emb_q, # selected_volumes x 128, B x 128

        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)

        # text forward
        txt_res_dict = self.forward_text(batch, stage=stage)

        # Initialize result dictionary
        result_dict = {}

        # Process text results
        result_dict.update({f'l_{k}': v for k, v in txt_res_dict['l_txt_res_dict'].items()})
        result_dict.update({f'r_{k}': v for k, v in txt_res_dict['r_txt_res_dict'].items()})

        # img_3D forward
        img_3D_res_dict = self.forward_image_3D(batch, stage=stage)

        # Process 3D image results
        result_dict.update({f'l_{k}': v for k, v in img_3D_res_dict['l_img_3D_res_dict'].items()})
        result_dict.update({f'r_{k}': v for k, v in img_3D_res_dict['r_img_3D_res_dict'].items()})

        return result_dict

if __name__ == '__main__':
    pass