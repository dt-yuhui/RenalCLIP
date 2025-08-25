import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18
import os
import numpy as np
from utils.util import load_state_with_same_shape
from utils.logger import print_log
from collections import OrderedDict
import torchvision
import copy


BASE_DIR = fr'/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP'
CUSTOM_PRETRAINED_DIR = fr"/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/clip_output"


class RenalModel(nn.Module):
    def __init__(self,
                 use_grad_checkpointing=False,
                 mode='finetune',
                 pretrained_exp_name=None,
                 pretrained_metric='acc',
                 model_type='student',
                 logger=None,
                 clip_hidden_dim=None,
                 clip_output_dim=128,
                 ):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.mode = mode
        self.pretrained_exp_name = pretrained_exp_name
        self.pretrained_metric = pretrained_metric
        self.model_type = model_type
        self.logger = logger
        self.clip_hidden_dim = clip_hidden_dim
        self.clip_output_dim = clip_output_dim

        self.init_models()

    def init_models(self):
        self.image_encoder = resnet18(num_classes=2, 
                                      shortcut_type='A', 
                                      mode=self.mode, 
                                      use_grad_checkpointing=self.use_grad_checkpointing,
                                      clip_hidden_dim=self.clip_hidden_dim, 
                                      clip_output_dim=self.clip_output_dim)

        if self.pretrained_exp_name is not None:
            self.load_pretrained_weights(image_backbone=self.image_encoder)
        else:
            print_log("==================== image_encoder train from scratch ====================", logger=self.logger)

    def load_pretrained_weights(self, image_backbone):
        print_log("==================== load pre-trained image_encoder ====================", logger=self.logger)
        file_list = os.listdir(os.path.join(CUSTOM_PRETRAINED_DIR, self.pretrained_exp_name))
        if self.pretrained_metric in ["acc", "loss"]:
            file_extension = self.pretrained_metric + '.pt'

        for file_name in file_list:
            if file_name.endswith(file_extension):
                checkpoint_path = os.path.join(CUSTOM_PRETRAINED_DIR, self.pretrained_exp_name, file_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state_dict = checkpoint['model']
        filtered_image_encoder_weight = load_state_with_same_shape(image_backbone,
                                                                    model_state_dict,
                                                                    model_type=self.model_type,
                                                                    load_module_name='image_encoder')
        msg = image_backbone.load_state_dict(filtered_image_encoder_weight, strict=False)
        print_log(f"image_backbone from {self.pretrained_exp_name}: {msg}", logger=self.logger)

    def forward3D(self, imgs):
        kidneys_feat = self.image_encoder.forward_features(imgs)
        kidneys_feat = self.image_encoder.avgpool(kidneys_feat).flatten(1)

        return kidneys_feat

    def forward(self, imgs):

        img_feat_3D = self.forward3D(imgs)

        return img_feat_3D

if __name__ == '__main__':
    print('hello world')