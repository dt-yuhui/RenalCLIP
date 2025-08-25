import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes)
            )
            
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        logits = self.block_forward(x)

        return logits


class FineTuner(nn.Module):
    def __init__(self,
                 backbone,
                 in_features,
                 task_info,
                 dropout,
                 hidden_dim=None,
                 finetune_type='linear_prob',
                 ):
        """_summary_

        Args:
            backbone (_type_): _description_
            in_features (_type_): _description_
            task_info (_type_): {task_name1: num_classes1, task_name2: num_classes2, ...}
            dropout (_type_): _
            hidden_dim (_type_, optional): _description_. Defaults to None.
            finetune_type (str, optional): _description_. Defaults to 'linear_prob'.
        """
        super().__init__()

        self.backbone = backbone
        self.finetune_type = finetune_type
        self.task_info = task_info

        # if linear_prob, freeze backbone
        # else update all params
        if self.finetune_type == 'linear_prob':
            print("==================== freezing backbone ====================")
            for param in self.backbone.parameters():
                param.requires_grad = False
    
        # 为每一个任务准备一个独立的线性分类器
        self.classifiers = nn.ModuleDict()
        for task_name, num_classes in task_info.items():
            self.classifiers[task_name] = SSLEvaluator(
                n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim)

    def forward_features(self, batch):
        # backbone forward
        if self.finetune_type == 'linear_prob':
            with torch.no_grad():
                feats_dict = self.backbone(batch)

        elif self.finetune_type == 'finetune' or self.finetune_type == 'train_from_scratch' or self.finetune_type == "pretrain":
            feats_dict = self.backbone(batch)
        else:
            raise NotImplementedError(fr'finetune type {self.finetune_type} not implemented!')

        return feats_dict

    def forward_head(self, feats_dict):

        kidney_logits = {}
        for task_name, classifier in self.classifiers.items():
            kidney_logits[task_name] = classifier(feats_dict)

        return kidney_logits

    def forward(self, batch):
        feats_dict = self.forward_features(batch)
        logits = self.forward_head(feats_dict)

        return logits


if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    import models.encoder as Encoder
    image_encoder_q = Encoder.encoder_small(norm_cfg2D='BN2', norm_cfg3D='BN3',
                                            activation_cfg='ReLU',
                                            weight_std=False, img_size2D=224, img_size3D=[32, 128, 128],
                                            modal_type='MM', hidden_dim=2048, output_dim=256)
    finetuner = FineTuner(image_encoder_q, 512, 3, 0)
    import pdb; pdb.set_trace()
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([[0],[1]])
    batch = (x,y)
    loss, _, _ = finetuner(batch, '2D')