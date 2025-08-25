import os
import argparse
import utils.util as util
import numpy as np


def get_clip_args():
    parser = argparse.ArgumentParser()

    # Distributed training parameters
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--batch_size_per_gpu', type=int, default=32)

    # 3D dataset parameters (with augmentation)
    parser.add_argument('--datalist_3d', default='/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/dataset_RenalCLIP.json', type=str, help='json for 3D datasets')
    parser.add_argument('--split_file_name', default='/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json', type=str, help='json for 3D datasets')
    parser.add_argument('--a_min', default=-1000.0, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=3000, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--resize_3d', default=140, type=int, help='Resize x, y, z')
    parser.add_argument('--resize_slices', default=32, type=int, help='The slices num')
    parser.add_argument('--max_words', default=224, type=int, help='Max len of words')
    parser.add_argument('--cropsize_3d', default=128, type=int, help='Augmentation parameters for 3D images')
    parser.add_argument('--crop_slices', default=32, type=int, help='The slices num')
    parser.add_argument("--RandRotated_range_in_plane", default=np.pi / 180.0 * 10, type=float, help="range_x of random rotation")
    parser.add_argument("--RandRotated_range_out_of_plane", default=np.pi / 180.0 * 10, type=float, help="range_y of random rotation")
    parser.add_argument("--RandTranslated_range_in_plane", default=10, type=int, help="range_z of random translation")
    parser.add_argument("--RandTranslated_range_out_of_plane", default=0, type=int, help="range_z of random translation")
    parser.add_argument("--RandRotated_prob", default=0.1, type=float, help="RandRotated aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--RandGaussianSmoothd_prob", default=0.1, type=float, help="RandGaussianSmoothd aug probability")
    parser.add_argument('--pre_processed', default=True, type=util.bool_flag, help="whether to use pre_processed images")
    parser.add_argument('--pre_processed_type', default="regular", type=str, help="type of pre_processed images, choices ['MONAI', 'regular', 'both_kidneys', 'one_kidney']")
    parser.add_argument('--txt_transform', default=True, type=util.bool_flag, help='whether to shuffle sentence')

    # Model parameters 
    parser.add_argument('--use_grad_checkpointing', default=False, type=util.bool_flag, help='whether to use checkpointing')
    parser.add_argument('--freeze_bert', default=False, type=util.bool_flag, help='whether to freeze bert')
    parser.add_argument('--bert_type', default='zh-cn', type=str, help='choices of bert type')
    parser.add_argument('--img_encoder_pretrained_weight', default=None, type=str, help='type of image encoder pretrained weight')
    parser.add_argument('--modalities', default=('N', 'A', 'V', 'D'), type=util.parse_tuple, help="input modalities")
    parser.add_argument('--text_encoder_freeze_layers', default=9, type=int, help='how many layers are frozen in text encoder')
    parser.add_argument('--checkpoint_layers', default=0, type=int, help="beginning layers to enable gradient checkpoint")
    parser.add_argument('--clip_output_dim', default=128, type=int, help='The output dim in the projection head')
    parser.add_argument('--clip_hidden_dim', default=None, type=int, help='The hidden dim in the projection head')
    parser.add_argument('--text_proj', default=False, type=util.bool_flag, help='whether to use proj after text encoder')
    parser.add_argument('--multiple_3d', default=False, type=util.bool_flag, help='whether to use proj after text encoder')

    # Training/Optimization parameters
    parser.add_argument('--resume', default=None, type=str, help='Path for continue training')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')

    parser.add_argument('--use_fp16', type=util.bool_flag, default=False, help="Whether or not to use half precision for training.")
    parser.add_argument("--base_lr_img", default=2e-4, type=float, help="Learning rate for train from scratch at the end of linear warmup (highest LR used during training).")
    parser.add_argument("--base_lr_text", default=2e-5, type=float, help="Learning rate for train from scratch at the end of linear warmup (highest LR used during training).")
    parser.add_argument("--base_lr_scratch", default=1e-4, type=float, help="Learning rate for train from scratch at the end of linear warmup (highest LR used during training).")
    parser.add_argument("--momentum", default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Initial value of the weight decay.")
    parser.add_argument('--clip_grad', type=float, default=0.2, help="""Maximal parameter gradient norm if using gradient clipping.""")
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--beta1', type=float, default=0.9, help="""Initial value of the weight decay.""")
    parser.add_argument('--beta2', type=float, default=0.98, help="""Initial value of the weight decay.""")

    # llm2vec
    parser.add_argument('--llm2vec', type=util.bool_flag, default=False, help='whether to enable llm2vec')
    parser.add_argument('--llm2vec_type', type=str, default="wiki", help='type of llm2vec, choices in [wiki | radiology]')

    # Output parameters
    parser.add_argument('--exp_name', default="RenalCLIP_BASELINEv2", type=str, help='Path to save logs, tensorboard and checkpoints.')
    parser.add_argument('--log_name', default="RenalCLIP", type=str, help='log name')
    args = parser.parse_args()

    args.experiment_path = os.path.join('/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/clip_output', args.exp_name)
    args.tfboard_path = os.path.join('/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/clip_output', 'tfboard', args.exp_name)

    create_experiment_dir(args)

    return args


def get_downstream_args_img():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--split_random_state', default=42, type=int, help='Random data split seed.')
    parser.add_argument('--test', default=False, type=util.bool_flag)
    parser.add_argument('--exp_name', default="", type=str, help='Path to save logs, tensorboard and checkpoints.')
    parser.add_argument('--ckpt_name', default="", type=str, help='Path to save logs, tensorboard and checkpoints.')
    parser.add_argument('--log_name', default="RenalCLIP", type=str, help='log name')
    parser.add_argument('--resume', default=None, type=str, help='Path for checkpoint')
    parser.add_argument('--in_features', default=768, type=int, help='Input features')
    parser.add_argument('--num_classes', default=2, type=int, help='num classes')
    parser.add_argument('--finetune_type', default="train_from_scratch", help='type of finetune')
    parser.add_argument('--cuda', type=util.bool_flag, default=True, help='enables CUDA training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--downstream_task', default='BMC', type=str,help="type of downstream task")
    parser.add_argument('--test_only', default=False, type=util.bool_flag, help="whether to tarin & valid")
    parser.add_argument('--external_hospitals', default=(), type=util.parse_tuple, help="name of hospitals of external validation")
    parser.add_argument('--data_root', default='/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/RenalCLIP', type=str, help="data root")
    parser.add_argument('--dataset_names', default=('internal_downstream', 'XM', 'ZY', 'RJ', 'SD'), type=util.parse_tuple, help='data set ratio for datasets, train/valid/test')
    parser.add_argument('--ZY_IC', default='v1', type=str,help="type of ZY IC")

    # dataset
    parser.add_argument('--datalist_3d', default='/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/dataset_RenalCLIP.json', type=str, help='json for 3D datasets')
    parser.add_argument('--sample_ratio', default=1.0, type=float, help='sample ratio of training set')
    parser.add_argument('--split_file_name', default='../data/data_split_benchmark.json', type=str, help='json for 3D datasets')
    parser.add_argument('--a_min', default=-1000.0, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=3000, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--resize_3d', default=140, type=int, help='Resize x, y, z')
    parser.add_argument('--resize_slices', default=32, type=int, help='The slices num')
    parser.add_argument('--cropsize_3d', default=128, type=int, help='Augmentation parameters for 3D images')
    parser.add_argument('--crop_slices', default=32, type=int, help='The slices num')
    parser.add_argument("--RandRotated_range_in_plane", default=np.pi / 180.0 * 10, type=float, help="range_x of random rotation")
    parser.add_argument("--RandRotated_range_out_of_plane", default=np.pi / 180.0 * 10, type=float, help="range_y of random rotation")
    parser.add_argument("--RandTranslated_range_in_plane", default=10, type=int, help="range_z of random translation")
    parser.add_argument("--RandTranslated_range_out_of_plane", default=0, type=int, help="range_z of random translation")
    parser.add_argument("--RandRotated_prob", default=0.1, type=float, help="RandRotated aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--RandGaussianSmoothd_prob", default=0.1, type=float, help="RandGaussianSmoothd aug probability")

    parser.add_argument('--modalities', default=('A'), type=util.parse_tuple, help="input modalities")
    parser.add_argument('--modalities_pool', default=('N', 'A', 'V', 'D'), type=util.parse_tuple, help="input modalities")
    parser.add_argument('--multiple_3d', default=False, type=util.bool_flag, help="input modalities")

    parser.add_argument('--pre_processed', default=True, type=util.bool_flag, help="whether to use pre_processed images")
    parser.add_argument('--pre_processed_type', default="regular", type=str, help="type of pre_processed images, choices")

    # model
    parser.add_argument('--model_type', default='student', type=str, choices=['student', 'teacher'], help="model type")
    parser.add_argument('--pretrained_exp_name', default=None, type=str, help="clip name")
    parser.add_argument('--pretrained_metric', default=None, type=str, help="load which clip, best loss/acc or last ep")
    parser.add_argument('--split_pretrained_path', default=None, type=util.parse_dict)
    parser.add_argument('--clip_output_dim', default=128, type=int, help='The output dim in the projection head')
    parser.add_argument('--clip_hidden_dim', default=None, type=int, help='The hidden dim in the projection head')
    
    # MLP
    parser.add_argument('--dropout_rate', default=0.0, type=float)
    parser.add_argument('--hidden_dim', default=None, type=int)

    parser.add_argument("--max_lr", default=5e-4, type=float, help="Learning rate at the end of linear warmup (highest LR used during training).")
    parser.add_argument("--min_lr", default=1e-8, type=float, help="Learning rate at the end of linear warmup (highest LR used during training).")
    parser.add_argument('--weight_decay', type=float, default=0.005, help="""Initial value of the weight decay.""")
    parser.add_argument('--beta1', type=float, default=0.9, help="""Initial value of the weight decay.""")
    parser.add_argument('--beta2', type=float, default=0.999, help="""Initial value of the weight decay.""")

    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--val_iter', default=10, type=int)
    parser.add_argument('--fusion', default=False, type=util.bool_flag, help="whether to fuse 3D imgs to 4D")

    parser.add_argument('--multi_task_ids', default=(1, 2, 3, 7, 8, 9, 11, 13, 14, 15, 17, 18, 19, 20), type=util.parse_tuple, help="ids of multi tasks")
    parser.add_argument('--use_grad_checkpointing', default=False, type=util.bool_flag, help='whether to use checkpointing')
    parser.add_argument('--checkpoint_layers', default=0, type=int, help="beginning layers to enable gradient checkpoint")

    # optimizer / schedule
    parser.add_argument('--schedule', type=str, default='LWCA', help='')
    parser.add_argument('--cycle_steps', type=int, default=-1, help='')
    parser.add_argument('--cycle_mult', type=float, default=1.0, help='')
    parser.add_argument('--cycle_gamma', type=float, default=1.0, help='')

    # zeroshot
    parser.add_argument('--zeroshot_attributes', type=util.parse_tuple, default=('BMC', ), help='whether to enable zeroshot')
    parser.add_argument('--bert_type', default="emilyalsentzer/Bio_ClinicalBERT", type=str, help='choices of bert type')
    parser.add_argument('--avg_before_align', type=util.bool_flag, default=False, help='whether to enable zeroshot')
    parser.add_argument('--use_max_similarity', type=util.bool_flag, default=False, help='whether to enable zeroshot')

    args = parser.parse_args()

    args.downstream_path = os.path.join(fr'/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/downstream_output', args.finetune_type)
    args.experiment_path = os.path.join(args.downstream_path, args.exp_name)
    args.ckpt_path = os.path.join(args.downstream_path, args.ckpt_name)
    args.tfboard_path = os.path.join(args.downstream_path, 'tfboard', args.exp_name)
    args.val_iter = args.epochs // 10
    args.min_lr = (args.max_lr / 100.0)


    if "text_label" in args.downstream_task:
        TXT_LABEL_NUMS = {
            '1': 3,
            '2': 4,
            '3': 5,
            '7': 2,
            '8': 5,
            '9': 4,
            '11': 7,
            '13': 3,
            '14': 7,
            '15': 4,
            '17': 3,
            '18': 5,
            '19': 5,
            '20': 3,
        }
        txt_label = args.downstream_task.split('_')[-1]
        args.num_classes = TXT_LABEL_NUMS[txt_label]

    return args


def get_report_generation_args():
    parser = argparse.ArgumentParser()

    # Distributed training parameters
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_root', default='/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/RenalCLIP', type=str, help="data root")

    # 3D dataset parameters (with augmentation)
    parser.add_argument('--datalist_3d', default='/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/dataset_RenalCLIP.json', type=str, help='json for 3D datasets')
    parser.add_argument('--dataset_names', default=('internal_downstream', 'XM', 'ZY', 'RJ', 'SD'), type=util.parse_tuple, help='data set ratio for datasets, train/valid/test')
    parser.add_argument('--downstream_task', default='caption', type=str,help="type of downstream task")
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="data root")

    parser.add_argument('--split_file_name', default='/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/data/data_split_latest.json', type=str, help='json for 3D datasets')
    parser.add_argument('--a_min', default=-1000.0, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=3000, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--resize', default=False, type=util.bool_flag, help='Resize x, y, z')
    parser.add_argument('--resize_3d', default=140, type=int, help='Resize x, y, z')
    parser.add_argument('--resize_slices', default=32, type=int, help='The slices num')
    parser.add_argument('--max_words_3d', default=512, type=int, help='Max len of words')
    parser.add_argument('--sent_num_3d', default=3, type=int, help='Sent num')
    parser.add_argument('--cropsize_3d', default=128, type=int, help='Augmentation parameters for 3D images')
    parser.add_argument('--crop_slices', default=32, type=int, help='The slices num')
    parser.add_argument("--RandRotated_range_in_plane", default=np.pi / 180.0 * 10, type=float, help="range_x of random rotation")
    parser.add_argument("--RandRotated_range_out_of_plane", default=np.pi / 180.0 * 10, type=float, help="range_z of random rotation")
    parser.add_argument("--RandTranslated_range_in_plane", default=10, type=int, help="range_z of random translation")
    parser.add_argument("--RandTranslated_range_out_of_plane", default=0, type=int, help="range_z of random translation")
    parser.add_argument("--RandRotated_prob", default=0.1, type=float, help="RandRotated aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--RandGaussianSmoothd_prob", default=0.1, type=float, help="RandGaussianSmoothd aug probability")
    parser.add_argument('--pre_processed_type', default="one_kidney_tc_v3", type=str, help="type of pre_processed images, choices ['MONAI', 'regular', 'both_kidneys', 'one_kidney']")
    parser.add_argument('--report_type', default='full', type=str, choices=['full', 'impression', 'findings'], help='report type in CLIP pretraining')

    # Model parameters 
    parser.add_argument('--model_type', default='student', type=str, choices=['student', 'teacher'], help="model type")
    parser.add_argument('--use_grad_checkpointing', default=False, type=util.bool_flag, help='whether to use checkpointing')
    parser.add_argument('--img_encoder_type', default="cnn", type=str, help='type of image encoder, cnn or vit')
    parser.add_argument('--img_encoder_pretrained_weight', default=None, type=str, help='type of image encoder pretrained weight')
    parser.add_argument('--modalities', default=('N', 'A', 'V', 'D'), type=util.parse_tuple, help="input modalities")
    parser.add_argument('--checkpoint_layers', default=0, type=int, help="beginning layers to enable gradient checkpoint")
    parser.add_argument('--multiple_3d', default=False, type=util.bool_flag, help='whether to use proj after text encoder')
    
    parser.add_argument('--language_model_name', default="BioMistral/BioMistral-7B", type=str, help='whether to use proj after text encoder')
    parser.add_argument('--image_feature_dim', default=512, type=int, help='whether to use proj after text encoder')
    parser.add_argument('--num_image_tokens', default=8, type=int, help='whether to use proj after text encoder')
    parser.add_argument('--pretrained_type', default=None, type=str, help="pretrained model type")
    parser.add_argument('--pretrained_exp_name', default=None, type=str, help="clip name")
    parser.add_argument('--pretrained_metric', default=None, type=str, help="load which clip, best loss/acc or last ep")
    parser.add_argument('--finetune_type', default='finetune', type=str, help="load which clip, best loss/acc or last ep")

    # LoRA Args
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA rank (r).')
    parser.add_argument('--lora_alpha', type=int, default=4, help='LoRA alpha (scaling factor).')
    parser.add_argument('--lora_dropout', type=float, default=0.2, help='LoRA dropout rate.')

    # Training/Optimization parameters
    parser.add_argument('--stage_A_epochs', type=int, default=20)
    parser.add_argument('--stage_B_epochs', type=int, default=5)
    parser.add_argument('--val_epoch', default=2, type=int)
    parser.add_argument('--save_model', default=True, type=util.bool_flag, help='Number of epochs of training.')
    parser.add_argument('--test_only', default=False, type=util.bool_flag)

    parser.add_argument("--stage_A_adapter_lr", default=2e-4, type=float, help="lr of adapter.")
    parser.add_argument("--stage_B_adapter_lr", default=2e-4, type=float, help="lr of adapter.")
    parser.add_argument("--lora_lr", default=5e-4, type=float, help="lr of lora in llm.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Initial value of the weight decay.")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_freq', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--force_stage_A_retrain', type=util.bool_flag, default=False, help='If set, forces Stage A to retrain even if best adapter exists.')

    # Output parameters
    parser.add_argument('--exp_name', default="RenalCLIP_BASELINEv2", type=str, help='Path to save logs, tensorboard and checkpoints.')
    parser.add_argument('--log_name', default="RenalCLIP", type=str, help='log name')
    parser.add_argument('--ckpt_name', default="", type=str, help='Path to save logs, tensorboard and checkpoints.')
    args = parser.parse_args()

    args.downstream_path = os.path.join(fr'/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/downstream_output', args.finetune_type)
    args.experiment_path = os.path.join(args.downstream_path, args.exp_name)
    args.ckpt_path = os.path.join(args.downstream_path, args.ckpt_name)
    args.tfboard_path = os.path.join(args.downstream_path, 'tfboard', args.exp_name)

    create_experiment_dir(args)

    return args

def create_experiment_dir(args):
    if util.is_main_process():
        if not os.path.exists(args.experiment_path):
            os.makedirs(args.experiment_path, exist_ok=True)
            print('Create experiment path successfully at %s' % args.experiment_path)
        if not os.path.exists(args.tfboard_path):
            os.makedirs(args.tfboard_path, exist_ok=True)
            print('Create TFBoard path successfully at %s' % args.tfboard_path)
