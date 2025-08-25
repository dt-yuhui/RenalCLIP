import os

import numpy as np
import torch
import time
import argparse
import sys
from models.finetuner import *
from models.RenalModel import RenalModel

from utils.util import *
from utils.logger import *
from utils.parser import get_downstream_args_img
import pandas as pd

from zero_shot.zeroshot_val import zeroshot_one_hospital, zeroshot_setup, load_text_embeddings_from_disk
from datasets.data_loader_RenalCLIP_zeroshot import DatasetRenalCLIPZeroshot
from collections import OrderedDict


TEXT_EMBEDDINGS_ROOT = fr"/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/RenalCLIP/zeroshot_text_embeddings"
CUSTOM_PRETRAINED_DIR = fr"/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/clip_output"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)

    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
    else:
        for filename in os.listdir(args.tfboard_path):
            file_path = os.path.join(args.tfboard_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print('Deleted file: %s' % file_path)

    # save logits
    args.logits_path = os.path.join(args.experiment_path, "logits")
    if not os.path.exists(args.logits_path):
        os.makedirs(args.logits_path)
        print('Create logits path successfully at %s' % args.logits_path)


def main(args):
    create_experiment_dir(args)
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}_zeroshot.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    log_args_to_file(args, 'args', logger=logger)
    decorated_main = log_exceptions(logger_name=args.log_name, log_file=log_file)(main_func)
    decorated_main(args, logger)

def main_func(args, logger):
    fix_random_seeds(args.seed, pretrain=False)

    writer = create_tfboard_on_master(os.path.join(args.tfboard_path))

    text_embed_name = 'llm2vec-rad'
    image_embed_name = 'ours'
    
    avg_before_align = args.avg_before_align
    use_max_similarity = args.use_max_similarity
    text_embeddings_dict = load_text_embeddings_from_disk(os.path.join(TEXT_EMBEDDINGS_ROOT, text_embed_name), avg_before_align)

    with torch.no_grad():
        zeroshot_tokenizer, zeroshot_templates, zeroshot_attributes = zeroshot_setup(args)
        task_info = {attr: len(zeroshot_attributes[attr]) for attr in zeroshot_attributes}

        zeroshot_ds_int_test = DatasetRenalCLIPZeroshot(args, 'test', hospital='internal')
        zeroshot_int_test_dl = torch.utils.data.DataLoader(
            zeroshot_ds_int_test,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        _ = zeroshot_one_hospital(
            dataloader=zeroshot_int_test_dl,
            task_info=task_info,
            hospital='internal',
            writer=writer,
            image_embed_name=image_embed_name,
            text_embeddings_dict=text_embeddings_dict,
            avg_before_align=avg_before_align,
            use_max_similarity=use_max_similarity,
        )

        for hospital in args.external_hospitals:
            zeroshot_ds_ext_test = DatasetRenalCLIPZeroshot(args, 'test', hospital=hospital)

            zeroshot_ext_test_dl = torch.utils.data.DataLoader(
                zeroshot_ds_ext_test,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            _ = zeroshot_one_hospital(
                dataloader=zeroshot_ext_test_dl,
                task_info=task_info,
                hospital=hospital,
                writer=writer,
                image_embed_name=image_embed_name,
                text_embeddings_dict=text_embeddings_dict,
                avg_before_align=avg_before_align,
                use_max_similarity=use_max_similarity,
            )


if __name__ == '__main__':
    args = get_downstream_args_img()
    main(args)
