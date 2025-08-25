import os

import numpy as np
import torch
import time
import argparse
import sys
from models.finetuner import *
from models.RenalModel import RenalModel
import torch.nn.functional as F
import torch.nn as nn

from utils.util import *
from utils.logger import *
from datasets.data_loader_RenalCLIP_downstream_img import DatasetRenalCLIPDownstreamImg
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from utils.data_util import custom_collate_fn_downstream_img
from utils.parser import get_downstream_args_img
import pandas as pd

import csv

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


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)

    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
    else:
        # 清空 TFBoard 文件夹
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
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    log_args_to_file(args, 'args', logger=logger)

    decorated_main = log_exceptions(logger_name=args.log_name, log_file=log_file)(main_func)
    decorated_main(args, logger)

def main_func(args, logger):
    # fix seeds
    fix_random_seeds(args.seed, pretrain=False)

    # dataset & dataloader
    train_ds = DatasetRenalCLIPDownstreamImg(args, 'train')

    if hasattr(train_ds, "sample_weights"):
        train_sampler = WeightedRandomSampler(
                                            weights=train_ds.sample_weights,
                                            num_samples=len(train_ds),
                                            replacement=True)
    else:
        train_sampler = RandomSampler(train_ds)

    train_dataloader = DataLoader(
        train_ds,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn_downstream_img(args),
    )

    print_log(f"==================== train dataset load complete ====================: {len(train_ds)} image-text pairs", logger=logger)

    train_label_distribution = train_ds.task_distribution
    log_dict_to_logger(train_label_distribution, pre="train data distribution", logger=logger)

    # device
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print_log('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices', logger=logger)
    else:
        print_log('Using CPU', logger=logger)

    backbone = RenalModel(  mode=args.finetune_type,
                            pretrained_exp_name=args.pretrained_exp_name,
                            pretrained_metric=args.pretrained_metric,
                            model_type=args.model_type,
                            logger=logger,
                            )
                            
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.downstream_task != 'multi_task':
        task_info = {args.downstream_task: args.num_classes}
    else:
        multi_task_ids = args.multi_task_ids
        task_info = {}
        for multi_task in multi_task_ids:
            task_name = f"text_label_{multi_task}"
            task_classes = TXT_LABEL_NUMS[f"{multi_task}"]
            task_info.update({task_name: task_classes})
    
    finetuner = FineTuner(backbone=backbone,
                          in_features=args.in_features,
                          dropout=args.dropout_rate,
                          hidden_dim=args.hidden_dim,
                          finetune_type=args.finetune_type,
                          task_info=task_info,
                          )

    finetuner = finetuner.to(device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    print_log("==================== optimizer ====================", logger=logger)

    # optimizer
    update_params = get_params(finetuner, False)
    params_groups = [{'params': update_params['params'], 'lr': args.max_lr}]

    optimizer = torch.optim.AdamW(
        params_groups,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    training_steps = len(train_dataloader) * args.epochs

    if args.schedule == 'LWCA':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.epochs if args.cycle_steps < 0 else args.cycle_steps,
            cycle_mult=args.cycle_mult,
            max_lr=(args.max_lr, ),
            min_lr=(args.min_lr, ),
            warmup_steps=min(5, int(args.epochs * 0.1)),
            gamma=args.cycle_gamma,
        )
    elif args.schedule == 'Exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_steps*0.2, gamma=0.5)

    # tensorboard
    writer = create_tfboard_on_master(os.path.join(args.tfboard_path))

    # ce weight
    ce_weight = train_ds.class_weights.cuda() if hasattr(train_ds, "class_weights") else None
    criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='sum')

    # train
    if not args.test_only:
        print_log("==================== start training ====================", logger=logger)
        for epoch in range(args.epochs):
          
            train_model(
                finetuner=finetuner,
                data_loader=train_dataloader,
                task_info=task_info,
                optimizer=optimizer,
                criterion=criterion,
                logger=logger,
                writer=writer,
                args=args,
                epoch=epoch,
                )
            
            scheduler.step()

            save_dict = {
                'model': finetuner.state_dict(),
            }

            # save last ep ckpt
            if (epoch+1) == args.epochs:
                torch.save(save_dict, os.path.join(args.experiment_path, 'best-checkpoint.pth'))
            
            # test
            if ((epoch+1) % args.val_iter == 0) or (epoch == 0):
                with torch.no_grad():
                    print_log("==================== start testing ====================", logger=logger)
                    finetuner.eval()
                    # internal validation
                    hospital = 'internal'
                    test_model(
                            finetuner=finetuner,
                            task_info=task_info,
                            hospital=hospital,
                            logger=logger,
                            writer=writer,
                            args=args,
                            epoch=epoch
                            )

                    # external validation
                    for external_hospital in args.external_hospitals:
                        test_model(finetuner=finetuner,
                            task_info=task_info,
                            hospital=external_hospital,
                            logger=logger,
                            writer=writer,
                            args=args,
                            epoch=epoch
                            )

    # test
    # load pre-trained img encoder
    if args.split_pretrained_path is not None:
        # For late fusion.
        # Each modality has its own checkpoint.
        # While these checkpoints can be either the same or different, we use the same one for all modalities in this work.
        finetuner_dict = {}
        for modality, pretrained_path in args.split_pretrained_path.items():
            finetuner = FineTuner(
                        backbone=backbone,
                        in_features=args.in_features,
                        dropout=args.dropout_rate,
                        hidden_dim=args.hidden_dim,
                        finetune_type=args.finetune_type,
                        task_info=task_info,
                        )

            finetuner = finetuner.to(device)
            checkpoint = torch.load(os.path.join(args.downstream_path, pretrained_path, 'best-checkpoint.pth'))
            finetuner.load_state_dict(checkpoint['model'])
            finetuner.eval()
            finetuner_dict[modality] = finetuner
    
    # standard inference, using only one enhanced phase
    else:
        ckpt_path = args.ckpt_path if args.test_only else args.experiment_path
        checkpoint = torch.load(os.path.join(ckpt_path, 'best-checkpoint.pth'))
        finetuner.load_state_dict(checkpoint['model'])
        finetuner.eval()

    with torch.no_grad():
        # internal validation
        hospital = 'internal'
        test_model(
                finetuner=finetuner if args.split_pretrained_path is None else finetuner_dict,
                task_info=task_info,
                hospital=hospital,
                logger=logger,
                writer=writer,
                args=args,
                epoch=args.epochs,
                save_flag=True,
                )

        # external validation
        for external_hospital in args.external_hospitals:
            test_model(finetuner=finetuner if args.split_pretrained_path is None else finetuner_dict,
                task_info=task_info,
                hospital=external_hospital,
                logger=logger,
                writer=writer,
                args=args,
                epoch=args.epochs,
                save_flag=True,
                )

                        
def train_model(finetuner, data_loader, task_info, optimizer, criterion, logger, writer, args, epoch):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    finetuner.train()

    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header, logger)):
        imgs = batch['imgs'].cuda()

        # Check if y is a dictionary (multi-task situation) and handle appropriately
        if isinstance(batch['labels'], dict):
            for key in batch['labels']:
                batch['labels'][key] = batch['labels'][key].cuda().squeeze()
        else:
            batch['labels'] = batch['labels'].cuda().squeeze()

        y = batch['labels']
        logits = finetuner(imgs)

        task_losses = {}
        for task in task_info.keys():
            task_logits = logits[task].float()
            task_labels = y[task].long()
            
            # Create a mask that identifies non-missing labels efficiently
            mask = task_labels != -1

            valid_task_labels = task_labels[mask]
            valid_task_logits = task_logits[mask]

            if mask.any():  # Only compute loss if there are any non-missing labels
                task_loss = criterion(valid_task_logits, valid_task_labels)
                task_losses[task] = task_loss / mask.sum()  # Normalize loss by the number of non-missing labels

        # Combine task losses as needed, for example by summing them up
        total_loss = sum(task_losses.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        res_dict = get_metrics(logits_dict=logits, labels_dict=y, task_info=task_info)
        global_steps = epoch * len(data_loader) + it

        # 初始化平均metrics
        avg_metrics = {
            "loss": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
            "precision": 0.0,
        }

        # log metrics for each task
        for task, metrics in res_dict.items():
            task_loss = task_losses[task].item()
            writer.add_scalar(f"{task}/loss", task_loss, global_steps)
            for metric, value in metrics.items():
                writer.add_scalar(f"{task}/{metric}", value, global_steps)
            
            avg_metrics["loss"] += task_loss
            avg_metrics["recall"] += metrics["recall"]
            avg_metrics["precision"] += metrics["precision"]
            avg_metrics["f1_score"] += metrics["f1_score"]
            avg_metrics["pr_auc"] += metrics["pr_auc"]
            avg_metrics["roc_auc"] += metrics["roc_auc"]
            avg_metrics["precision"] += metrics["precision"]

        num_tasks = len(task_info)
        for metric in avg_metrics:
            avg_metrics[metric] /= num_tasks

        for metric, value in avg_metrics.items():
            writer.add_scalar(f"average/{metric}", value, global_steps)

        log_loss_to_file(avg_metrics, logger=logger)


def test_model(finetuner, task_info, hospital, logger, writer, args, epoch, save_flag=False):
        if args.split_pretrained_path is not None:
            # {N: model_N, A: model_A, V: model_V, D: model_D}
            for v in finetuner.values():
                v.eval()
        else:
            finetuner.eval()
        test_ds = DatasetRenalCLIPDownstreamImg(args, 'test', hospital=hospital)
        test_dataloader = DataLoader(
            test_ds,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn_downstream_img(args),
        )
        print_log(f"==================== {hospital} test dataset load complete ====================: {len(test_ds)} images", logger=logger)
        test_label_distribution = test_ds.task_distribution
        log_dict_to_logger(test_label_distribution, pre=f"{hospital} test data distribution", logger=logger)

        logits_dict = {task: [] for task in task_info.keys()}
        labels_dict = {task: [] for task in task_info.keys()}
        loss_dict = {task: [] for task in task_info.keys()}
        pid_list = []
        result_list = []

        with torch.no_grad():
            test_metric_logger = MetricLogger(delimiter="  ")
            header = 'Test:'
            
            for batch in test_metric_logger.log_every(test_dataloader, 10, header, logger):
                pid = batch['patients']
                imgs = batch['imgs'].cuda()
                
                if isinstance(batch['labels'], dict):
                    for key in batch['labels']:
                        batch['labels'][key] = batch['labels'][key].cuda()
                else:
                    batch['labels'] = batch['labels'].cuda()
                     
                if args.split_pretrained_path is not None:
                    # Late fusion is performed by averaging the logits of the available phases.
                    # A modality_mask is used to control which phases are available.
                    if imgs.dim() == 5:  
                        imgs = imgs.unsqueeze(1)
                    modality_mask = ~torch.isnan(imgs).all(dim=(2, 3, 4, 5))
                    B, T, C, D, H, W = imgs.shape
                    logits = {task: torch.zeros(B, args.num_classes * T).cuda() for task in task_info.keys()}
                    for t in range(T):
                        current_input = imgs[:, t]
                        current_mask = modality_mask[:, t]
                        valid_indexes = current_mask.nonzero(as_tuple=True)[0]
                        valid_input = current_input[current_mask]

                        current_finetuner = finetuner[args.modalities[t]]
                        logit = current_finetuner(valid_input)

                        for task in task_info.keys():
                            task_logits = logit[task]
                            logits[task][valid_indexes, t * args.num_classes:(t + 1) * args.num_classes] = task_logits
                    
                    for task in task_info.keys():
                        logits[task] = logits[task].view((B, T, args.num_classes))
                        logits_sum = logits[task].sum(dim=1)
                        logits_non_zero_sum = modality_mask.sum(dim=1, keepdim=True)
                        logits[task] = logits_sum / logits_non_zero_sum

                else:
                    # standard inference
                    logits = finetuner(imgs)

                y = batch['labels']
                
                if isinstance(y, dict):
                    for key in y:
                        y[key] = y[key].squeeze()
                else:
                    y = y.squeeze()
                
                task_losses = {}
                for task in task_info.keys():
                    task_logits = logits[task].float()
                    task_labels = y[task].long()

                    mask = task_labels != -1

                    valid_task_labels = task_labels[mask]
                    valid_task_logits = task_logits[mask]
                    valid_task_logits = valid_task_logits.squeeze(0)

                    task_loss = F.cross_entropy(valid_task_logits, valid_task_labels)
                    task_losses[task] = task_loss
                    
                    logits_dict[task].append(valid_task_logits.detach().cpu().numpy())
                    labels_dict[task].append(valid_task_labels.detach().cpu().numpy())
                    loss_dict[task].append(task_losses[task].item())

                    preds = torch.argmax(valid_task_logits, dim=1)
                    correct = preds == valid_task_labels
                    task_logits_list = valid_task_logits.detach().cpu().numpy().tolist()
                    result_list.extend(zip(pid, valid_task_labels.cpu().numpy(), preds.cpu().numpy(), correct.cpu().numpy(), task_logits_list))
                    
                pid_list.append(pid)
            
            # calculate metrics
            for task in task_info.keys():
                logits_dict[task] = np.concatenate(logits_dict[task], axis=0)
                labels_dict[task] = np.concatenate(labels_dict[task], axis=0)
                loss_dict[task] = np.mean(loss_dict[task])
            
            pid_list = np.concatenate(pid_list, axis=0)
            
            detailed = True if args.downstream_task != 'multi_task' else False
            test_res_dict = get_metrics(logits_dict=logits_dict, labels_dict=labels_dict, task_info=task_info, detailed=detailed)
            
            for task in test_res_dict.keys():
                test_res_dict[task]["loss"] = loss_dict[task]
            
            avg_metrics = {metric: 0.0 for metric in test_res_dict[next(iter(test_res_dict))].keys()}

            # save results to tensorboard
            for task, metrics in test_res_dict.items():
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            writer.add_scalar(f"test/{task}/{hospital}/{metric}/{sub_metric}", sub_value, epoch)
                    else:
                        writer.add_scalar(f"test/{task}/{hospital}/{metric}", value, epoch)
                    
                    if not isinstance(value, dict):
                        avg_metrics[metric] += value
            
            for metric in avg_metrics:
                avg_metrics[metric] /= len(task_info)
            
            for metric, value in avg_metrics.items():
                writer.add_scalar(f"test/average/{hospital}/{metric}", value, epoch)

            if save_flag:
                # Save results to a CSV file.
                with open(os.path.join(args.logits_path, f'{hospital}_test_results.csv'), mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['PID', 'Label', 'Pred Label', 'Correct', 'Logits'])
                    writer.writerows(result_list)


if __name__ == '__main__':
    args = get_downstream_args_img()

    main(args)
