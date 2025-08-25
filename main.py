import os
import datetime
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.distributed.nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.util import *
from utils.logger import *
from utils.parser import get_clip_args
from models.RenalCLIP import RenalCLIP

from utils.data_util import custom_collate_fn_CLIP
from datasets.data_loader_RenalCLIP import DatasetRenalCLIP

torch.autograd.set_detect_anomaly(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def train_func(args):
    init_distributed_mode(args)

    global WORLD_SIZE
    WORLD_SIZE = torch.distributed.get_world_size()
    # seed for random, numpy, torch, python, monai
    fix_random_seeds(args.seed, pretrain=True)

    # logger 
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    print_log("==================== Parameters ====================", logger=logger)
    log_args_to_file(args, 'args', logger=logger)

    print_log("==================== preparing data ... ====================", logger=logger)

    train_ds = DatasetRenalCLIP(args, 'train')
    valid_ds = DatasetRenalCLIP(args, 'valid')

    train_collate_fn = custom_collate_fn_CLIP(args, 'train')
    valid_collate_fn = custom_collate_fn_CLIP(args, 'valid')

    print_log(f"==================== train dataset load complete ====================: {len(train_ds)} image-text pairs", logger=logger)
    print_log(f"==================== valid dataset load complete ====================: {len(valid_ds)} image-text pairs", logger=logger)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler=torch.utils.data.DistributedSampler(train_ds, shuffle=True),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate_fn,
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        sampler=torch.utils.data.DistributedSampler(valid_ds, shuffle=False),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=valid_collate_fn,
    )

    print_log("==================== building network ... ====================", logger=logger)


    model = RenalCLIP(
                        use_grad_checkpointing=args.use_grad_checkpointing,
                        bert_type=args.bert_type,
                        clip_output_dim=args.clip_output_dim,
                        clip_hidden_dim=args.clip_hidden_dim,
                        mode='pretrain',
                        modalities=args.modalities,
                        pretrained_img_encoder_weight=args.img_encoder_pretrained_weight,
                        max_words=args.max_words,
                        text_proj=args.text_proj,
                        llm2vec=args.llm2vec,
                        )
    # move networks to gpu
    model = model.cuda()

    '''
    you can freeze bert from last layer to first layer.
    set num of layer in config.yaml
    default is freeze 9 layers
    '''
    if not args.freeze_bert:
        if args.text_encoder_freeze_layers is not None:
            for layer_idx in range(int(args.text_encoder_freeze_layers)):
                for param in list(model.text_encoder_q_student.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
                    # print_log(f"Freezing text encoder layer {layer_idx}", logger=logger)
                else:
                    param.requires_grad = True
                    # print_log(f"Finetuning text encoder layer {layer_idx}", logger=logger)
    else:
        print_log("freezing bert", logger=logger)
        
        for name, param in model.text_encoder_q_student.named_parameters():
            if "global_embedding" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print_log(f"=========== trainable params in text encoder: {name} ===========", logger=logger)

    # synchronize batch norm (if any)
    if has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # DDP wrapper...
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, device_ids=[args.gpu])
    model._set_static_graph()

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    print_log("==================== preparing optimizer and schedulers ====================", logger=logger)

    update_params = get_params(model, seperated=True)
    params_groups = [
        {'params': update_params['image_params'], 'lr': args.base_lr_img},
        {'params': update_params['text_params'], 'lr': args.base_lr_text},
        {'params': update_params['scratch_params'], 'lr': args.base_lr_scratch},
    ]

    training_steps = len(train_dl) * args.epochs

    # clip loss & byol loss
    trainloss = TrainLoss()
    
    optimizer = torch.optim.AdamW(
        params_groups,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=training_steps,
        cycle_mult=1.0,
        max_lr=tuple([item['lr'] for item in params_groups], ),
        min_lr=tuple([max(item['lr'] / 100., 1e-6) for item in params_groups], ),
        warmup_steps=int(args.epochs * 0.1) * len(train_dl)
    )

    print_log("Loss, optimizer and schedulers ready.", logger=logger)

    to_restore = {"epoch": 0}
    if args.resume:
        print_log("============ optionally resume training ... ============", logger=logger)
        restart_from_checkpoint(
            args.resume,
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
        )

    start_epoch = to_restore["epoch"]

    print_log("Starting training !", logger=logger)

    writer = create_tfboard_on_master(os.path.join(args.tfboard_path))

    best_valid_stats = {'best_loss_epoch': -1,
                        'valid_loss': np.inf,
                        'best_acc_epoch': -1,
                        'valid_acc': -np.inf,
                        }

    for epoch in range(start_epoch, args.epochs):
        train_stats, valid_stats = train_valid_one_epoch(model,
                                                        trainloss,
                                                        train_dl,
                                                        valid_dl,
                                                        epoch,
                                                        best_valid_stats,
                                                        optimizer,
                                                        lr_scheduler,
                                                        fp16_scaler,
                                                        logger,
                                                        args,
                                                        writer
                                                        )

        valid_loss = valid_stats['loss']
        valid_acc = valid_stats['acc5']

        if valid_loss < best_valid_stats['valid_loss']:
            best_valid_stats['valid_loss'] = valid_loss
            best_valid_stats['best_loss_epoch'] = epoch

        if valid_acc > best_valid_stats['valid_acc']:
            best_valid_stats['valid_acc'] = valid_acc
            best_valid_stats['best_acc_epoch'] = epoch

def train_one_epoch(model, trainloss, data_loader, optimizer, lr_schedule, epoch, fp16_scaler, logger, args, writer):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    display_metrics_names = [
        'loss_ita_0',
        'loss_ita_1',
        'loss_ita',
        'loss',
        
        'i2t_acc1',
        't2i_acc1',
        'acc1',
        'i2t_acc5',
        't2i_acc5', 
        'acc5',
    ]

    for it, subjects_batch in enumerate(metric_logger.log_every(data_loader, 10, header, logger)):

        # move tensor to gpu
        subjects_batch['left_imgs'] = subjects_batch['left_imgs'].cuda(non_blocking=True)
        subjects_batch['right_imgs'] = subjects_batch['right_imgs'].cuda(non_blocking=True)
        
        tokenized_batch = model.module._tokenize_report(
            batch_raw_reports=subjects_batch['raw_reports'],
            mode='train'
        )
        subjects_batch['left_report_tokens_ids'] = tokenized_batch['left_report_tokens_ids'].cuda(non_blocking=True)
        subjects_batch['left_attention_mask'] = tokenized_batch['left_attention_mask'].cuda(non_blocking=True)
        subjects_batch['right_report_tokens_ids'] = tokenized_batch['right_report_tokens_ids'].cuda(non_blocking=True)
        subjects_batch['right_attention_mask'] = tokenized_batch['right_attention_mask'].cuda(non_blocking=True)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=fp16_scaler is not None, dtype=torch.bfloat16):
            result_dict = model.forward(subjects_batch, stage='train')
            loss_dict = trainloss(feat_dict=result_dict, stage='train', logit_scale=model.module.logit_scale)
            loss = loss_dict['loss']

        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = clip_gradients(model, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place

            if args.clip_grad:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)  # grad clip helps in both amp and fp32
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        global_steps = epoch * len(data_loader) + it

        if is_main_process():
            # tensorboard
            for k in display_metrics_names:
                v = loss_dict[k]
                if isinstance(v, torch.Tensor):
                    v = v.item()
                writer.add_scalar(f"train/{k}", v, global_steps)
            writer.add_scalar(f"train/softmax_temp", 1 / model.module.logit_scale.data.exp().item(), global_steps)

        # logging
        lr_schedule.step()
        metric_logger.update(loss_train=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_status = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return train_status


def valid_one_epoch(model, trainloss, data_loader, epoch, writer, fp16_scaler, args):
    metric_logger = MetricLogger(delimiter="  ")

    display_metrics_names = [
        'loss_ita',
        'loss',
        'i2t_acc1',
        't2i_acc1',
        'acc1',
        'i2t_acc5',
        't2i_acc5',
        'acc5',
    ]

    gather_all_results_dict = {
        'l_patient_ID': [],

        'l_report_emb_q_student': [],

        'l_img_feat_q_student_3D': [],
        'l_img_emb_q_student_3D': [],

        'r_patient_ID': [],

        'r_report_emb_q_student': [],

        'r_img_feat_q_student_3D': [],
        'r_img_emb_q_student_3D': [],
    }

    for it, subjects_batch in enumerate(data_loader):

        subjects_batch['left_imgs'] = subjects_batch['left_imgs'].cuda(non_blocking=True)
        subjects_batch['right_imgs'] = subjects_batch['right_imgs'].cuda(non_blocking=True)

        tokenized_batch = model.module._tokenize_report(
            batch_raw_reports=subjects_batch['raw_reports'],
            mode='valid'
        )
        subjects_batch['left_report_tokens_ids'] = tokenized_batch['left_report_tokens_ids'].cuda(non_blocking=True)
        subjects_batch['left_attention_mask'] = tokenized_batch['left_attention_mask'].cuda(non_blocking=True)
        subjects_batch['right_report_tokens_ids'] = tokenized_batch['right_report_tokens_ids'].cuda(non_blocking=True)
        subjects_batch['right_attention_mask'] = tokenized_batch['right_attention_mask'].cuda(non_blocking=True)

        results_dict = model.forward(subjects_batch, stage='valid')

        for key in results_dict.keys():
            gather_all_results_dict[key].append(results_dict[key])

        gather_all_results_dict['l_patient_ID'].append(subjects_batch['l_patient_ID'])
        gather_all_results_dict['r_patient_ID'].append(subjects_batch['r_patient_ID'])

    # gather features from all gpus
    for key in gather_all_results_dict.keys():
        if 'patient_ID' not in key:
            if not all(element is None for element in gather_all_results_dict[key]):

                if gather_all_results_dict[key] and isinstance(gather_all_results_dict[key][0], torch.Tensor):
                    # print(gather_all_results_dict[key][0])
                    gather_all_results_dict[key] = torch.cat(gather_all_results_dict[key], dim=0)
                    gather_all_results_dict[key] = distributed_concat(gather_all_results_dict[key])

                elif gather_all_results_dict[key] and isinstance(gather_all_results_dict[key][0], dict):
                    # 如果是字典类型的，单独处理每个键
                    dict_keys = gather_all_results_dict[key][0].keys()
                    combined_dict = {k: [] for k in dict_keys}
                    for mini_batch_dict in gather_all_results_dict[key]:
                        for k in dict_keys:
                            combined_dict[k].append(mini_batch_dict[k])  # 使用 append 而不是 extend
                    for k in dict_keys:
                        if combined_dict[k]:  # 确保 combined_dict[k] 不为空
                            # 拼接 combined_dict[k] 中的每一个子列表
                            combined_dict[k] = torch.cat([torch.cat(sublist, dim=0) if isinstance(sublist, list) else sublist for sublist in combined_dict[k]], dim=0)
                            combined_dict[k] = distributed_concat(combined_dict[k])
                    gather_all_results_dict[key] = combined_dict
        
            else:
                gather_all_results_dict[key] = None
        else:
            gather_pid_list = [None for _ in range(WORLD_SIZE)]
            gather_all_results_dict[key] = flatten_list(gather_all_results_dict[key])
            torch.distributed.all_gather_object(gather_pid_list, gather_all_results_dict[key])
            gather_all_results_dict[key] = flatten_list(gather_pid_list)

    loss_dict = trainloss(feat_dict=gather_all_results_dict, logit_scale=model.module.logit_scale, stage='valid')

    if is_main_process():
        # tensorboard
        for k in display_metrics_names:
            v = loss_dict[k]
            if isinstance(v, torch.Tensor):
                v = v.item()
            writer.add_scalar(f"valid/{k}", v, epoch)

    return loss_dict


def train_valid_one_epoch(model,
                          trainloss,
                          train_loader,
                          valid_loader,
                          epoch,
                          best_valid_stats,
                          optimizer,
                          lr_schedule,
                          fp16_scaler,
                          logger,
                          args,
                          writer,
                          ):
    start_time = time.time()
    # train
    model.train()
    train_stats = train_one_epoch(model,
                                  trainloss,
                                  train_loader,
                                  optimizer,
                                  lr_schedule,
                                  epoch,
                                  fp16_scaler,
                                  logger,
                                  args,
                                  writer,
                                  )

    # valid
    model.eval()
    with torch.no_grad():
        valid_stats = valid_one_epoch(
            model=model,
            trainloss=trainloss,
            data_loader=valid_loader,
            epoch=epoch,
            writer=writer,
            fp16_scaler=fp16_scaler,
            args=args,
        )
        # valid_acc = valid_stats['acc1']
        valid_acc = valid_stats['acc5']
        valid_loss = valid_stats['loss']

    save_dict = {
        'model': model.state_dict(),
    }
    if fp16_scaler is not None:
        save_dict['fp16_scaler'] = fp16_scaler.state_dict()

    # save ckpt every epoch
    save_on_master(save_dict, os.path.join(args.experiment_path, f'{args.exp_name}-latest.pth'))

    # Save best model
    pre_best_valid_loss, pre_best_valid_loss_epoch = best_valid_stats['valid_loss'], best_valid_stats['best_loss_epoch']
    pre_best_valid_acc, pre_best_valid_acc_epoch = best_valid_stats['valid_acc'], best_valid_stats['best_acc_epoch']
    if valid_loss < pre_best_valid_loss:
        filename = os.path.join(args.experiment_path, f"{args.exp_name}-model-best-loss.pt")
        save_on_master(save_dict, filename)

    if valid_acc > pre_best_valid_acc:
        filename = os.path.join(args.experiment_path, f"{args.exp_name}-model-best-acc.pt")
        save_on_master(save_dict, filename)

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'valid_{k}': v for k, v in valid_stats.items()},
                 'epoch': epoch}
    if is_main_process():
        log_loss_to_file(log_stats, logger=logger)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_log('Training ' + ' time {}'.format(total_time_str))

    return train_stats, valid_stats


class TrainLoss(nn.Module):
    def __init__(self, ):
        super().__init__()

    def _clip_loss(self, scores):
        # when using InfoNCE-like loss
        # one=hot label
        bz = scores.size(0)
        label = torch.arange(bz, device=scores.device).long()

        image_loss = F.cross_entropy(scores, label)
        caption_loss = F.cross_entropy(scores.T, label)

        return image_loss, caption_loss, (image_loss + caption_loss) / 2

    # CLIP
    def forward_clip_loss(self, feat_dict, logit_scale):
        l_report_emb_q_student = feat_dict['l_report_emb_q_student']
        l_img_emb_q_student = feat_dict['l_img_emb_q_student_3D'] if feat_dict['l_img_emb_q_student_3D'] is not None else None

        r_report_emb_q_student = feat_dict['r_report_emb_q_student']
        r_img_emb_q_student = feat_dict['r_img_emb_q_student_3D'] if feat_dict['r_img_emb_q_student_3D'] is not None else None

        report_emb_q_student = torch.cat([l_report_emb_q_student, r_report_emb_q_student], dim=0)
        img_emb_q_student = torch.cat([l_img_emb_q_student, r_img_emb_q_student], dim=0)

        loss_ita_0, loss_ita_1 = 0, 0
        img_emb_q_global_student = distributed_concat(img_emb_q_student)
        report_emb_q_global_student = distributed_concat(report_emb_q_student)

        scores = img_emb_q_global_student.mm(report_emb_q_global_student.t())
        scores *= logit_scale
        _, _, loss_ita_0 = self._clip_loss(scores=scores)

        return loss_ita_0, loss_ita_1

    def img_text_retrieval_precision(self, feat_dict, stage, logit_scale, top_k=(1,)):

        def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=top_k):
            ''' Compute the accuracy over the k top predictions for the specified values of k'''
            with torch.no_grad():
                maxk = max(top_k)
                batch_size = target.size(0)

                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                res = []
                correct_indices = {k: [] for k in top_k}
                for k in top_k:
                    correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(100.0 / batch_size))

                    # Record correct indices for top-k
                    correct_k_indices = torch.nonzero(correct[:k].contiguous(), as_tuple=False)[:, 1].flatten()
                    correct_indices[k] = correct_k_indices.tolist()

                return res, correct_indices
            
        with torch.no_grad():

            l_report_emb_q_student = feat_dict['l_report_emb_q_student']
            l_img_emb_q_student = feat_dict['l_img_emb_q_student_3D'] if feat_dict['l_img_emb_q_student_3D'] is not None else None

            r_report_emb_q_student = feat_dict['r_report_emb_q_student']
            r_img_emb_q_student = feat_dict['r_img_emb_q_student_3D'] if feat_dict['r_img_emb_q_student_3D'] is not None else None

            img_emb_q_student = torch.cat([l_img_emb_q_student, r_img_emb_q_student], dim=0)
            report_emb_q_student = torch.cat([l_report_emb_q_student, r_report_emb_q_student], dim=0)

            if stage == 'train':
                img_emb_q_global_student = distributed_concat(img_emb_q_student)
                report_emb_q_global_student = distributed_concat(report_emb_q_student)

            else:
                img_emb_q_global_student = img_emb_q_student
                report_emb_q_global_student = report_emb_q_student
                l_patient_ids = feat_dict['l_patient_ID']
                r_patient_ids = feat_dict['r_patient_ID']
                patient_ids = np.concatenate([l_patient_ids, r_patient_ids], 0)

            scores = img_emb_q_global_student.mm(report_emb_q_global_student.t())
            scores *= logit_scale
            loss_i2t, loss_t2i, loss_ita = self._clip_loss(scores=scores)

            # convert type
            scores = scores.to(dtype=torch.float32)
            scores1 = scores.transpose(0, 1)
            bz = scores.size(0)
            labels = torch.arange(bz, device=scores.device).long()

            # acc@1
            i2t_acc1, correct_i2t_indices_top1 = precision_at_k(scores, labels, top_k=[1])
            t2i_acc1, correct_t2i_indices_top1 = precision_at_k(scores1, labels, top_k=[1])
            i2t_acc1, t2i_acc1 = i2t_acc1[0], t2i_acc1[0]

            # acc@5
            i2t_acc5, correct_i2t_indices_top5 = precision_at_k(scores, labels, top_k=[5])
            t2i_acc5, correct_t2i_indices_top5 = precision_at_k(scores1, labels, top_k=[5])
            i2t_acc5, t2i_acc5 = i2t_acc5[0], t2i_acc5[0]

            # average
            acc1 = (i2t_acc1 + t2i_acc1) / 2.
            acc5 = (i2t_acc5 + t2i_acc5) / 2.

            return_dict = {
                'i2t_acc1': i2t_acc1.item(),
                't2i_acc1': t2i_acc1.item(),
                'acc1': acc1.item(),
                'i2t_acc5': i2t_acc5.item(),
                't2i_acc5': t2i_acc5.item(),
                'acc5': acc5.item(),
                'loss_ita': loss_ita.item(),
            }

            return return_dict


    def forward(self, feat_dict, logit_scale, stage='train'):
        logit_scale = logit_scale.exp()
        # clip
        if stage == 'train':
            loss_ita_0, loss_ita_1 = self.forward_clip_loss(feat_dict=feat_dict, logit_scale=logit_scale)
            loss_ita = loss_ita_0 + loss_ita_1
            train_dict = self.img_text_retrieval_precision(feat_dict=feat_dict, stage=stage, top_k=(1, 5), logit_scale=logit_scale)
            i2t_acc1, t2i_acc1, acc1, i2t_acc5, t2i_acc5, acc5, _ = train_dict.values()
        else:
            # retrieval
            val_dict = self.img_text_retrieval_precision(feat_dict=feat_dict, stage=stage, top_k=(1, 5), logit_scale=logit_scale)
            i2t_acc1, t2i_acc1, acc1, i2t_acc5, t2i_acc5, acc5, loss_ita = val_dict.values()
            loss_ita_0, loss_ita_1 = 0, 0
            loss_ita = loss_ita

        loss = loss_ita

        return_dict = {
            'loss_ita_0': loss_ita_0,
            'loss_ita_1': loss_ita_1,
            'loss_ita': loss_ita,
            'loss': loss,
            'i2t_acc1': i2t_acc1,
            't2i_acc1': t2i_acc1,
            'acc1': acc1,
            'i2t_acc5': i2t_acc5,
            't2i_acc5': t2i_acc5,
            'acc5': acc5
        }


        return return_dict

if __name__ == '__main__':
    args = get_clip_args()
    train_func(args)
