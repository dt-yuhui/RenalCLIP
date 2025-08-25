# Most functions are reused from DINO

import os
import sys
import time
import math
import random
import argparse
import datetime
import subprocess
import warnings
from collections import defaultdict, deque
from utils.logger import *
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import json
from torch import nn
import torch.distributed as dist
from sklearn.metrics import recall_score, precision_recall_curve, auc, confusion_matrix, f1_score, roc_curve, roc_auc_score, precision_score
from collections.abc import Iterable


def load_state_with_same_shape(model, weights, model_type, load_module_name='image_encoder'):
    model_state = model.state_dict()

    # remove ddp
    if list(weights.keys())[0].startswith('module.'):
        weights = {k.partition('module.')[-1]: weights[k] for k in weights.keys()}

    if load_module_name == 'image_encoder':

        if model_type == 'student':
            load_weights = {k.replace('image_encoder_q_student.', ''): weights[k] for k in weights if k.startswith('image_encoder_q_student')}
        elif model_type == 'teacher':
            load_weights = {k.replace('image_encoder_q_teacher.', ''): weights[k] for k in weights if k.startswith('image_encoder_q_teacher')}
        else:
            raise NotImplementedError(f'model_type {model_type} not implemented!')

    elif load_module_name == 'text_encoder_model':
        if model_type == '':
            load_weights = {k.replace('text_encoder_q.model.', ''): weights[k] for k in weights if k.startswith('text_encoder_q.model')}
        else:
            load_weights = {k.replace(f'text_encoder_q_{model_type}.model.', ''): weights[k] for k in weights if k.startswith(f'text_encoder_q_{model_type}.model')}

    else:
        raise NotImplementedError(f"load_module_name: {load_module_name} is not implemented! Should in [image_encoder, text_encoder, fusion_model]")
    load_weights = {**load_weights}

    filtered_weights = {
        k: v for k, v in load_weights.items() if k in model_state and v.size() == model_state[k].size()
    }

    print(f"Loading {load_module_name} weights:" + ', '.join(filtered_weights.keys()))

    return filtered_weights


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms
                

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded {} from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def parse_tuple(value):
    return tuple(value.split(','))


def parse_dict(arg):
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Provided argument must be a valid JSON string.")


def fix_random_seeds(seed=42, pretrain=True):
    """
    Fix random seeds.
    """
    from monai.utils import set_determinism
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    set_determinism(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not pretrain:
        torch.use_deterministic_algorithms(True, warn_only=True)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_log(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB), logger=logger)
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)), logger=logger)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_log('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)), logger=logger)

    def log_every_list(self, iterable_list, print_freq, header=None):
        i = 0
        iterable2, iterable1 = iterable_list
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable1)))) + 'd'
        # space_fmt = ':d'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in zip(iterable2, iterable1):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable1) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable1) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable1), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable1), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable1)))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def remove_on_master(filename_to_remove):
    if is_main_process():
        os.remove(filename_to_remove)


def create_tfboard_on_master(tfboard_to_save):
    if is_main_process():
        writer = SummaryWriter(tfboard_to_save)

        return writer


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def _distributed_concat(tensor):
    gathered_tensors = torch.distributed.nn.all_gather(tensor)
    return torch.cat(gathered_tensors, dim=0)

def distributed_concat(inputs):
    if isinstance(inputs, dict):
        gathered_concat = {}
        for key in inputs:
            gathered_concat[key] = _distributed_concat(inputs[key])
        return gathered_concat
    else:
        return _distributed_concat(inputs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    dist.barrier()
    setup_for_distributed(args.rank == 0)


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def get_params(model, seperated=True):
    if seperated:
        img_params_to_update = []
        text_params_to_update = []
        scratch_params_to_update = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'image_encoder_q_student' in name: 
                if 'global_embedding' not in name:
                    img_params_to_update.append(param)
                else:
                    scratch_params_to_update.append(param)

            elif 'text_encoder_q_student' in name:
                if 'global_embedding' not in name:
                    text_params_to_update.append(param)
                else:
                    scratch_params_to_update.append(param)

            elif 'logit_scale' in name:
                scratch_params_to_update.append(param)

        return {'image_params': img_params_to_update, 'text_params': text_params_to_update, 'scratch_params': scratch_params_to_update}

    else:
        params_to_update = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            else:
                print(f"params: {name} to update")
                params_to_update.append(param)
                
        return {'params': params_to_update}


class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(self,
                 optimizer,
                 first_cycle_steps,
                 cycle_mult=1.,
                 max_lr=(0.1, 0.01),
                 min_lr=(0.001, 0.0001),
                 warmup_steps=0,
                 gamma=1.,
                 last_epoch=-1):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        min_lr_list = self.min_lr if isinstance(self.min_lr, tuple) else [self.min_lr] * len(self.optimizer.param_groups)
        for param_group, min_lr in zip(self.optimizer.param_groups, min_lr_list):
            param_group['lr'] = min_lr
            self.base_lrs.append(min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr, max_lr in zip(self.base_lrs, self.max_lr)]
        else:
            return [base_lr + (max_lr - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                                                                  (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr, max_lr in zip(self.base_lrs, self.max_lr)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = tuple(base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lr)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def get_metrics(logits_dict, labels_dict, task_info, detailed=False):
    results = {}

    for task_name, classes in task_info.items():
        logits = logits_dict[task_name]
        labels = labels_dict[task_name]

        logits = to_numpy(logits)
        labels = to_numpy(labels)

        # Ignore samples with the label -1
        valid_mask = labels != -1
        logits = logits[valid_mask]
        labels = labels[valid_mask]

        # If all labels are -1, skip the calculation for the current task
        if len(labels) == 0:
            continue

        actual_labels = np.unique(labels)
        
        # checks if the actual labels contain only 0 and 2
        # If so, it remaps the labels and adjusts the logits accordingly:
        # Label 2 is re-mapped to 1.
        # The third column of the logits (corresponding to label 2) is moved to the second column (corresponding to the new label 1).
        # The number of classes is adjusted to 2.
        
        if len(actual_labels) == 2 and 0 in actual_labels and 2 in actual_labels and classes == 3:
            print(f"Warning: Task {task_name} has only labels 0 and 2. Treating as a binary classification (0 vs 2).")
            
            labels = np.where(labels == 2, 1, labels)             
            logits_for_binary = np.stack([logits[:, 0], logits[:, 2]], axis=1)
            logits = logits_for_binary
            
            current_classes = 2
            
        else:
            current_classes = classes

        # Stable softmax calculation
        logits = logits - np.max(logits, axis=1, keepdims=True)
        softmax_logits = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        pred_labels = np.argmax(logits, axis=1)

        metrics_dict = {}

        cm = confusion_matrix(labels, pred_labels, labels=np.arange(current_classes))
        
        # calculate ROC AUC（One-vs-Rest）
        if current_classes == 2:
            try:
                roc_auc = roc_auc_score(labels, softmax_logits[:, 1])  # Use the probability of the positive class
                metrics_dict["roc_auc"] = roc_auc
            except ValueError as e:
                print(f"Error in calculating ROC AUC for task {task_name}: {e}")
                continue
        else:
            # calculate macro-average roc auc
            try:
                roc_auc = roc_auc_score(labels, softmax_logits, multi_class='ovr')
                metrics_dict["roc_auc"] = roc_auc
            except ValueError as e:
                print(f"Error in calculating ROC AUC for task {task_name}: {e}")
                continue

            if detailed:
                class_roc_auc = {}
                for i in range(current_classes): # 使用 current_classes
                    class_roc_auc[i] = roc_auc_score((labels == i).astype(int), softmax_logits[:, i])
                metrics_dict["roc_auc_classes"] = class_roc_auc
        
        if current_classes == 2:

            binarized_labels = np.zeros((labels.size, current_classes))
            binarized_labels[np.arange(labels.size), labels] = 1
            
            precisions, recalls, _ = precision_recall_curve(binarized_labels[:, 1], softmax_logits[:, 1])
            pr_auc = auc(recalls, precisions)
            
            TN = cm[0, 0]
            FP = cm[1, 0]
            specificity = TN / (TN + FP) if TN + FP != 0 else 0
            
            precision = precision_score(labels, pred_labels, zero_division=0)
            recall = recall_score(labels, pred_labels, zero_division=0)
            f1 = f1_score(labels, pred_labels, zero_division=0)

            metrics_dict.update({
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "specificity": specificity,
            })
        else:
            # calculate macro-average metrics
            binarized_labels = np.zeros((labels.size, current_classes))
            binarized_labels[np.arange(labels.size), labels] = 1
            
            pr_auc = {}
            for i in range(current_classes):
                precisions, recalls, _ = precision_recall_curve(binarized_labels[:, i], softmax_logits[:, i])
                pr_auc[i] = auc(recalls, precisions)
            pr_auc_macro = np.mean(list(pr_auc.values()))

            specificity = []
            for i in range(current_classes):
                TN = np.sum(cm) - (np.sum(cm[:, i]) + np.sum(cm[i, :]) - cm[i, i])
                FP = np.sum(cm[:, i]) - cm[i, i]
                specificity_i = TN / (TN + FP) if TN + FP != 0 else 0
                specificity.append(specificity_i)
            specificity_macro = np.mean(specificity)

            precision_macro = precision_score(labels, pred_labels, average='macro', zero_division=0)
            recall_macro = recall_score(labels, pred_labels, average='macro', zero_division=0)
            f1_macro = f1_score(labels, pred_labels, average='macro', zero_division=0)
            metrics_dict.update({
                "pr_auc": pr_auc_macro,
                "precision": precision_macro,
                "recall": recall_macro,
                "f1_score": f1_macro,
                "specificity": specificity_macro,
            })

        results[task_name] = metrics_dict
    
    return results


def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if is_iterable(item):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened
    

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    return x
