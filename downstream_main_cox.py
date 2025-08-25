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

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
    

class CoxPHNet(nn.Module):
    def __init__(self, 
                 backbone,
                 dropout_rate=0,
                 n_hidden=128,
                 input_dim=512,
                 ):
        super().__init__()

        self.backbone = backbone
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)
    

class IdentityNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x


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

    args.logits_path = os.path.join(args.experiment_path, "logits")
    if not os.path.exists(args.logits_path):
        os.makedirs(args.logits_path)
        print('Create logits path successfully at %s' % args.logits_path)


def main(args):
    create_experiment_dir(args)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    log_args_to_file(args, 'args', logger=logger)

    decorated_main = log_exceptions(logger_name=args.log_name, log_file=log_file)(main_func)
    decorated_main(args, logger)

def main_func(args, logger):
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn_downstream_img(args),
        sampler=train_sampler
    )

    _train_ds = DatasetRenalCLIPDownstreamImg(args, split='train', hospital='internal', transforms_mode='test')
    _train_dataloader = DataLoader(
        _train_ds,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        batch_size=args.batch_size,
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

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    print_log("==================== optimizer ====================", logger=logger)
    
    model = CoxPHNet(
        backbone=backbone, 
        dropout_rate=args.dropout_rate,
        input_dim=args.in_features).to(device)
    
    identity_net = IdentityNet()
    cox_model = CoxPH(identity_net, optimizer=None)
    criterion = cox_model.loss  # Using Pycox's loss function, partial_log_likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs if args.cycle_steps < 0 else args.cycle_steps,
        cycle_mult=args.cycle_mult,
        max_lr=(args.max_lr, ),
        min_lr=(args.min_lr, ),
        warmup_steps=min(5, args.epochs * 0.1),
        gamma=args.cycle_gamma,
    )

    # tensorboard
    writer = create_tfboard_on_master(os.path.join(args.tfboard_path))

    # train & valid
    if not args.test_only:
        print_log("==================== start training ====================", logger=logger)
        for epoch in range(args.epochs):
            
            if ((epoch+1) % args.val_iter == 0) or (epoch == 0):
                valid_flag = True
            else:
                valid_flag = False
            
            train_model(
                model=model,
                cox_model=cox_model,
                data_loader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                logger=logger,
                writer=writer,
                args=args,
                epoch=epoch,
                _data_loader=_train_dataloader if valid_flag else None
                )

            scheduler.step()

            save_dict = {
                'model': model.state_dict(),
                'cox_model': cox_model,
                'optimizer': optimizer.state_dict(),
                'args': args,
                'epoch': epoch + 1,
            }

            # save last ep ckpt
            if (epoch+1) == args.epochs:
                torch.save(save_dict, os.path.join(args.experiment_path, 'best-checkpoint.pth'))
    
            # test
            if ((epoch+1) % args.val_iter == 0) or (epoch == 0):
                with torch.no_grad():
                    print_log("==================== start testing ====================", logger=logger)
                    model.eval()
                    # internal validation
                    hospital = 'internal'
                    test_model(
                            model=model,
                            cox_model=cox_model,
                            hospital=hospital,
                            logger=logger,
                            writer=writer,
                            args=args,
                            epoch=epoch
                            )

                    # external validation
                    for external_hospital in args.external_hospitals:
                        test_model(
                            model=model,
                            cox_model=cox_model,
                            hospital=external_hospital,
                            logger=logger,
                            writer=writer,
                            args=args,
                            epoch=epoch
                            )

    # load pre-trained img encoder
    # similar way as done in downstream_main_img.py
    if args.split_pretrained_path is not None:
        model_dict, cox_model_dict = {}, {}
        for modality, pretrained_path in args.split_pretrained_path.items():
            checkpoint = torch.load(os.path.join(args.downstream_path, pretrained_path, 'best-checkpoint.pth'))
            model = CoxPHNet(backbone, input_dim=args.in_features).to(device)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            cox_model = checkpoint['cox_model']

            model_dict[modality], cox_model_dict[modality] = model, cox_model
    else:
        ckpt_path = args.ckpt_path if args.test_only else args.experiment_path
        checkpoint = torch.load(os.path.join(ckpt_path, 'best-checkpoint.pth'))
        model = CoxPHNet(backbone, input_dim=args.in_features).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        cox_model = checkpoint['cox_model']

    # test
    with torch.no_grad():
        print_log("==================== start testing ====================", logger=logger)
        model.eval()
        # internal validation
        hospital = 'internal'
        test_model(
                model=model if args.split_pretrained_path is None else model_dict,
                cox_model=cox_model if args.split_pretrained_path is None else cox_model_dict,
                hospital=hospital,
                logger=logger,
                writer=writer,
                args=args,
                epoch=args.epochs,
                save_flag=True,
                )

        # external validation
        for external_hospital in args.external_hospitals:
            test_model(
                model=model if args.split_pretrained_path is None else model_dict,
                cox_model=cox_model if args.split_pretrained_path is None else cox_model_dict,
                hospital=external_hospital,
                logger=logger,
                writer=writer,
                args=args,
                epoch=args.epochs,
                save_flag=True,
                )

                        
def train_model(model, cox_model, data_loader, optimizer, criterion, logger, writer, args, epoch, _data_loader):

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    model.train()

    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header, logger)):
        imgs = batch['imgs'].cuda()

        event = batch['labels']['rfs']["event"].cuda().squeeze()
        duration = batch['labels']['rfs']["duration"].cuda().squeeze()
        
        if torch.all(event == 0):
            continue

        output = model(imgs)
        loss = criterion(output, duration, event)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_steps = epoch * len(data_loader) + it

        metrics = {
            "loss": loss.item(),
        }

        for metric, value in metrics.items():
            writer.add_scalar(f"train/{metric}", value, global_steps)

        log_loss_to_file(metrics, logger=logger)
    
    if _data_loader is not None:
        collected_ouputs = []
        collected_durations = []
        collected_events = []
        with torch.no_grad():
            model.eval()
            for it, batch in enumerate(metric_logger.log_every(_data_loader, 10, header, logger)):
                imgs = batch['imgs'].cuda()

                event = batch['labels']['rfs']["event"].cuda().squeeze()
                duration = batch['labels']['rfs']["duration"].cuda().squeeze()

                output = model(imgs)

                collected_ouputs.append(output)
                collected_durations.append(duration)
                collected_events.append(event)
        
        # Calculate baseline hazards and the C-index after the entire epoch
        all_outputs = torch.cat(collected_ouputs)
        all_durations = torch.cat(collected_durations)
        all_events = torch.cat(collected_events)

        # use the entire dataset as input to calculate baseline hazards
        cox_model.compute_baseline_hazards(all_outputs, (all_durations, all_events))

        # calculate survival prediction
        survival_prediction = cox_model.predict_surv_df(all_outputs.detach().cpu().numpy())
        
        eval_surv = EvalSurv(
            survival_prediction,
            all_durations.cpu().numpy(),
            all_events.cpu().numpy(),
            censor_surv='km'
        )

        # calculate and save c-index
        epoch_cindex = eval_surv.concordance_td('antolini')
        writer.add_scalar(f"train/c-index", epoch_cindex, epoch)
        logger.info(f"Epoch: {epoch}, c-index: {epoch_cindex}")


def test_model(model, cox_model, hospital, logger, writer, args, epoch, save_flag=False):

    if args.split_pretrained_path is not None:
        for v in model.values():
            v.eval()
        if isinstance(cox_model, dict):
            cox_model = next(iter(cox_model.values()))
    else:
        model.eval()

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
    
    with torch.no_grad():
        test_metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'
        collected_outputs = []
        collected_durations = {'os': [], 'rfs': [], 'dss': []}
        collected_events = {'os': [], 'rfs': [], 'dss': []}
        collected_pids = []
        
        for batch in test_metric_logger.log_every(test_dataloader, 10, header, logger):
            pid = batch['patients']
            imgs = batch['imgs'].cuda()

            if isinstance(pid, torch.Tensor):
                collected_pids.extend(pid.cpu().numpy())
            else:
                collected_pids.extend(pid)
            
            for outcome in ['os', 'rfs', 'dss']:
                event = batch['labels'][outcome]["event"].cuda().squeeze()
                duration = batch['labels'][outcome]["duration"].cuda().squeeze()
                
                collected_durations[outcome].append(duration)
                collected_events[outcome].append(event)    

            if args.split_pretrained_path is not None:
            # late fusion, similar way done in downstream_main_img.py
                if imgs.dim() == 5:  
                    imgs = imgs.unsqueeze(1)
                modality_mask = ~torch.isnan(imgs).all(dim=(2, 3, 4, 5))
                B, T, C, D, H, W = imgs.shape
                outputs = torch.zeros(B, T).cuda()
                for t in range(T):
                    current_input = imgs[:, t]
                    current_mask = modality_mask[:, t]
                    valid_indexes = current_mask.nonzero(as_tuple=True)[0]
                    valid_input = current_input[current_mask]

                    current_model = model[args.modalities[t]]
                    current_output = current_model(valid_input)
                    outputs[valid_indexes, t:(t + 1)] = current_output
                outputs_sum = outputs.sum(dim=1, keepdim=True)
                outputs_non_zero_sum = modality_mask.sum(dim=1, keepdim=True)
                output = outputs_sum / outputs_non_zero_sum

            else:
                output = model(imgs)
            
            collected_outputs.append(output)
            
        all_outputs = torch.cat(collected_outputs)
        all_pids = np.array(collected_pids)
        metrics = {}

        # --- Dynamic Calculation of standard_prediction_times ---
        # Combine all durations from all outcomes to find the overall maximum follow-up time
        all_cohort_durations = []
        for outcome_durations in collected_durations.values():
            all_cohort_durations.extend([d.item() for sublist in outcome_durations for d in sublist])
        
        if len(all_cohort_durations) > 0:
            # Determine the maximum observed duration in the current cohort
            max_observed_duration = np.max(all_cohort_durations)
            # Define the step for standard time points (e.g., every 12 months)
            time_step = 12 
            # Calculate the upper bound for prediction times, rounding up to the nearest multiple of time_step
            # Add a small epsilon to max_observed_duration to ensure it's included if it's exactly a multiple
            upper_bound_for_times = int(np.ceil((max_observed_duration + 0.1) / time_step)) * time_step
            
            # Ensure a minimum upper bound, e.g., at least 60 months if max_observed_duration is very small
            if upper_bound_for_times < 60: # Example: set a minimum range of 5 years (60 months)
                upper_bound_for_times = 60 

            # Create the standard_prediction_times array
            # Start from 'time_step' (e.g., 12 months), up to upper_bound_for_times + 1 (exclusive end)
            standard_prediction_times = np.arange(time_step, upper_bound_for_times + 1, time_step).astype(float)
            
            print_log(f"Dynamically set standard_prediction_times for {hospital}: {standard_prediction_times.tolist()}", logger=logger)
        else:
            # Fallback if no durations are collected (e.g., empty test set)
            print_log(f"Warning: No durations collected for {hospital}. Using default standard_prediction_times.", logger=logger)
            standard_prediction_times = np.arange(12, 181, 12).astype(float) # Fallback to default


        # --- Prepare DataFrame to hold combined results by PID ---
        combined_df = pd.DataFrame({'pid': all_pids})
        combined_df.set_index('pid', inplace=True)


        # Iterate over each outcome to calculate metrics and collect data
        for outcome in ['os', 'rfs', 'dss']:
            print_log(f"Processing outcome: {outcome.upper()}", logger=logger)
            
            all_durations = torch.cat(collected_durations[outcome])
            all_events = torch.cat(collected_events[outcome])
            
            mask = all_events != -1
            
            filtered_durations = all_durations[mask]
            filtered_events = all_events[mask]
            filtered_outputs = all_outputs[mask]
            filtered_pids_for_outcome = all_pids[mask.cpu().numpy()]

            if len(filtered_pids_for_outcome) == 0:
                print_log(f"Warning: No valid data for outcome {outcome} in {hospital}. Skipping C-index and survival probability calculation.", logger=logger)
                continue


            # --- Calculate Single Risk Score (Linear Predictor) ---
            filtered_risk_scores = cox_model.predict(filtered_outputs.detach().cpu().numpy())

            # --- Get Time-Dependent Survival Probabilities (0-1 range) ---
            survival_prediction_df_raw = cox_model.predict_surv_df(
                filtered_outputs.detach().cpu().numpy()
            )
            
            survival_prediction_df_reindexed = survival_prediction_df_raw.reindex(
                standard_prediction_times, method='ffill'
            )
            survival_prediction_df_reindexed.fillna(1.0, inplace=True) 

            # --- Evaluate C-index ---
            eval_surv = EvalSurv(
                survival_prediction_df_raw, 
                filtered_durations.cpu().numpy(),
                filtered_events.cpu().numpy(),
                censor_surv='km'
            )
            epoch_cindex = eval_surv.concordance_td('antolini')
            metrics[f'c-index-{outcome}'] = epoch_cindex

            # --- Store this outcome's event, duration, risk_score into the combined_df ---
            outcome_core_data = pd.DataFrame({
                # Ensure these arrays are 1-dimensional using .ravel() or .flatten()
                f'{outcome}_event': filtered_events.cpu().numpy().ravel(),
                f'{outcome}_duration': filtered_durations.cpu().numpy().ravel(),
                f'{outcome}_risk_score': filtered_risk_scores.ravel() 
            }, index=filtered_pids_for_outcome)

            combined_df = combined_df.merge(outcome_core_data, left_index=True, right_index=True, how='left')

            # --- Add survival probabilities to the combined_df ---
            prob_data_for_merge = pd.DataFrame(index=filtered_pids_for_outcome)
            
            for patient_col_idx, pid in enumerate(filtered_pids_for_outcome):
                patient_probs_series = survival_prediction_df_reindexed.iloc[:, patient_col_idx]
                
                for time_point in standard_prediction_times:
                    col_name = f'{outcome}_surv_prob_at_{int(time_point)}m'
                    prob_val = patient_probs_series.get(time_point, np.nan) 
                    prob_data_for_merge.loc[pid, col_name] = prob_val

            combined_df = combined_df.merge(prob_data_for_merge, left_index=True, right_index=True, how='left')


        # --- Finalize combined_df structure ---
        combined_df.reset_index(inplace=True)

        column_order = ['pid']
        for outcome in ['os', 'rfs', 'dss']:
            column_order.extend([f'{outcome}_event', f'{outcome}_duration', f'{outcome}_risk_score'])
            for time_point in standard_prediction_times: # Use the dynamically generated times
                column_order.append(f'{outcome}_surv_prob_at_{int(time_point)}m')

        combined_df = combined_df.reindex(columns=column_order)

    # --- CSV Writing ---
    if save_flag:
        os.makedirs(args.logits_path, exist_ok=True) 
        
        results_csv_path = os.path.join(args.logits_path, f'{hospital}_test_results.csv')
        combined_df.to_csv(results_csv_path, index=False)
        print_log(f"Results saved to {results_csv_path}", logger=logger)

    for metric, value in metrics.items():
        writer.add_scalar(f"test_model/average/{hospital}/{metric}", value, epoch)

    return metrics

        
if __name__ == '__main__':
    args = get_downstream_args_img()
    main(args)
