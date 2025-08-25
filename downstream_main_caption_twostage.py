import os
import argparse
import math
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np

# import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM

from datasets.data_loader_RenalCLIP_caption import ImageCaptionDataset
from models.CaptionModel import ImageCaptionModel
from models.RenalModel import RenalModel

import torchvision.models as models
import torch.nn as nn

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

from utils.util import *
from utils.logger import *
from utils.parser import get_report_generation_args
from datasets.data_loader_RenalCLIP_caption import ImageCaptionDataset
import csv
import gc


TEXT_PRETRAINED_DIR = fr"/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/pretrained_models/language_family"
TEST_HOSPITALS = ['internal', '厦门', '山东', '瑞金', '张掖']
    
def compute_bleu(references, hypotheses, n=1):

    tokenized_refs = [[ref.split()] for ref in references]
    tokenized_hyps = [hyp.split() for hyp in hypotheses]
    
    if n == 1:
        weights = (1, 0, 0, 0)
    elif n == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n == 3:
        weights = (0.33, 0.33, 0.33, 0)
    elif n == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        raise ValueError("n must be 1, 2, 3, or 4")
    
    score = corpus_bleu(tokenized_refs, tokenized_hyps, weights=weights)
    return score

def compute_meteor(references, hypotheses):

    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = word_tokenize(ref)
        hyp_tokens = word_tokenize(hyp)
        score = meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)
    return sum(scores) / len(scores)

def compute_rouge(references, hypotheses):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores)
    }

def compute_bert_score(references, hypotheses):

    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False, model_type=os.path.join(TEXT_PRETRAINED_DIR, "FacebookAI/roberta-large"), num_layers=17)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def compute_cider(references, hypotheses):

    refs = {i: [r] for i, r in enumerate(references)}
    hyps = {i: [h] for i, h in enumerate(hypotheses)}
    
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(refs, hyps)
    return score

def evaluate_generated_captions(references, hypotheses):
    bleu1 = compute_bleu(references, hypotheses, n=1)
    bleu2 = compute_bleu(references, hypotheses, n=2)
    bleu4 = compute_bleu(references, hypotheses, n=4)
    
    meteor = compute_meteor(references, hypotheses)
    
    rouge_scores = compute_rouge(references, hypotheses)
    rouge1 = rouge_scores['rouge1']
    rouge2 = rouge_scores['rouge2']
    rougeL = rouge_scores['rougeL']
    
    bert_scores = compute_bert_score(references, hypotheses)
    bert_precision = bert_scores['precision']
    bert_recall = bert_scores['recall']
    bert_f1 = bert_scores['f1']
    
    cider = compute_cider(references, hypotheses)

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "METEOR": meteor,
        "CIDEr": cider,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "BERTScore_precision": bert_precision,  
        "BERTScore_recall": bert_recall,
        "BERTScore_f1": bert_f1
    }

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


def train_stage_A(args, logger, writer, device, image_encoder, language_model, tokenizer, eos_token_id, prefix_tokens, suffix_tokens):
    """
    Stage 1 of the LLaVA-like training: visual feature alignment.
    Only the visual connection layer (projection layer) is trained.
    """
    print_log("\n" + "="*20 + " Starting Stage A Training " + "="*20 + "\n", logger=logger)

    # dataloader for stage A
    train_dataset_A = ImageCaptionDataset(args, hospital='internal', split='train', stage=1) # 7k data
    val_dataset_A = ImageCaptionDataset(args, hospital='internal', split='test', stage=1) # Validation for Stage A
    print_log(f"==================== Stage A train dataset load complete ====================: {len(train_dataset_A)} image-caption pairs", logger=logger)
    print_log(f"==================== Stage A val dataset load complete ====================: {len(val_dataset_A)} image-caption pairs", logger=logger)

    train_loader_A = DataLoader(
        train_dataset_A,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader_A = DataLoader(
        val_dataset_A,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build the ImageCaptionModel
    # The projection_hidden_dim must match the hidden_size of the LLM
    projection_hidden_dim = language_model.config.hidden_size

    model = ImageCaptionModel(
        image_encoder=image_encoder,
        language_model=language_model,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        projection_hidden_dim=projection_hidden_dim,
        num_image_tokens=args.num_image_tokens,
        image_feature_dim=args.image_feature_dim,
    )
    model.to(device)

    # Freeze the Image Encoder and LLM, and only train the projection layer.
    model.image_encoder.eval()
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    model.language_model.eval()
    for param in model.language_model.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.projection.parameters(), lr=args.stage_A_adapter_lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader_A) * args.stage_A_epochs
    warmup_ratio = 0.1
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float('inf')
    best_epoch_A = 0
    
    checkpoint_dir_A = os.path.join(args.experiment_path, "stage_A_checkpoints")
    os.makedirs(checkpoint_dir_A, exist_ok=True)

    for epoch_A in range(args.stage_A_epochs):
        model.train()
        train_loss_A = 0.0
        metric_logger_A = MetricLogger(delimiter="  ")
        header_A = 'Stage A Training...Epoch: [{}/{}]'.format(epoch_A + 1, args.stage_A_epochs)

        for it, (images, captions, patient) in enumerate(metric_logger_A.log_every(train_loader_A, args.log_freq, header_A, logger)):
            global_steps_A = epoch_A * len(train_loader_A) + it
            images = images.to(device=device)

            tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len)
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

            outputs = model(images, input_ids, attention_mask)
            loss_A = outputs.loss
            loss_A.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            train_loss_A += loss_A.item()
            writer.add_scalar(f"StageA/train_loss", loss_A.item(), global_steps_A)
            metric_logger_A.update(loss_train_A=loss_A.item())
            metric_logger_A.update(adapter_lr_A=optimizer.param_groups[0]["lr"])

        avg_train_loss_A = train_loss_A / len(train_loader_A)
        print_log(f"=====Stage A Epoch {epoch_A+1} Train Loss: {avg_train_loss_A}=====", logger=logger)

        # Stage A Validation
        model.eval()
        val_loss_A = 0.0
        with torch.no_grad():
            metric_logger_val_A = MetricLogger(delimiter="  ")
            header_val_A = 'Stage A Validating...Epoch: [{}/{}]'.format(epoch_A + 1, args.stage_A_epochs)
            for it, (images, captions, patient) in enumerate(metric_logger_val_A.log_every(val_loader_A, args.log_freq, header_val_A, logger)):
                images = images.to(device=device)
                tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len)
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)
                outputs = model(images, input_ids, attention_mask)
                val_loss_A += outputs.loss.item()

        avg_val_loss_A = val_loss_A / len(val_loader_A)
        perplexity_A = math.exp(avg_val_loss_A) if avg_val_loss_A < 700 else float('inf')
        print_log(f"=====Stage A Epoch {epoch_A+1} Val Loss: {avg_val_loss_A}, Perplexity: {perplexity_A}=====", logger=logger)
        writer.add_scalar(f"StageA/val_loss", avg_val_loss_A, epoch_A)
        writer.add_scalar(f"StageA/perplexity", perplexity_A, epoch_A)

        # Save the best model based on the validation loss
        if avg_val_loss_A < best_val_loss:
            best_val_loss = avg_val_loss_A
            best_epoch_A = epoch_A + 1
            best_adapter_path = os.path.join(checkpoint_dir_A, "adapter_stageA_best.bin")
            torch.save(model.projection.state_dict(), best_adapter_path)
            print_log(f"===== Stage A Best Adapter saved at epoch {best_epoch_A} with Val Loss: {best_val_loss:.4f} =====", logger=logger)

    print_log(f"\nStage A training complete. Best Stage A Adapter saved from epoch {best_epoch_A}.", logger=logger)
    print_log("="*20 + " Stage A Training Finished " + "="*20 + "\n", logger=logger)

    # Return the model, where the llm and image_encoder are the original frozen ones, and the adapter is trained.
    # Load the best adapter weights before returning to ensure they are used for Stage B.
    if os.path.exists(best_adapter_path):
        model.projection.load_state_dict(torch.load(best_adapter_path, map_location=device))
        print_log(f"Loaded best Stage A Adapter for Stage B training.", logger=logger)
    else:
        print_log(f"Warning: Best Stage A adapter not found at {best_adapter_path}. Using last epoch's adapter for Stage B.", logger=logger) # type: ignore

    return model


def train_stage_B(args, logger, writer, device, model, tokenizer, eos_token_id):
    """
    Stage 2 of LLaVA-like training: downstream task fine-tuning.
    Finetune both the visual connection layer (projection layer) and LLM with LoRA.
    """
    print_log("\n" + "="*20 + " Starting Stage B Training " + "="*20 + "\n", logger=logger)

    # dataloader for stage B
    train_dataset_B = ImageCaptionDataset(args, hospital='internal', split='train', stage=2) # 400 data
    test_dataset_B = ImageCaptionDataset(args, hospital='internal', split='test', stage=2) # for validation
    print_log(f"==================== Stage B train dataset load complete ====================: {len(train_dataset_B)} image-caption pairs", logger=logger)
    print_log(f"==================== Stage B test dataset load complete ====================: {len(test_dataset_B)} image-caption pairs", logger=logger)

    train_loader_B = DataLoader(
        train_dataset_B,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader_B = DataLoader(
        test_dataset_B,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Image encoder still remains frozen
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    
    # projection layer remains trainable
    for param in model.projection.parameters():
        param.requires_grad = True

    # finetune LLM with LoRA
    if isinstance(model.language_model, torch.nn.Module) and hasattr(model.language_model, 'base_model'):
         for n, p in model.language_model.named_parameters():
             if 'lora' in n:
                 p.requires_grad = True
             else:
                 p.requires_grad = False
    else:
        print_log("Warning: model.language_model is not a PEFT model. LoRA parameters might not be correctly activated.", logger=logger)

    print_log(f"LLM trainable parameters (Stage B) summary:", logger=logger)
    model.language_model.print_trainable_parameters()

    print_log(f"Projection layer trainable parameters (Stage B): {sum(p.numel() for p in model.projection.parameters() if p.requires_grad)}", logger=logger)

    optimizer = optim.AdamW([
        {'params': model.projection.parameters(), 'lr': args.stage_B_adapter_lr},
        {'params': model.language_model.parameters(), 'lr': args.lora_lr}
    ], weight_decay=args.weight_decay)

    total_steps = len(train_loader_B) * args.stage_B_epochs
    warmup_ratio = 0.1
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    checkpoint_dir_B = os.path.join(args.experiment_path, "stage_B_checkpoints")
    os.makedirs(checkpoint_dir_B, exist_ok=True)

    for epoch_B in range(args.stage_B_epochs):
        # --- Stage B Epoch-wise Evaluation ---
        print_log(f"\n" + "="*10 + f" Stage B Evaluation for Epoch {epoch_B+1} " + "="*10 + "\n", logger=logger)
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        references = []
        hypotheses = []
        patient_ids = []

        with torch.no_grad():
            metric_logger_eval = MetricLogger(delimiter="  ")
            header_eval = f'Stage B Evaluating Epoch [{epoch_B+1}/{args.stage_B_epochs}]...'
            for it, (images, captions, patient) in enumerate(metric_logger_eval.log_every(test_loader_B, args.log_freq, header_eval, logger)):
                images = images.to(device=device)

                tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len)
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)

                outputs = model(images, input_ids, attention_mask)
                loss = outputs.loss
                
                # Model generation for evaluation metrics
                generated_ids = model.generate(
                    images,
                    max_new_tokens=args.max_seq_len,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=eos_token_id,
                    early_stopping=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

                val_loss += loss.item()
                num_val_batches += 1

                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                references.extend(captions)
                hypotheses.extend(generated_texts)
                if isinstance(patient, str):
                    patient_ids.append(patient)
                else:
                    patient_ids.extend(patient)

            avg_val_loss = val_loss / num_val_batches
            perplexity = math.exp(avg_val_loss) if avg_val_loss < 700 else float('inf')

            print_log(f"=====Stage B Epoch {epoch_B+1} Val Loss: {avg_val_loss}, Perplexity: {perplexity}=====", logger=logger)

            # metrics
            bleu1 = compute_bleu(references, hypotheses, n=1)
            bleu2 = compute_bleu(references, hypotheses, n=2)
            bleu4 = compute_bleu(references, hypotheses, n=4)
            meteor = compute_meteor(references, hypotheses)
            rouge_scores = compute_rouge(references, hypotheses)
            bert_scores = compute_bert_score(references, hypotheses)
            cider = compute_cider(references, hypotheses)

            metrics_dict = {
                "val_loss": avg_val_loss,
                "perplexity": perplexity,
                "BLEU-1": bleu1,
                "BLEU-2": bleu2,
                "BLEU-4": bleu4,
                "METEOR": meteor,
                "CIDEr": cider,
                "ROUGE-1": rouge_scores['rouge1'],
                "ROUGE-2": rouge_scores['rouge2'],
                "ROUGE-L": rouge_scores['rougeL'],
                "BERTScore_precision": bert_scores['precision'],
                "BERTScore_recall": bert_scores['recall'],
                "BERTScore_f1": bert_scores['f1']
            }

            log_loss_to_file(metrics_dict, logger=logger)
            for metric, value in metrics_dict.items():
                writer.add_scalar(f"StageB/val_{metric}", value, epoch_B)
        model.train()
        train_loss_B = 0.0
        metric_logger_B = MetricLogger(delimiter="  ")
        header_B = 'Stage B Training...Epoch: [{}/{}]'.format(epoch_B + 1, args.stage_B_epochs)

        for it, (images, captions, patient) in enumerate(metric_logger_B.log_every(train_loader_B, args.log_freq, header_B, logger)):
            global_steps_B = epoch_B * len(train_loader_B) + it
            images = images.to(device=device)

            tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len)
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

            outputs = model(images, input_ids, attention_mask)
            loss_B = outputs.loss
            loss_B.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            train_loss_B += loss_B.item()
            writer.add_scalar(f"StageB/train_loss", loss_B.item(), global_steps_B)
            metric_logger_B.update(loss_train_B=loss_B.item())
            metric_logger_B.update(adapter_lr_B=optimizer.param_groups[0]["lr"])
            metric_logger_B.update(lora_lr_B=optimizer.param_groups[1]["lr"])

        avg_train_loss_B = train_loss_B / len(train_loader_B)
        print_log(f"=====Stage B Epoch {epoch_B+1} Train Loss: {avg_train_loss_B}=====", logger=logger)
        writer.add_scalar(f"StageB/avg_train_loss", avg_train_loss_B, epoch_B)


    print_log(f"\nStage B training complete.", logger=logger)
    print_log("="*20 + " Stage B Training Finished " + "="*20 + "\n", logger=logger)

    # --- Final Evaluation---
    print_log("\n" + "="*20 + " Starting Final Evaluation (Last Epoch Model) " + "="*20 + "\n", logger=logger)
    
    model.eval()
    references = []
    hypotheses = []
    patient_ids = []
    all_case_metrics = [] # List to store metrics for each individual case

    with torch.no_grad():
        metric_logger_final_eval = MetricLogger(delimiter="  ")
        header_final_eval = 'Final Evaluation (Last Epoch Model)...'
        for it, (images, captions, patient) in enumerate(metric_logger_final_eval.log_every(test_loader_B, args.log_freq, header_final_eval, logger)):
            images = images.to(device=device)

            tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len)
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

            # Note: We don't need outputs.loss.item() for this final pass if only concerned with generation metrics
            # But kept for consistency if you want to log overall loss as well.
            outputs = model(images, input_ids, attention_mask) 

            generated_ids = model.generate(
                images,
                max_new_tokens=args.max_seq_len,
                num_beams=4,
                do_sample=False,
                eos_token_id=eos_token_id,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Collect data for individual case metrics
            for i in range(len(captions)):
                ref_i = captions[i]
                hyp_i = generated_texts[i]
                patient_id_i = patient[i] if isinstance(patient, list) else patient # Handle single string or list

                # Store for overall metrics
                references.append(ref_i)
                hypotheses.append(hyp_i)
                patient_ids.append(patient_id_i)

                # Calculate individual case metrics
                # Ensure compute_* functions can handle single string/list inputs
                case_bleu1 = compute_bleu([ref_i], [hyp_i], n=1)
                case_bleu2 = compute_bleu([ref_i], [hyp_i], n=2)
                case_bleu4 = compute_bleu([ref_i], [hyp_i], n=4)
                case_meteor = compute_meteor([ref_i], [hyp_i])
                case_cider = compute_cider([ref_i], [hyp_i])
                case_rouge = compute_rouge([ref_i], [hyp_i])
                case_bert_score = compute_bert_score([ref_i], [hyp_i])

                case_metrics = {
                    "BLEU-1": case_bleu1,
                    "BLEU-2": case_bleu2,
                    "BLEU-4": case_bleu4,
                    "METEOR": case_meteor,
                    "CIDEr": case_cider,
                    "ROUGE-1": case_rouge['rouge1'],
                    "ROUGE-2": case_rouge['rouge2'],
                    "ROUGE-L": case_rouge['rougeL'],
                    "BERTScore_precision": case_bert_score['precision'],
                    "BERTScore_recall": case_bert_score['recall'],
                    "BERTScore_f1": case_bert_score['f1']
                }
                all_case_metrics.append(case_metrics)

        # --- Calculate and Log Overall Metrics (Optional, as these were done per-epoch) ---
        # You can choose to skip this part if you only want per-case metrics in CSV
        # and already log overall metrics per-epoch.
        avg_val_loss = val_loss / len(test_loader_B) # Recalculate avg loss if needed
        perplexity = math.exp(avg_val_loss) if avg_val_loss < 700 else float('inf')
        
        # Calculate overall metrics for the entire test set (from collected references/hypotheses)
        overall_bleu1 = compute_bleu(references, hypotheses, n=1)
        overall_bleu2 = compute_bleu(references, hypotheses, n=2)
        overall_bleu4 = compute_bleu(references, hypotheses, n=4)
        overall_meteor = compute_meteor(references, hypotheses)
        overall_rouge_scores = compute_rouge(references, hypotheses)
        overall_bert_scores = compute_bert_score(references, hypotheses)
        overall_cider = compute_cider(references, hypotheses)

        metrics_dict_final = {
            "val_loss": avg_val_loss,
            "perplexity": perplexity,
            "BLEU-1": overall_bleu1,
            "BLEU-2": overall_bleu2,
            "BLEU-4": overall_bleu4,
            "METEOR": overall_meteor,
            "CIDEr": overall_cider,
            "ROUGE-1": overall_rouge_scores['rouge1'],
            "ROUGE-2": overall_rouge_scores['rouge2'],
            "ROUGE-L": overall_rouge_scores['rougeL'],
            "BERTScore_precision": overall_bert_scores['precision'],
            "BERTScore_recall": overall_bert_scores['recall'],
            "BERTScore_f1": overall_bert_scores['f1']
        }
        print_log(f"=====Final Last Epoch Model Overall Evaluation Metrics=====", logger=logger)
        log_loss_to_file(metrics_dict_final, logger=logger)
        for metric, value in metrics_dict_final.items():
            # writer.add_scalar(f"final_last_epoch_test/{metric}", value, args.stage_B_epochs) # Final TensorBoard tag
            writer.add_scalar(f"StageB/val_{metric}", value, args.stage_B_epochs) # Final TensorBoard tag

        # --- Save individual case results to CSV ---
        results_csv_path = os.path.join(args.experiment_path, 'report_generation_test_results_last_epoch_model_per_case.csv') # Unique filename

        # Prepare CSV header and data
        if all_case_metrics:
            # Get metric names from the first case's metrics (assuming consistent keys)
            metric_column_names = list(all_case_metrics[0].keys())
        else:
            metric_column_names = [] # Fallback if no cases (shouldn't happen with test_loader_B)

        csv_header = ['PatientID', 'Reference', 'Hypothesis'] + metric_column_names
        csv_data = []

        for i in range(len(patient_ids)):
            row = [patient_ids[i], references[i], hypotheses[i]]
            if all_case_metrics: # Ensure there are metrics to append
                for col_name in metric_column_names:
                    # Use .get() for safety, though keys should be consistent
                    row.append(all_case_metrics[i].get(col_name, ''))
            csv_data.append(row)
        
        # Save the results to the CSV file
        with open(results_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(csv_header)
            writer_csv.writerows(csv_data)

        print_log(f"Final last epoch model results (per-case metrics) saved to {results_csv_path}", logger=logger)
        print_log("="*20 + " Final Evaluation Finished " + "="*20 + "\n", logger=logger)

        # save checkpoint
        if args.save_model:
            checkpoint_dir = args.experiment_path
            # Save the Image Encoder weights (although they are frozen, save a copy for completeness or as a backup)
            torch.save(model.image_encoder.state_dict(), os.path.join(checkpoint_dir, "image_encoder.bin"))
            print_log("===== Saved Image Encoder =====", logger=logger)
            # Save the Adapter weights (after Stage B training)
            torch.save(model.projection.state_dict(), os.path.join(checkpoint_dir, "adapter_stageB_final.bin"))
            print_log("===== Saved Stage B Final Adapter =====", logger=logger)
            # save llm + lora
            model.language_model.save_pretrained(os.path.join(checkpoint_dir, "llm_lora"))
            print_log("===== Saved LLM + LoRA =====", logger=logger)


def test_stage_B(args, logger, writer, device, model, tokenizer, eos_token_id, hospital):
    """
    Standalone testing function with streaming CSV write for per-case metrics.
    """
    # dataloader for stage B
    test_dataset_B = ImageCaptionDataset(args, hospital=hospital, split='test', stage=2)
    print_log(f"==================== Stage B test dataset for '{hospital}' load complete: {len(test_dataset_B)} image-caption pairs", logger=logger)

    test_loader_B = DataLoader(
        test_dataset_B,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print_log(f"\n" + "="*20 + f" Starting Final Evaluation for {hospital} " + "="*20 + "\n", logger=logger)
    
    model.eval()

    # --- Streaming CSV Write Logic ---
    results_csv_path = os.path.join(args.experiment_path, f'{hospital}_report_generation_results.csv')
    csv_header = ["PatientID", "Ground Truth", "Prediction", "bleu1", "bleu2", "bleu4", "meteor", "cider", "rouge1", "rouge2", "rougeL", "bert_precision", "bert_recall", "bert_f1"]

    with open(results_csv_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(csv_header)

        with torch.no_grad():
            metric_logger_final_eval = MetricLogger(delimiter="  ")
            header_final_eval = f'Testing on {hospital}...'
            
            for it, (images, captions, patients) in enumerate(metric_logger_final_eval.log_every(test_loader_B, args.log_freq, header_final_eval, logger)):
                images = images.to(device=device)

                generated_ids = model.generate(
                    images,
                    max_new_tokens=args.max_seq_len,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=eos_token_id,
                    early_stopping=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Process and write results for each sample in the batch
                for i in range(len(captions)):
                    ref_i = captions[i]
                    hyp_i = generated_texts[i]
                    patient_id_i = patients[i] if isinstance(patients, list) else patients

                    # Calculate metrics for the single current case
                    metrics = evaluate_generated_captions([ref_i], [hyp_i])
                    
                    # Prepare row and write to CSV immediately
                    row = [
                        patient_id_i, ref_i, hyp_i,
                        metrics["BLEU-1"], metrics["BLEU-2"], metrics["BLEU-4"],
                        metrics["METEOR"], metrics["CIDEr"],
                        metrics["ROUGE-1"], metrics["ROUGE-2"], metrics["ROUGE-L"],
                        metrics["BERTScore_precision"], metrics["BERTScore_recall"], metrics["BERTScore_f1"]
                    ]
                    csv_writer.writerow(row)

    print_log(f"Final evaluation results (per-case metrics) saved to {results_csv_path}", logger=logger)
    print_log("="*20 + " Final Evaluation Finished " + "="*20 + "\n", logger=logger)

def main(args):
    create_experiment_dir(args)
    fix_random_seeds(args.seed)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    print_log("==================== Parameters ====================", logger=logger)
    log_args_to_file(args, 'args', logger=logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = create_tfboard_on_master(os.path.join(args.tfboard_path))

    # Initialize the Image Encoder
    # it should remain frozen throughout the entire training process.
    image_encoder = RenalModel( mode=args.finetune_type,
                                pretrained_exp_name=args.pretrained_exp_name,
                                pretrained_metric=args.pretrained_metric,
                                model_type=args.model_type,
                                logger=logger,
                            )
    image_encoder.eval()
    for param in image_encoder.parameters():
        param.requires_grad = False
    image_encoder.to(device=device)

    # load LLM and tokenizer (BioMistral-7B)
    language_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(TEXT_PRETRAINED_DIR, args.language_model_name),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(TEXT_PRETRAINED_DIR, args.language_model_name),
        local_files_only=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id

    # Prompt Text
    prefix_text = "<s>[INST] "
    suffix_text = """
        Generate a radiology report for the given CT scan of suspected renal cancer. Use the following format:
        Findings: [Detailed observations based on the CT scan.]
        Impression: [Summary of the diagnostic conclusions.]
        [/INST]
        """
    
    prefix_tokens = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids
    suffix_tokens = tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt").input_ids

    stage_A_best_adapter_path = os.path.join(args.experiment_path, "stage_A_checkpoints", "adapter_stageA_best.bin")

    # ====================================================================================================
    # Check if Stage A has been trained and the best adapter has been saved
    if os.path.exists(stage_A_best_adapter_path) and not args.force_stage_A_retrain:
        print_log("\n" + "="*20 + " Skipping Stage A Training " + "="*20 + "\n", logger=logger)
        print_log(f"Found existing Stage A best adapter at: {stage_A_best_adapter_path}", logger=logger)

        # build ImageCaptionModel
        # Note: The language_model here is the original one,
        # because the LLM was frozen during Stage A training, and we only care its structure and projection_hidden_dim
        projection_hidden_dim = language_model.config.hidden_size
        model_trained_in_stage_A = ImageCaptionModel(
            image_encoder=image_encoder,
            language_model=language_model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            projection_hidden_dim=projection_hidden_dim,
            num_image_tokens=args.num_image_tokens,
            image_feature_dim=args.image_feature_dim,
        )
        model_trained_in_stage_A.to(device)

        # Ensure the Image Encoder and LLM remain frozen 
        # (although LoRA will be applied in Stage B, this is the loading logic for the Stage A model)
        model_trained_in_stage_A.image_encoder.eval()
        for param in model_trained_in_stage_A.image_encoder.parameters():
            param.requires_grad = False
        model_trained_in_stage_A.language_model.eval()
        for param in model_trained_in_stage_A.language_model.parameters():
            param.requires_grad = False

        # load trained projection layer from stage A
        model_trained_in_stage_A.projection.load_state_dict(torch.load(stage_A_best_adapter_path, map_location=device))
        print_log(f"Successfully loaded Stage A best adapter for Stage B preparation.", logger=logger)

    else:
        # --- Start stage A training ---
        model_trained_in_stage_A = train_stage_A(args, logger, writer, device, image_encoder, language_model, tokenizer, eos_token_id, prefix_tokens, suffix_tokens)
    # ====================================================================================================

    # --- Prepare for stage B, finetune LLM with LoRA ---
    # Apply LoRA to the LLM. Note that model_trained_in_stage_A.language_model is still the original LLM instance.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # peft
    peft_language_model = get_peft_model(model_trained_in_stage_A.language_model, lora_config)
    peft_language_model.to(device)
    print_log(f"\nPEFT Language Model for Stage B: {peft_language_model}", logger=logger)
    peft_language_model.print_trainable_parameters()

    # Reinitialize ImageCaptionModel, passing in the PEFT-wrapped LLM.
    # The projection layer should retain the weights trained in Stage A.
    projection_hidden_dim = peft_language_model.config.hidden_size
    model_for_stage_B = ImageCaptionModel(
        image_encoder=image_encoder,
        language_model=peft_language_model,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        projection_hidden_dim=projection_hidden_dim,
        num_image_tokens=args.num_image_tokens,
        image_feature_dim=args.image_feature_dim,
    )

    best_adapter_path_A = os.path.join(args.experiment_path, "stage_A_checkpoints", "adapter_stageA_best.bin")
    if os.path.exists(best_adapter_path_A):
        model_for_stage_B.projection.load_state_dict(torch.load(best_adapter_path_A, map_location=device))
        print_log(f"Loaded best Stage A Adapter weights into model for Stage B.", logger=logger)
    else:
        print_log(f"Warning: Best Stage A adapter not found at {best_adapter_path_A}. Starting Stage B with randomly initialized adapter (or default init).", logger=logger)

    model_for_stage_B.to(device)


    # --- Stage B ---
    if not args.test_only:
        # --- Start stage B training ---
        train_stage_B(args, logger, writer, device, model_for_stage_B, tokenizer, eos_token_id)
    else:
        # Load trained model from stage B
        print_log("Test-only mode activated. Attempting to load Stage B model for evaluation.", logger=logger)
        stage_B_checkpoint_dir = args.experiment_path
        # Load adapter after stage B
        final_adapter_path = os.path.join(stage_B_checkpoint_dir, "adapter_stageB_final.bin")
        if os.path.exists(final_adapter_path):
            model_for_stage_B.projection.load_state_dict(torch.load(final_adapter_path, map_location=device))
            print_log(f"Loaded Stage B final adapter from {final_adapter_path}", logger=logger)
        else:
            print_log(f"Warning: Stage B final adapter not found at {final_adapter_path}. Cannot perform test.", logger=logger)
            return

        # Load lora LLM + LoRA
        lora_model_path = os.path.join(stage_B_checkpoint_dir, "llm_lora")
        if os.path.exists(lora_model_path):
            peft_language_model = AutoPeftModelForCausalLM.from_pretrained(lora_model_path)
            peft_language_model.to(device)
            model_for_stage_B.language_model = peft_language_model
            print_log(f"Loaded Stage B LoRA model from {lora_model_path}", logger=logger)
        else:
            print_log(f"Warning: Stage B LoRA model not found at {lora_model_path}. Cannot perform test.", logger=logger)
            return
        for hospital in TEST_HOSPITALS:
            test_stage_B(args, logger, writer, device, model_for_stage_B, tokenizer, eos_token_id, hospital)

    writer.close()
    print_log("Training process finished.", logger=logger)
    

if __name__ == '__main__':

    args = get_report_generation_args()
    main(args)