import torch
import numpy as np
from transformers import AutoTokenizer
import os
import json
from utils.util import *

BERT_BASE_DIR = fr'/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/pretrained_models/language_family'
ZEROSHOT_BASE_DIR = '/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/RenalCLIP/zero_shot'
IMAGE_EMBEDDINGS_ROOT = fr"/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/RenalCLIP/image_embeddings"

def zeroshot_setup(args):
    TEXT_MODEL_NAME = os.path.join(BERT_BASE_DIR, args.bert_type)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, local_files_only=True, trust_remote_code=True)

    with open(os.path.join(ZEROSHOT_BASE_DIR, 'zeroshot_prompt.json'), 'r') as json_file:
        data = json.load(json_file)

    templates = data["templates"]
    attributes = data["attributes"]

    selected_attributes = {attr: attributes[attr] for attr in args.zeroshot_attributes if attr in attributes}

    return tokenizer, templates, selected_attributes



def load_text_embeddings_from_disk(text_embeddings_dir, avg_before_align=True):
    """
    Loads text embeddings for each task and corresponding label from the disk.

    Args:
        text_embeddings_dir (str): The root directory for text embeddings.
            The file structure should be, for example:
            text_embeddings_dir/
                BMC/
                    0.npy
                    1.npy
                RENAL_AIR/
                    0.npy
                    ...

    Returns:
        dict: A nested dictionary with the structure {task_name: {label: embedding}}.
    """
    embeddings_dict = {}
    prefix = "templates_with_avg" if avg_before_align else "templates_wo_avg"
    text_embeddings_dir = os.path.join(text_embeddings_dir, prefix)

    for task_name in os.listdir(text_embeddings_dir):
        task_dir = os.path.join(text_embeddings_dir, task_name)
        if not os.path.isdir(task_dir):
            continue

        print(f"Loading embeddings for task: {task_name}")
        embeddings_dict[task_name] = {}

        for label_file in os.listdir(task_dir):
            label = os.path.splitext(label_file)[0]
            label_embeddings_path = os.path.join(task_dir, label_file)

            template_embeddings = np.load(label_embeddings_path)  # shape: (1, embedding_dim) if avg_before_align else (# templates, embedding_dim)
            embeddings_dict[task_name][label] = template_embeddings

    return embeddings_dict


def load_image_embeddings_from_disk(image_embeddings_dir, image_encoder_name, patient_ids):
    if isinstance(patient_ids, list):
        embeddings = []
        for patient_id in patient_ids:
            image_embedding_path = os.path.join(image_embeddings_dir, image_encoder_name, patient_id, "image_embedding.npy")
            
            if os.path.exists(image_embedding_path):
                embedding_np = np.load(image_embedding_path)
                embedding_tensor = torch.from_numpy(embedding_np) 
                embeddings.append(embedding_tensor)
            else:
                print(f"Warning: Image embedding not found for patient_id: {patient_id} at {image_embedding_path}")

        if embeddings:
            embeddings = torch.stack(embeddings, dim=0)
        else:
            print("No embeddings were loaded.")
            embeddings = None
    elif isinstance(patient_ids, str):      
        image_embeddings_path = os.path.join(image_embeddings_dir, image_encoder_name, patient_ids, "image_embedding.npy")
        embeddings = np.load(image_embeddings_path)
        embeddings = torch.from_numpy(embeddings)
    else:
        raise TypeError(f"type of {patient_ids} is not valid")

    return embeddings


def zeroshot_one_hospital(
    dataloader, 
    task_info,
    hospital,
    writer,
    image_embed_name,
    text_embeddings_dict,
    avg_before_align=True,
    use_max_similarity=False,
    bootstrap_num_samples=1000,
    seed=42,
):
    """ 
    Perform zero-shot classification from a specific hospital, supporting three strategies.
    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader for the images.

        task_info (dict): A dictionary describing the task.

        hospital (str): The name of the current hospital.

        writer (SummaryWriter): Used for logging.

        image_embed_name (str): The name of the image encoder.

        text_embeddings_dict (dict): A dictionary of pre-loaded text embeddings.

        avg_before_align (bool): If True, use the averaging strategy; otherwise, do not average.

        use_max_similarity (bool): When avg_before_align is False, if True, use the maximum similarity strategy; otherwise, use the prompt-by-prompt evaluation strategy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_bootstrap_mode = not avg_before_align and not use_max_similarity

    _task_info = task_info.copy()
    print(f"\n--- Running Zero-shot test on '{hospital}' hospital ---")


    if is_bootstrap_mode:
        # Perform bootstrapping on the prompts within each category
        np.random.seed(seed)
        num_prompts_class_0 = len(text_embeddings_dict[next(iter(task_info))]['0'])
        num_prompts_class_1 = len(text_embeddings_dict[next(iter(task_info))]['1'])
        indices_0 = np.random.randint(0, num_prompts_class_0, size=bootstrap_num_samples)
        indices_1 = np.random.randint(0, num_prompts_class_1, size=bootstrap_num_samples)
        prompt_pairs_indices = list(zip(indices_0, indices_1))
        
        print(f"Running prompt sensitivity analysis with {bootstrap_num_samples} random pairs (seed={seed})...")

        # Outer loop: Iterate through each pre-defined prompt combination
        for i, idx_pair in enumerate(prompt_pairs_indices):
            idx_0, idx_1 = idx_pair
            
            logits_dict_single_pair = {task_name: [] for task_name in task_info.keys()}
            labels_dict_single_pair = {task_name: [] for task_name in task_info.keys()}
            
            # Inner loop: Iterate through the entire dataset.
            for batch in dataloader:
                patient_ids, labels = batch['patients'], batch['labels']
                with torch.no_grad():
                    image_embeddings = load_image_embeddings_from_disk(IMAGE_EMBEDDINGS_ROOT, image_embed_name, patient_ids).to(device)
                    for task_name in task_info.keys():
                        task_labels = labels[task_name].to(device)
                        task_text_embeds = text_embeddings_dict[task_name]
                        prompt_0_embed = torch.tensor(task_text_embeds['0'][idx_0], dtype=torch.float32).unsqueeze(0).to(device)
                        prompt_1_embed = torch.tensor(task_text_embeds['1'][idx_1], dtype=torch.float32).unsqueeze(0).to(device)
                        text_embeddings = torch.cat([prompt_0_embed, prompt_1_embed], dim=0)
                        logits = image_embeddings @ text_embeddings.T
                        for j in range(len(task_labels)):
                            if task_labels[j] != -1:
                                logits_dict_single_pair[task_name].append(logits[j].cpu().numpy())
                                labels_dict_single_pair[task_name].append(task_labels[j].cpu().numpy())
            
            zeroshot_res_dict = {}
            
            zeroshot_res_dict = get_metrics(logits_dict=logits_dict_single_pair, labels_dict=labels_dict_single_pair, task_info=_task_info)
            if is_main_process():
                prompt_key = f"bootstrap_sample_{i}"
                for task, metrics in zeroshot_res_dict.items():
                    for metric, value in metrics.items():
                        writer.add_scalar(f"zeroshot_test/{task}/{hospital}/{prompt_key}/{metric}", value, 0)
    else:
        logits_dict = {task_name: [] for task_name in _task_info.keys()}
        labels_dict = {task_name: [] for task_name in _task_info.keys()}
        
        for batch in dataloader:
            patient_ids = batch['patients']
            labels = batch['labels']

            with torch.no_grad():
                image_embeddings = load_image_embeddings_from_disk(
                    image_embeddings_dir=IMAGE_EMBEDDINGS_ROOT, 
                    image_encoder_name=image_embed_name, 
                    patient_ids=patient_ids
                ).to(device)

                for task_name in _task_info.keys():
                    task_labels = labels[task_name].to(device)
                    task_text_embeds = text_embeddings_dict[task_name]
                    categories = len(list(task_text_embeds.keys()))

                    if avg_before_align:
                        text_embeds_list = [torch.tensor(task_text_embeds[str(c)], dtype=torch.float32) for c in range(categories)]
                        text_embeddings = torch.cat(text_embeds_list).to(device)
                        logits = image_embeddings @ text_embeddings.T
                        
                        for i in range(len(task_labels)):
                            if task_labels[i] != -1:
                                logits_dict[task_name].append(logits[i].cpu().numpy())
                                labels_dict[task_name].append(task_labels[i].cpu().numpy())
                    else:
                        if not use_max_similarity: raise ValueError("This code path is for sensitivity analysis. Set use_max_similarity=True for ensemble mode.")
                        if categories != 2: raise ValueError("Max Similarity method currently supports binary classification only.")

                        embed_list_0 = task_text_embeds['0']
                        text_embeds_0 = torch.stack([torch.tensor(e, dtype=torch.float32) for e in embed_list_0]).to(device)
                        
                        embed_list_1 = task_text_embeds['1']
                        text_embeds_1 = torch.stack([torch.tensor(e, dtype=torch.float32) for e in embed_list_1]).to(device)
                        
                        sims_0 = image_embeddings @ text_embeds_0.T
                        sims_1 = image_embeddings @ text_embeds_1.T
                        
                        # For each category, we select the prompt that is most similar to the given image embedding
                        score_0, _ = torch.max(sims_0, dim=1)
                        score_1, _ = torch.max(sims_1, dim=1)
                        
                        logits = torch.stack([score_0, score_1], dim=1)

                        for i in range(len(task_labels)):
                            if task_labels[i] != -1:
                                logits_dict[task_name].append(logits[i].cpu().numpy())
                                labels_dict[task_name].append(task_labels[i].cpu().numpy())

        zeroshot_res_dict = {}
        
        zeroshot_res_dict = get_metrics(logits_dict=logits_dict, labels_dict=labels_dict, task_info=_task_info)
        if is_main_process():
            for task, metrics in zeroshot_res_dict.items():
                for metric, value in metrics.items():
                    writer.add_scalar(f"zeroshot_test/{task}/{hospital}/{metric}", value, 0)
    
    return zeroshot_res_dict