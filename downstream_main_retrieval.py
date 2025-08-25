import os
import numpy as np
import torch
import pandas as pd
import sys
import openpyxl

IMAGE_EMBEDDINGS_ROOT = "/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/RenalCLIP/image_embeddings"
TEXT_EMBEDDINGS_ROOT = "/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/RenalCLIP/retrieval_text_embeddings"

RESULTS_DIR = "./"
EXCEL_RESULTS_FILE = os.path.join(RESULTS_DIR, "retrieval_demo.xlsx")

def get_patient_ids_for_cohort(hospital_name):
    """
    Return a list of all patient IDs for the specified hospital
    """

    try:
        from utils.parser import get_report_generation_args
        from datasets.data_loader_RenalCLIP_retrieval import ImageCaptionDataset
    except ImportError as e:
        print(f"Error importing modules. Error: {e}")
        return []

    args = get_report_generation_args()
    dataset = ImageCaptionDataset(args, hospital=hospital_name)

    patient_ids = [item['patient_id'] for item in dataset._filenames if isinstance(item, dict) and 'patient_id' in item]
    return patient_ids

def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k):
    ''' Compute the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def img_text_retrieval_precision(img_feat, txt_feat):
    """
    To assess the precision of image-text retrieval, 
    this function calculates the average accuracy for retrieving the correct text for a given image, and vice-versa, at specified top-k ranks.
    """
    with torch.no_grad():
        scores = img_feat.mm(txt_feat.t())
        scores1 = scores.transpose(0, 1)
        bz = scores.size(0)
        labels = torch.arange(bz, device=scores.device).long()

        i2t_acc1, i2t_acc3, i2t_acc5 = precision_at_k(scores, labels, top_k=[1, 3, 5])
        t2i_acc1, t2i_acc3, t2i_acc5 = precision_at_k(scores1, labels, top_k=[1, 3, 5])

        return {
            'i2t_acc1': i2t_acc1.item(), 't2i_acc1': t2i_acc1.item(), 'acc1': ((i2t_acc1 + t2i_acc1) / 2.).item(),
            'i2t_acc3': i2t_acc3.item(), 't2i_acc3': t2i_acc3.item(), 'acc3': ((i2t_acc3 + t2i_acc3) / 2.).item(),
            'i2t_acc5': i2t_acc5.item(), 't2i_acc5': t2i_acc5.item(), 'acc5': ((i2t_acc5 + t2i_acc5) / 2.).item(),
        }

model_configs = {
    'ours': {'img_encoder_type': 'cnn', 'text_embed_name': 'llm2vec-rad', 'model_name_for_img_embedding': 'ours'},
}

HOSPITAL_COHORTS = ["internal", "瑞金", "山东", "张掖", "厦门"]


def run_retrieval_analysis():

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_final_results = []

    for hospital_name in HOSPITAL_COHORTS:
        print(f"\n===== Starting process cohort: {hospital_name} =====")
        all_patient_ids = get_patient_ids_for_cohort(hospital_name)
        
        if not all_patient_ids:
            print(f"Cohort: '{hospital_name}' Patient ID not found. Skipping.")
            continue
        
        print(f"{len(all_patient_ids)} patients found.")

        for model_name, config in model_configs.items():
            print(f"\n--- Start process model: {model_name} for cohort: {hospital_name} ---")
            
            all_img_feats_list, all_txt_feats_list = [], []
            for patient_id in all_patient_ids:
                img_path = os.path.join(IMAGE_EMBEDDINGS_ROOT, config['model_name_for_img_embedding'], patient_id, 'image_embedding.npy')
                txt_path = os.path.join(TEXT_EMBEDDINGS_ROOT, config['text_embed_name'], patient_id, "text_embedding.npy")
                try:
                    img_feat = torch.from_numpy(np.load(img_path)).float().to(device).squeeze()
                    txt_feat = torch.from_numpy(np.load(txt_path)).float().to(device).squeeze()
                    all_img_feats_list.append(img_feat)
                    all_txt_feats_list.append(txt_feat)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"Load embedding from {patient_id} occurs an error: {e}")

            if not all_img_feats_list or not all_txt_feats_list:
                print(f"Error: Can't load any embedding from model -- {model_name} at cohort -- {hospital_name}")
                continue

            all_img_feats_tensor = torch.stack(all_img_feats_list)
            all_txt_feats_tensor = torch.stack(all_txt_feats_list)
            print(f"Sucessfully load {all_img_feats_tensor.shape[0]} image-text pairs")

            with torch.no_grad():
                retrieval_metrics = img_text_retrieval_precision(all_img_feats_tensor, all_txt_feats_tensor)
            
            print(f"Calculation completed: {retrieval_metrics}")

            for metric_name, value in retrieval_metrics.items():
                all_final_results.append({
                    'model': model_name,
                    'cohort': hospital_name,
                    'metric': metric_name,
                    'value': value / 100.0
                })

    # save results
    if not all_final_results:
        print("\n\n Can't generate any results. Please check data paths and file contents.")
        return

    print(f"\n\nAll Retrieval metrics have been successfully calculated!")
    final_df = pd.DataFrame(all_final_results)
    final_df.to_excel(EXCEL_RESULTS_FILE, index=False, engine='openpyxl')
    
    print(f"Results saved to: {EXCEL_RESULTS_FILE}")
    print("Results preview:")
    print(final_df.head())

if __name__ == "__main__":
    print("Starting retrieval...")
    run_retrieval_analysis()
    print("\nAnalysis complete.")