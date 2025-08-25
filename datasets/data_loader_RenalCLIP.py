import numpy as np
import json
import os
from torch.utils.data import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import torch
import re
import random

from utils.data_util import get_transforms, preprocess_text

BASE_DIR = fr'/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/RenalCLIP'
LANGUAGE_BASE_DIR = fr'/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/pretrained_models/language_family'


class DatasetRenalCLIP(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()

        list_path = args.datalist_3d
        if not os.path.exists(list_path):
            raise RuntimeError(f"{list_path} does not exits!")
        fin = open(list_path, 'r')
        list_dataset = json.load(fin)

        self.modalities = args.modalities
        self.txt_transform = args.txt_transform
        self.path2sent = {}
        self.filenames, _filenames = [], []
        self.split = split
        self.max_words = args.max_words
        self.bert_type = args.bert_type
        self.pre_processed_type = args.pre_processed_type
        self.seed = args.seed
        self.split_file_name = args.split_file_name 
        self.multiple_3d = args.multiple_3d
        self.llm2vec = args.llm2vec
        self.llm2vec_type = args.llm2vec_type

        if self.llm2vec:
            if self.split == 'train':
                self.text_features_indices = range(5)
            else:
                self.text_features_indices = [0]

        self.keys = ["img_L", "img_R"]

        self.transforms = get_transforms(mode=split, args=args, keys=self.keys)


        for k, v in list_dataset.items():
            self.read_per_dataset(k, v)

    def read_per_dataset(self, dataset_name, dataset_attr):
        self.df = pd.read_excel(dataset_attr['info'], sheet_name='internal_pretrain', engine='openpyxl')
        dataset_root = dataset_attr['data_root']
        self.load_text_data(dataset_name, dataset_root)

        with open(self.split_file_name, 'r') as file:
            data_split = json.load(file)
        
        pretrain_data_split = data_split['pretrain_data']
        _train_filenames = pretrain_data_split['train_set']
        _valid_filenames = pretrain_data_split['valid_set']

        train_filenames = [item for item in self.filenames if item['patient_id'] in _train_filenames]
        valid_filenames = [item for item in self.filenames if item['patient_id'] in _valid_filenames]

        if self.split == "train":
            self._filenames = train_filenames
        elif self.split == 'valid':
            self._filenames = valid_filenames
        else:
            raise NotImplementedError(f'split mode: {self.split} not implemented!')

    def load_text_data(self, dataset_name, dataset_root):
        for idx, row in enumerate(self.df.itertuples()):
            patient_id = str(getattr(row, '医技号'))
            split_translation = json.loads(getattr(row, '拆分翻译'))

            split_findings, split_impression = split_translation['Report'], split_translation['Diagnosis']
            left_kidney_findings, left_kidney_impression = preprocess_text(split_findings.get('Left Kidney', '')), preprocess_text(split_impression.get('Left Kidney', ''))
            right_kidney_findings, right_kidney_impression = preprocess_text(split_findings.get('Right Kidney', '')), preprocess_text(split_impression.get('Right Kidney', ''))

            if len(left_kidney_findings) == 0:
                left_kidney_findings = 'No abnormalities in the left kidney.'

            if len(right_kidney_findings) == 0:
                right_kidney_findings = 'No abnormalities in the right kidney.'

            if self.llm2vec:
                left_kidney_llm2vec_features, right_kidney_llm2vec_features = [], []
                for i in self.text_features_indices:
                    llm2vec_path = os.path.join(BASE_DIR, "llm2vec_features", self.llm2vec_type, patient_id)
                    left_kidney_llm2vec_feature = np.load(os.path.join(llm2vec_path, f"left_kidney_llm2vec_{i}.npy"))
                    right_kidney_llm2vec_feature = np.load(os.path.join(llm2vec_path, f"right_kidney_llm2vec_{i}.npy"))
                    left_kidney_llm2vec_features.append(left_kidney_llm2vec_feature)
                    right_kidney_llm2vec_features.append(right_kidney_llm2vec_feature)

            info_dict = {
                'patient_id': patient_id,
                'left_kidney_findings': left_kidney_findings,
                'left_kidney_impression': left_kidney_impression,
                'right_kidney_findings': right_kidney_findings,
                'right_kidney_impression': right_kidney_impression,
                'left_kidney_llm2vec_features': left_kidney_llm2vec_features if self.llm2vec else None,
                'right_kidney_llm2vec_features': right_kidney_llm2vec_features if self.llm2vec else None,
            }
            self.filenames.append(info_dict)


    def get_images(self, sample):
        patient = sample['patient_id']
        data_path = os.path.join(BASE_DIR, 'nii_npy_aligned_preprocessed', self.pre_processed_type, patient)

        data = {}
        CT_listdir = os.listdir(data_path)

        # 4d -> multiple 3d
        exist_modalities = list({file[0] for file in CT_listdir})
        if self.multiple_3d:
            if self.split == 'train':
                random_select_modality = random.choice(exist_modalities)
            else:
                if "A" in exist_modalities:
                    random_select_modality = "A"
                elif "V" in exist_modalities:
                    random_select_modality = "V"
                else:
                    raise NotImplementedError(f"No valid modality available in patient: {patient}")
        else:
            if "A" in exist_modalities:
                random_select_modality = "A"
            elif "V" in exist_modalities:
                random_select_modality = "V"
            else:
                raise NotImplementedError(f"No valid modality available in patient: {patient}")

        for side in ["L", "R"]:
            data[f"img_{side}"] = os.path.join(data_path, f"{random_select_modality}_{side}_image_data.npy")

        CT_volumes = self.transforms(data)

        return CT_volumes

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        sample = self._filenames[idx]
        patient = sample['patient_id']
        
        images = self.get_images(sample)
        
        raw_reports = {
            'left_kidney_findings': sample['left_kidney_findings'],
            'left_kidney_impression': sample['left_kidney_impression'],
            'right_kidney_findings': sample['right_kidney_findings'],
            'right_kidney_impression': sample['right_kidney_impression'],
        }

        if self.llm2vec:
            random_select_feature_index = random.choice(self.text_features_indices)
            llm2vec_features = {
                k: v[random_select_feature_index] for k, v in sample.items() if "llm2vec" in k
            }
        else:
            llm2vec_features = {
                k: v for k, v in sample.items() if "llm2vec" in k
            }

        return images, raw_reports, llm2vec_features, patient


if __name__ == '__main__':
    print("hello world")