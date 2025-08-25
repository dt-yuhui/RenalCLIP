import numpy as np
import json
import os
from torch.utils.data import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import torch
import re
import argparse
from utils.util import *
from utils.data_util import get_transforms


TXT_LABEL_DICT = {
    'N': 0,
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'DK': -1,
}

hospital_map = {
    '厦门': 'XM',
    '连云': 'LY',
    '张掖': 'ZY',
    '瑞金': 'RJ',
    '山东': 'SD',
    'TCIA': 'TCIA',
}

class DatasetRenalCLIPZeroshot(Dataset):
    def __init__(self, 
                 args, 
                 split='test', 
                 hospital="internal",
                 ):
        super().__init__()

        list_path = args.datalist_3d
        if not os.path.exists(list_path):
            raise RuntimeError(f"{list_path} does not exits!")
        fin = open(list_path, 'r')
        list_dataset = json.load(fin)

        self.split_file_name = args.split_file_name
        self.modalities = args.modalities
        self.filenames, _filenames = [], []
        self.split = split
        self.pre_processed = args.pre_processed
        self.pre_processed_type = args.pre_processed_type
        self.hospital = hospital
        self.zeroshot_tasks = args.zeroshot_attributes
        self.data_root = args.data_root
        self.args = args
        self.EXTERNAL_HOSPITALS = ['厦门', '连云', '张掖', '瑞金', '山东']

        self.keys = ['img']
        self.transforms = get_transforms(mode=split, args=args, keys=self.keys)

        for k, v in list_dataset.items():
            self.read_per_dataset(k, v)

    def read_per_dataset(self, dataset_name, dataset_attr):
        
        if self.hospital == 'internal':

            self.df = pd.read_excel(dataset_attr['info'],
                                    sheet_name='internal_downstream',
                                    engine='openpyxl', dtype={'医技号': str})

        elif self.hospital == 'external':

            df_list = []
            for hos in self.EXTERNAL_HOSPITALS:
                sheet = hospital_map[hos]          # 'XM' / 'LY' / ...
                df_tmp = pd.read_excel(dataset_attr['info'],
                                    sheet_name=sheet,
                                    engine='openpyxl', dtype={'医技号': str})
                df_list.append(df_tmp)
            self.df = pd.concat(df_list, ignore_index=True)

        else:

            self.df = pd.read_excel(dataset_attr['info'],
                                    sheet_name=hospital_map[self.hospital],
                                    engine='openpyxl', dtype={'医技号': str})
        
        dataset_root = dataset_attr['data_root']
        self.load_text_data(dataset_name, dataset_root)

        if self.split is None:
            self._filenames = self.filenames
        else:
            with open(self.split_file_name, 'r') as file:
                data_split = json.load(file)
        
            if self.hospital == 'internal':
                downstream_data_split = data_split['downstream_data_internal']

            elif self.hospital == 'external':
                downstream_data_split = {'train_set': [], 'valid_set': [], 'test_set': []}
                for hos in self.EXTERNAL_HOSPITALS:
                    _split = data_split['downstream_data_external'][hos]
                    for k in downstream_data_split.keys():
                        downstream_data_split[k].extend(_split.get(k, []))
            else:
                downstream_data_split = data_split['downstream_data_external'][self.hospital]

            _train_filenames = downstream_data_split.get('train_set', [])
            _valid_filenames = downstream_data_split.get('valid_set', [])
            _test_filenames = downstream_data_split.get('test_set', [])

            train_filenames = [item for item in self.filenames if item['patient_id'] in _train_filenames]
            valid_filenames = [item for item in self.filenames if item['patient_id'] in _valid_filenames]
            test_filenames = [item for item in self.filenames if item['patient_id'] in _test_filenames]

            if self.split == "train":
                self._filenames = train_filenames
            elif self.split == 'valid':
                self._filenames = valid_filenames
            elif self.split == 'test':
                self._filenames = test_filenames
            else:
                raise NotImplementedError(f'split mode: {self.split} not implemented!')

            if len(self._filenames) > 0:
                pass
            else:
                print(f"No data found in {self.split} set")

    def _sample_filenames(self, filenames, ratio):

        sample_size = int(len(filenames) * ratio)
        return random.sample(filenames, sample_size)

    def load_text_data(self, dataset_name, dataset_root):
        for row in self.df.itertuples():
            patient_id, tumor_side = getattr(row, '医技号'), getattr(row, 'tumor_side')
            _tumor_side = 'left' if tumor_side == "L" else "right"

            cls_dict = {}

            if 'BMC' in self.zeroshot_tasks:
                cls = getattr(row, '任务1_1识别良恶性')
                # benign malignant
                if cls == 1:
                    bmc_cls = 0
                elif cls == 2:
                    bmc_cls = 1
                elif cls == -1:
                    bmc_cls = -1
                cls_dict['BMC'] = bmc_cls

            if 'IC' in self.zeroshot_tasks:
                if self.hospital != '张掖':
                    cls = getattr(row, '任务2_1侵袭性鉴别')
                else:
                    if self.args.ZY_IC == 'v2':
                        cls = getattr(row, '任务2_1侵袭性鉴别_v2')
                    else:
                        cls = getattr(row, '任务2_1侵袭性鉴别')
                # exclude label 2.5 in invasive classification task
                if cls == 1:
                    ic_cls = 0
                elif cls == 2:
                    ic_cls = 1
                elif cls == -1:
                    ic_cls =-1
                cls_dict['IC'] = ic_cls

            # only one kidney
            info_dict = {'patient_id': patient_id, 'cls': cls_dict, 'tumor_side': tumor_side, "kidney_side": tumor_side}
            self.filenames.append(info_dict)
    
    def get_images(self, sample):
        patient, tumor_side = sample['patient_id'], sample['tumor_side']
        data_path = os.path.join(self.data_root, 'nii_npy_aligned_preprocessed', self.pre_processed_type, patient)

        data = {}
        CT_listdir = os.listdir(data_path)

        exist_modalities = list({file[0] for file in CT_listdir})
        if "A" in exist_modalities:
            random_select_modality = "A"
        elif "V" in exist_modalities:
            random_select_modality = "V"
        else:
            raise NotImplementedError(f"No valid modality available in patient: {patient}")

        data["img"] = os.path.join(data_path, f"{random_select_modality}_{tumor_side}_image_data.npy")

        CT_volumes = self.transforms(data)
        return CT_volumes
    

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        sample = self._filenames[idx]
        patient, cls, kidney_side = sample['patient_id'], sample['cls'], sample['kidney_side']

        data = {
            'labels': cls,
            'patients': patient,
            'kidneys_side': kidney_side,
        }

        return data


if __name__ == '__main__':
    print("hello world")
