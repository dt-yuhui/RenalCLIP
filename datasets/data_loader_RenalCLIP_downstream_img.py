import numpy as np
import json
import os
from torch.utils.data import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import torch
import re
from sklearn.model_selection import train_test_split
import argparse
from utils.util import *
from utils.data_util import get_transforms


BERT_BASE_DIR = fr'/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/pretrained_models/language_family'
# one-hot encoding
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
DOUBLE_SIDES_TASK_LIST = ["multi_task"]
hospital_map = {
    '厦门': 'XM',
    '连云': 'LY',
    '张掖': 'ZY',
    '瑞金': 'RJ',
    '山东': 'SD',
    'TCIA': 'TCIA',
}

class DatasetRenalCLIPDownstreamImg(Dataset):
    def __init__(self, args, split='train', hospital="internal", transforms_mode=None):
        super().__init__()

        list_path = args.datalist_3d
        if not os.path.exists(list_path):
            raise RuntimeError(f"{list_path} does not exits!")
        fin = open(list_path, 'r')
        list_dataset = json.load(fin)

        self.args = args
        self.split_file_name = args.split_file_name
        self.modalities = args.modalities
        self.path2sent = {}
        self.filenames, _filenames = [], []
        self.split = split
        self.sample_ratio = args.sample_ratio
        self.pre_processed = args.pre_processed
        self.pre_processed_type = args.pre_processed_type
        self.split_seed = args.seed
        self.downstream_task = args.downstream_task
        self.multi_task_ids = args.multi_task_ids
        self.finetune_type = args.finetune_type
        self.hospital = hospital
        self.multiple_3d = args.multiple_3d
        self.modalities_pool = args.modalities_pool
        self.data_root = args.data_root
        self.transforms_mode = self.split if transforms_mode is None else transforms_mode

        if "survival" in self.downstream_task:
            # no prognosis outcomes in LY & SD
            self.EXTERNAL_HOSPITALS = ['厦门', '张掖', '瑞金']
        else:
            self.EXTERNAL_HOSPITALS = ['厦门', '连云', '张掖', '瑞金', '山东']

        self.keys = list(self.modalities)
        self.transforms = get_transforms(mode=self.transforms_mode, args=args, keys=self.keys, downstream=True)

        for k, v in list_dataset.items():
            self.read_per_dataset(k, v)

        self.task_distribution = self.get_task_and_category_distribution() 

    def read_per_dataset(self, dataset_name, dataset_attr):
        if self.finetune_type == 'pretrain':
            # for image backbone pre-train, using multi-task learning framework
            self.df = pd.read_excel(dataset_attr['info'], 
                                    sheet_name='internal_pretrain', 
                                    engine='openpyxl', 
                                    dtype={"医技号": str})
        else:
            # internal validation
            if self.hospital == 'internal':
                self.df = pd.read_excel(dataset_attr['info'],
                                        sheet_name='internal_downstream',
                                        engine='openpyxl', 
                                        dtype={'医技号': str})

            elif self.hospital == 'external':
                # Combine the 5 cohorts to form the combined external test cohort.
                df_list = []
                for hos in self.EXTERNAL_HOSPITALS:
                    sheet = hospital_map[hos]          # 'XM' / 'LY' / ...
                    df_tmp = pd.read_excel(dataset_attr['info'],
                                        sheet_name=sheet,
                                        engine='openpyxl', 
                                        dtype={'医技号': str})
                    df_list.append(df_tmp)
                self.df = pd.concat(df_list, ignore_index=True)

            else:
                # The individual hospitals (e.g., XM) can still be specified separately.
                self.df = pd.read_excel(dataset_attr['info'],
                                        sheet_name=hospital_map[self.hospital],
                                        engine='openpyxl', 
                                        dtype={'医技号': str})
        dataset_root = dataset_attr['data_root']
        self.load_text_data(dataset_name, dataset_root)

        if self.split is None:
            self._filenames = self.filenames
        else:
            with open(self.split_file_name, 'r') as file:
                data_split = json.load(file)
            
            if self.finetune_type == 'pretrain':
                downstream_data_split = data_split['pretrain_data']
            else:
                if self.hospital == 'internal':
                    downstream_data_split = data_split['downstream_data_internal']

                elif self.hospital == 'external':
                    # Combine the 5 cohorts to form the combined external test cohort.
                    downstream_data_split = {'train_set': [], 'valid_set': [], 'test_set': []}
                    for hos in self.EXTERNAL_HOSPITALS:
                        _split = data_split['downstream_data_external'][hos]
                        for k in downstream_data_split.keys():
                            downstream_data_split[k].extend(_split.get(k, []))
                else:
                    downstream_data_split = data_split['downstream_data_external'][self.hospital]

            if self.hospital == 'internal' or self.hospital == 'pretrain':
                if self.sample_ratio < 1:
                    # for data efficiency experiment
                    sampled_downstream_data_split = downstream_data_split.get('train_set_proportion_by_seed', [])
                    _sampled_downstream_data_split = sampled_downstream_data_split.get(f"{self.sample_ratio}", [])
                    train_patient_ids = _sampled_downstream_data_split.get(f'seed_{self.split_seed}', [])            
                elif self.sample_ratio == 1.0:
                    train_patient_ids = downstream_data_split.get('train_set', [])
                else:
                    raise NotImplementedError(f"sample ratio: {self.sample_ratio} not implemented!")
            else:
                train_patient_ids = []
            valid_patient_ids = downstream_data_split.get('valid_set', [])
            test_patient_ids = downstream_data_split.get('test_set', []) if self.finetune_type != 'pretrain' else downstream_data_split.get('valid_set', [])

            train_filenames = [item for item in self.filenames if item['patient_id'] in train_patient_ids]
            valid_filenames = [item for item in self.filenames if item['patient_id'] in valid_patient_ids]
            test_filenames = [item for item in self.filenames if item['patient_id'] in test_patient_ids]

            if self.split == "train":
                self._filenames = train_filenames
            elif self.split == 'valid':
                self._filenames = valid_filenames
            elif self.split == 'test':
                self._filenames = test_filenames
            else:
                raise NotImplementedError(f'split mode: {self.split} not implemented!')

            if self.split == 'train':    
                if len(self._filenames) > 0:
                    # weighted data sampler
                    labels = [filename['cls'] for filename in self._filenames]

                    if self.downstream_task == 'multi_task':
                        pass
                    else:
                        if "survival" in self.downstream_task:
                            labels = [filename['cls']['rfs']["event"] for filename in self._filenames]
                        else:
                            labels = [filename['cls'][self.downstream_task] for filename in self._filenames]
                        labels = torch.tensor(labels)
                        self.class_counts = torch.bincount(labels)
                        self.class_weights = 1.0 / self.class_counts.float()
                        self.sample_weights = self.class_weights[labels]
                else:
                    print(f"No data found in {self.split} set")

    def load_text_data(self, dataset_name, dataset_root):

        for row in self.df.itertuples():
            patient_id, tumor_side = getattr(row, '医技号'), getattr(row, 'tumor_side') # get patient_id, tumor_side
            patient_modalities = getattr(row, "期相").split(',') # get modality info
            
            if self.downstream_task == 'BMC':
                cls = getattr(row, '任务1_1识别良恶性')
            elif self.downstream_task == 'IC':
                if self.hospital != '张掖':
                    cls = getattr(row, '任务2_1侵袭性鉴别')
                else:
                    if self.args.ZY_IC == 'v2':
                        cls = getattr(row, '任务2_1侵袭性鉴别_v2')
                    else:
                        cls = getattr(row, '任务2_1侵袭性鉴别')
            elif 'RENAL' in self.downstream_task:
                cls = getattr(row, self.downstream_task)

            if self.downstream_task == 'BMC':
                # benign vs. malignant
                if cls == 1:
                    cls = 0
                elif cls == 2:
                    cls = 1
                elif cls == -1:
                    continue
                else:
                    raise ValueError(f"Invalid BMC label: {cls}")
                cls = {'BMC': cls}

            elif self.downstream_task == 'IC':
                # aggressive vs. indolent
                if cls == 1:
                    cls = 0
                elif cls == 2:
                    cls = 1
                elif cls == -1:
                    continue
                else:
                    raise ValueError(f"Invalid IC label: {cls}")
                cls = {'IC': cls}
            
            elif 'RENAL' in self.downstream_task:
                # R.E.N.A.L. score
                if self.downstream_task == 'RENAL_AIA':
                    if cls != -1:
                        cls = cls.upper()
                    if cls == 'A':
                        cls = 0
                    elif cls == 'P':
                        cls = 1
                    elif cls == 'X':
                        if self.args.num_classes == 3:
                            cls = 2
                        else:
                            continue
                    elif cls == -1:
                        continue
                    
                    cls = {self.downstream_task: cls}
                
                else:
                    cls = float(cls)
                    cls = int(cls-1) if cls!=-1 else -1
                    if cls == -1:
                        continue
                    cls = {self.downstream_task: cls}

            elif "survival" in self.downstream_task:
                # OS, DSS, RFS endpoints
                os_event, os_duration = getattr(row, "OS"), float(getattr(row, "OStime"))
                rfs_event, rfs_duration = getattr(row, "RFS"), float(getattr(row, "RFStime"))
                dss_event, dss_duration = getattr(row, "DSS"), float(getattr(row, "DSStime"))

                bmc = getattr(row, '任务1_1识别良恶性')
                
                # Only include prognosis data for malignant cases.
                if bmc == 1:
                    continue
                
                # Train the model using only the RFS endpoint, then evaluate the trained model on the OS, DSS, and RFS endpoints.
                if self.split == 'train':
                    if rfs_event == -1:
                        continue
                
                cls = {
                    'os': {
                        "event": os_event,
                        "duration": os_duration
                    },
                    'rfs': {
                        "event": rfs_event,
                        "duration": rfs_duration
                    },
                    'dss': {
                        "event": dss_event,
                        "duration": dss_duration
                    }
                }

            elif self.downstream_task == 'multi_task':
                # multi_task_ids = [1, 2, 3, 7, 8, 9, 11, 13, 14, 15, 17, 18, 19, 20]
                multi_task_ids = self.multi_task_ids

                left_cls, right_cls = {}, {}
                for multi_task_id in multi_task_ids:
                    task_name = f"text_label_{multi_task_id}"
                    _left_task_cls, _right_task_cls = getattr(row, "left_"+task_name), getattr(row, "right_"+task_name)
                    left_task_cls, right_task_cls = TXT_LABEL_DICT[_left_task_cls], TXT_LABEL_DICT[_right_task_cls]
                    left_cls.update({task_name: left_task_cls})
                    right_cls.update({task_name: right_task_cls})

            else:
                raise NotImplementedError(f'downstream task {self.downstream_task} not implemented')

            if any(task in self.downstream_task for task in DOUBLE_SIDES_TASK_LIST):
                # consider two kidneys
                left_info_dict = {'patient_id': patient_id, 'cls': left_cls, 'tumor_side': tumor_side, "kidney_side": "L", "modalities": patient_modalities}
                right_info_dict = {'patient_id': patient_id, 'cls': right_cls, 'tumor_side': tumor_side, "kidney_side": "R", "modalities": patient_modalities}
                if left_cls:
                    self.filenames.append(left_info_dict)
                if right_cls:
                    self.filenames.append(right_info_dict)

            else:
                # only one kidney
                info_dict = {'patient_id': patient_id, 'cls': cls, 'tumor_side': tumor_side, "kidney_side": tumor_side, "modalities": patient_modalities}
                self.filenames.append(info_dict)

    def get_task_and_category_distribution(self):
        task_distribution = {"Task": self.downstream_task}

        if self.downstream_task == 'multi_task':
            all_task_labels = {task: [] for task in self._filenames[0]['cls'].keys()}
            for filename in self._filenames:
                for task_name, cls in filename['cls'].items():
                    all_task_labels[task_name].append(cls)

            task_distribution = {}

            for task_name, labels in all_task_labels.items():
                labels_tensor = torch.tensor(labels)

                temp_labels = labels_tensor.clone()
                temp_labels[temp_labels == -1] = temp_labels.max() + 1

                class_counts = torch.bincount(temp_labels)

                class_counts = class_counts[:-1]

                unique = np.arange(len(class_counts))
                counts = class_counts.numpy()

                label_distribution = dict(zip(unique, counts))
                task_distribution[task_name] = label_distribution

        elif "survival" in self.downstream_task:
            labels = [filename['cls']['rfs']["event"] for filename in self._filenames]

            unique, counts = np.unique(labels, return_counts=True)
            label_distribution = dict(zip(unique, counts))
            
            task_distribution[self.downstream_task] = label_distribution

        else:
            labels = [filename['cls'][self.downstream_task] for filename in self._filenames]

            unique, counts = np.unique(labels, return_counts=True)
            label_distribution = dict(zip(unique, counts))
            
            task_distribution[self.downstream_task] = label_distribution

        return task_distribution
    
    def get_images(self, sample):
        patient = sample['patient_id']
        kidney_side = sample['kidney_side']
        data_path = os.path.join(self.data_root, 'nii_npy_aligned_preprocessed', self.pre_processed_type, patient)

        data = {}
        CT_listdir = os.listdir(data_path)

        # single modality, using enhanced phase, i.e, A & V
        # check if modality A exists, if not, select modality V
        if len(self.keys) == 1:
            # 4d -> 3d
            select_modality = self.keys[0]
            if self.split == 'train':
                if self.multiple_3d:
                    # random select an available modality
                    patient_modalities = sample["modalities"]
                    _patient_modalities = list(set(patient_modalities)&set(self.modalities_pool))

                    random_select_modality = random.choice(_patient_modalities)
                else:
                    random_select_modality = self.keys[0]
                
                data[select_modality] = os.path.join(data_path, f'{random_select_modality}_{kidney_side}_image_data.npy')

            # 3d
            else:
                if "A" in self.keys:
                    no_A_flag = True
                    for key in self.keys:
                        if f'{key}_{kidney_side}_image_data.npy' in CT_listdir:
                            no_A_flag = False
                            data[key] = os.path.join(data_path, f'{key}_{kidney_side}_image_data.npy')
                    if no_A_flag:
                        data["A"] = os.path.join(data_path, f'V_{kidney_side}_image_data.npy')
                else:
                    for key in self.keys:
                        if f'{key}_{kidney_side}_image_data.npy' in CT_listdir:
                            data[key] = os.path.join(data_path, f'{key}_{kidney_side}_image_data.npy')
        # multi-modality
        else:
            for key in self.keys:
                if f'{key}_{kidney_side}_image_data.npy' in CT_listdir:
                    data[key] = os.path.join(data_path, f'{key}_{kidney_side}_image_data.npy')

        CT_volumes = self.transforms(data)

        return CT_volumes

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        sample = self._filenames[idx]
        patient, cls, kidney_side = sample['patient_id'], sample['cls'], sample['kidney_side']
        images = self.get_images(sample)

        return images, cls, patient, kidney_side


if __name__ == '__main__':
    print("hello world")
