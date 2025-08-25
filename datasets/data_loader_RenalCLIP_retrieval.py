import numpy as np
import json
import os
from torch.utils.data import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
from utils.data_util import get_transforms, preprocess_text, restore_special_cases


hospital_map = {
    '厦门': 'XM',
    '连云': 'LY',
    '张掖': 'ZY',
    '瑞金': 'RJ',
    '山东': 'SD',
    'TCIA': 'TCIA',
}

class ImageCaptionDataset(Dataset):
    def __init__(self, args, hospital="internal"):
        super().__init__()

        list_path = args.datalist_3d
        if not os.path.exists(list_path):
            raise RuntimeError(f"{list_path} does not exits!")
        fin = open(list_path, 'r')
        list_dataset = json.load(fin)

        self.modalities = args.modalities
        self.filenames, _filenames = [], []
        self.pre_processed_type = args.pre_processed_type
        self.seed = args.seed
        self.split_file_name = args.split_file_name 
        self.dataset_names = args.dataset_names
        self.data_root = args.data_root
        self.hospital = hospital
        self.columns_to_read = ['医院', '医技号', 'tumor_side', 'CT报告', '拆分翻译']

        self.keys = ["img"]
        self.transforms = get_transforms(mode='test', args=args, keys=self.keys)

        for k, v in list_dataset.items():
            self.read_per_dataset(k, v)

    def read_per_dataset(self, dataset_name, dataset_attr):
        if self.hospital == 'internal':
            self.df = pd.read_excel(dataset_attr['info'],
                        sheet_name='internal_downstream',
                        engine='openpyxl', dtype={'医技号': str})
        else:
            self.df = pd.read_excel(dataset_attr['info'],
                            sheet_name=hospital_map[self.hospital],
                            engine='openpyxl', dtype={'医技号': str})

        dataset_root = dataset_attr['data_root']
        self.load_text_data(dataset_name, dataset_root)

        with open(self.split_file_name, 'r') as file:
            caption_data_split = json.load(file)

        if self.hospital == 'internal':
            _caption_data_split = caption_data_split['downstream_data_internal']
        else:
            _caption_data_split = caption_data_split['downstream_data_external'][self.hospital]

        _test_filenames = _caption_data_split['test_set']
        test_filenames = [item for item in self.filenames if item['patient_id'] in _test_filenames]
        
        self._filenames = test_filenames

    def load_text_data(self, dataset_name, dataset_root):
        for idx, row in enumerate(self.df.itertuples()):
            patient_id, tumor_side = str(getattr(row, '医技号')), getattr(row, 'tumor_side')
            _split_translation = getattr(row, '拆分翻译')
            if pd.isna(_split_translation):
                continue
            split_translation = json.loads(getattr(row, '拆分翻译'))

            split_findings, split_impression = split_translation['Report'], split_translation['Diagnosis']
            if tumor_side == 'L':
                kidney_findings = preprocess_text(split_findings.get('Left Kidney', ''))
                kidney_impression = preprocess_text(split_impression.get('Left Kidney', ''))
            else:
                kidney_findings = preprocess_text(split_findings.get('Right Kidney', ''))
                kidney_impression = preprocess_text(split_impression.get('Right Kidney', ''))

            info_dict = {
                'patient_id': patient_id,
                'tumor_side': tumor_side,
                'kidney_findings': kidney_findings,
                'kidney_impression': kidney_impression,
            }

            self.filenames.append(info_dict)

    def get_caption(self, sample):
        # load raw report from files
        kidney_findings, kidney_impression = sample['kidney_findings'], sample['kidney_impression']
        kidney_findings = restore_special_cases(kidney_findings)
        kidney_impression = restore_special_cases(kidney_impression)

        report_list = ['Findings: ', kidney_findings, '\n', 'Impression:', kidney_impression]
        kidney_caption = ''.join(report_list)

        return kidney_findings, kidney_impression, kidney_caption

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
        patient = sample['patient_id']
        _, _, caption = self.get_caption(sample)        
        images = self.get_images(sample)['img']

        return images, caption, patient


if __name__ == '__main__':
    print("hello world")
