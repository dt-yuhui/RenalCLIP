from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Resized,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianSmoothd,
    RandRotated,
    ToTensord,
    MapTransform,
    Transform,
    Transposed,
    RandGridDistortiond,
    RandCoarseShuffled,
    Rand3DElasticd,
    RandAffined,
    RandAdjustContrastd,
    RandFlipd,

)

import torch
import numpy as np
import re
import random
import os
import torch.distributed
from transformers import BertTokenizer, AutoModelForCausalLM, AutoTokenizer
import json
import torch.distributed as dist
from multiprocessing import Manager


RANDOM_STATE = 42
ABBR_TO_IGNORE = ["R.E.N.A.L"]
BERT_BASE_DIR = fr'/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/taoyuhui/RenalCLIP/pretrained_models/language_family'


# Convert the dictionary to a string
def dict_to_str(d, sep="; "):
    if isinstance(d, dict):
        return sep.join([f"{k}: {dict_to_str(v, sep)}" for k, v in d.items()])
    else:
        return str(d)

def replace_special_cases(text):
    # Replacing a decimal point while keeping its original form
    if isinstance(text, dict):
        text = dict_to_str(text)
    
    text = re.sub(r'(\d)\.(\d)', r'\1<dot>\2', text)

    # Replace the abbreviated form while retaining its original form
    for i, abbr in enumerate(ABBR_TO_IGNORE):
        text = text.replace(abbr, f'<abbr{i}>')
    return text


def restore_special_cases(text):
    # Restore the original decimal point
    text = text.replace('<dot>', '.')

    # Restore the original abbreviation
    for i, abbr in enumerate(ABBR_TO_IGNORE):
        text = text.replace(f'<abbr{i}>', abbr)
    return text


def remove_numbered_prefixes(text):
    # Define the regex for matching a serial number
    pattern = r'\b\d+\s*[\.\)、,\-]\s*'

    # Use a regular expression to replace the matched serial number with an empty string
    stripped_text = re.sub(pattern, '', text)

    return stripped_text


def ensure_sentence_endings(text):
    sub_sentences = re.split(r'(\n)', text)

    # Add the missing punctuation
    new_sub_sentences = []
    for sub in sub_sentences:
        if sub.strip() and not re.match(r'(\n;。\.；)', sub):
            if not sub.strip().endswith(('。', '；', '.', '!', '?', ';')):
                sub = sub.strip() + '.'
        new_sub_sentences.append(sub)

    return ''.join(new_sub_sentences)


def preprocess_text(text):

    text = replace_special_cases(text)
    text = remove_numbered_prefixes(text)
    text = ensure_sentence_endings(text)

    return text


def sentence_shuffling(text):
    # Split sentences
    split_text = text.replace(';', ';|').replace('.', '.|').replace('；', '；|').replace('。', '。|').replace('\n', '')
    sub_sentences = split_text.split('|')

    # Remove empty sub-sentences
    sub_sentences = [s for s in sub_sentences if s.strip() and not re.match(r'[\n;。.；]', s)]

    random.shuffle(sub_sentences)
    shuffled_text = ''.join(sub_sentences)

    return shuffled_text


def get_transforms(mode, args, keys=('N', 'A', 'V', 'D'), downstream=False, resize=False):

    transform_list = [
        LoadImaged(keys=keys, allow_missing_keys=True),
    ]

    if mode == 'train':
        transform_list += [
            RandSpatialCropd(keys=keys,
                             roi_size=(args.cropsize_3d, 
                                       args.cropsize_3d, 
                                       args.crop_slices),
                             allow_missing_keys=True,                             
                             )
        ]

        if downstream:
            transform_list += [
                RandFlipd(keys=keys, prob=0.5, spatial_axis=[0], allow_missing_keys=True),
            ]

        transform_list += [
            RandAffined(keys=keys,
                        translate_range=(args.RandTranslated_range_in_plane, 
                                         args.RandTranslated_range_in_plane, 
                                         args.RandTranslated_range_out_of_plane),
                                        #  x, y, z shift
                        scale_range=(0.1, 0.1, 0), # x, y, z scale [0.9, 1.1]
                        rotate_range=(args.RandRotated_range_out_of_plane, args.RandRotated_range_out_of_plane, args.RandRotated_range_in_plane), 
                        prob=1, 
                        allow_missing_keys=True,
                        padding_mode='zeros',
                        ),
        ]

        transform_list += [
            RandScaleIntensityd(keys=keys, factors=0.1, prob=1, allow_missing_keys=True),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=1, allow_missing_keys=True),
            RandAdjustContrastd(keys=keys, prob=0.5, gamma=(0.5, 2.0), allow_missing_keys=True),
        ]

    else:
        transform_list += [
            CenterSpatialCropd(
                keys=keys,
                roi_size=(
                        args.cropsize_3d, 
                        args.cropsize_3d, 
                        args.crop_slices),
                allow_missing_keys=True,
                )
        ]
        if resize:
            transform_list += [
                Resized(
                    keys=keys,
                    spatial_size=(
                            args.resize_3d, 
                            args.resize_3d, 
                            args.resize_slices),
                    allow_missing_keys=True,
                    )
            ]

    transform_list += [
        ToTensord(keys=keys, allow_missing_keys=True),
    ]

    transform_list += [
        Transposed(keys=keys, indices=(0, 3, 1, 2), allow_missing_keys=True)    # C, W, H, D -> C, D, W, H
    ]

    transforms = Compose(transform_list)

    return transforms


def convert_to_tensor(value):
    if isinstance(value, dict):
        return {k: convert_to_tensor(v) for k, v in value.items()}
    elif isinstance(value, torch.Tensor):
        return value.clone().detach()
    else:
        return torch.tensor(value)


def check_label_form(patient_CT_labels):
   
    if patient_CT_labels and isinstance(patient_CT_labels[0], dict):
        first_inner_dict = next(iter(patient_CT_labels[0].values()))
        
        if isinstance(first_inner_dict, dict):
            inner_key_set = first_inner_dict.keys()
            task_labels = {key: [] for key in inner_key_set}
            
            for outer_dict in patient_CT_labels:
                inner_dict = next(iter(outer_dict.values()))
                for task_name, label in inner_dict.items():
                    task_labels[task_name].append(convert_to_tensor(label))
            
            for task_name in task_labels:
                task_labels[task_name] = torch.stack(task_labels[task_name])
            
            patient_CT_labels = task_labels
        else:
            for outer_dict in patient_CT_labels:
                for k, v in outer_dict.items():
                    outer_dict[k] = convert_to_tensor(v)

            patient_CT_labels = torch.utils.data.default_collate(patient_CT_labels)
            
    else:
        patient_CT_labels = torch.tensor(patient_CT_labels)

    return patient_CT_labels


def check_survival_label_form(patient_CT_labels):
    # Initialize dictionaries to accumulate events and durations for each outcome
    task_labels = {
        'os': {'event': [], 'duration': []},
        'rfs': {'event': [], 'duration': []},
        'dss': {'event': [], 'duration': []},
    }
    
    # Accumulate labels
    for label_dict in patient_CT_labels:
        for outcome in ['os', 'rfs', 'dss']:
            task_labels[outcome]['event'].append(label_dict[outcome]['event'])
            task_labels[outcome]['duration'].append(label_dict[outcome]['duration'])

    # Convert lists to tensors
    # Use torch.tensor since these are lists of numbers (events and durations)
    for outcome in ['os', 'rfs', 'dss']:
        task_labels[outcome]['event'] = torch.tensor(task_labels[outcome]['event'])
        task_labels[outcome]['duration'] = torch.tensor(task_labels[outcome]['duration'])

    return task_labels

def custom_collate_fn_downstream_img(args):
    num_modalities = len(args.modalities)
    def collate_fn(batch):
        patient_modalities, patients, kidneys_side = [], [], []
        patient_CT_volumes = []
        patient_CT_labels = []
        for b in batch:
            _CT_volumes, _modalities = [], []
            CT_volumes, cls, patient, kidney_side = b
            patients.append(patient)
            kidneys_side.append(kidney_side)

            # N, A, V, D
            for modality in args.modalities:
                if modality in CT_volumes:
                    CT_volume = CT_volumes[modality]
                    _CT_volumes.append(CT_volume)
                    _modalities.append(modality)
                    if num_modalities == 1:
                        patient_CT_labels.append(cls)
                else:
                    if num_modalities > 1:
                        CT_volume = torch.full((1, args.crop_slices, args.cropsize_3d, args.cropsize_3d), float('nan'))
                        _CT_volumes.append(CT_volume)

            # 3d / 4d label
            if num_modalities > 1:
                patient_CT_labels.append(cls)
            # 3d ct ===> 4d ct, if necessary

            _CT_volumes = torch.stack(_CT_volumes)
            patient_CT_volumes.append(_CT_volumes)
            patient_modalities.append(_modalities)

        # stack batch info
        if num_modalities > 1:
            patient_CT_volumes = torch.stack(patient_CT_volumes)
        else:
            patient_CT_volumes = torch.cat(patient_CT_volumes, 0)
        patient_modalities = np.array(patient_modalities, dtype=object)

        patients = np.array(patients)
        kidneys_side = np.array(kidneys_side)

        if 'survival' in args.downstream_task:
            patient_CT_labels = check_survival_label_form(patient_CT_labels)
        else:
            patient_CT_labels = check_label_form(patient_CT_labels)

        return_dict = {
            'imgs': patient_CT_volumes,
            'labels': patient_CT_labels,
            'exist_modalities': patient_modalities,
            'patients': patients,
            'kidneys_side': kidneys_side,
        }
        return return_dict
    return collate_fn


def custom_collate_fn_CLIP(args, split='train'):
    # 'A_L', 'N_L', 'A_R', 'N_R' ===> 'L' & 'R'
    # 'A', 'N' ===>

    def collate_fn(batch):

        patient_modalities, patients_l, patients_r = [], [], []
        patient_l_CT_volumes, patient_r_CT_volumes = [], []
        patient_l_kidney_llm2vec_features, patient_r_kidney_llm2vec_features = [], []

        patient_raw_reports = {
            'left_kidney_findings': [],
            'left_kidney_impression': [],
            'right_kidney_findings': [],
            'right_kidney_impression': [],
        }

        for b in batch:
            l_CT_volumes, r_CT_volumes, _modalities = [], [], []
            CT_volumes, raw_report, llm2vec_features, patient = b
            patients_l.append(patient + '_L')
            patients_r.append(patient + '_R')
            
            patient_raw_reports['left_kidney_findings'].append(raw_report['left_kidney_findings'])
            patient_raw_reports['left_kidney_impression'].append(raw_report['left_kidney_impression'])
            patient_raw_reports['right_kidney_findings'].append(raw_report['right_kidney_findings'])
            patient_raw_reports['right_kidney_impression'].append(raw_report['right_kidney_impression'])

            if args.llm2vec:
                patient_l_kidney_llm2vec_features.append(llm2vec_features['left_kidney_llm2vec_features'])
                patient_r_kidney_llm2vec_features.append(llm2vec_features['right_kidney_llm2vec_features'])

            # N, A, V, D
            # for modality in args.modalities:
            for modality in ('img', ):
                l_modality, r_modality = modality + "_L", modality + '_R'
                if l_modality in CT_volumes:
                    l_CT_volume, r_CT_volume = CT_volumes[l_modality], CT_volumes[r_modality]
                    l_CT_volumes.append(l_CT_volume)
                    r_CT_volumes.append(r_CT_volume)
                    _modalities.append(modality)

            l_CT_volumes = torch.stack(l_CT_volumes)
            r_CT_volumes = torch.stack(r_CT_volumes)

            patient_l_CT_volumes.append(l_CT_volumes)
            patient_r_CT_volumes.append(r_CT_volumes)
            patient_modalities.append(_modalities)

        # stack batch info
        patient_modalities = np.array(patient_modalities, dtype=object)
        patients_l = np.array(patients_l)
        patients_r = np.array(patients_r)
        
        if args.llm2vec:
            patient_l_kidney_llm2vec_features = torch.tensor(np.array(patient_l_kidney_llm2vec_features)).squeeze()
            patient_r_kidney_llm2vec_features = torch.tensor(np.array(patient_r_kidney_llm2vec_features)).squeeze()

        patient_l_CT_volumes = torch.cat(patient_l_CT_volumes, 0)
        patient_r_CT_volumes = torch.cat(patient_r_CT_volumes, 0)

        return_dict = {
            'left_imgs': patient_l_CT_volumes,
            'right_imgs': patient_r_CT_volumes,
            "left_llm2vec_features": patient_l_kidney_llm2vec_features,
            "right_llm2vec_features": patient_r_kidney_llm2vec_features,

            'exist_modalities': patient_modalities,
            "raw_reports": patient_raw_reports,
            'l_patient_ID': patients_l,
            'r_patient_ID': patients_r,
        }
        
        return return_dict
    
    return collate_fn