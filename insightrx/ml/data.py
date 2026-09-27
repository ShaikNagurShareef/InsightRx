"""mBRSET loading, label coding, frozen patient split and fundus transforms."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config as C
from .preprocess import MEAN, STD, FundusCrop, open_fundus  # noqa: F401  (re-exported: shared with the ONNX runtime)


def eval_transform(size):
    from torchvision import transforms as T
    return T.Compose([FundusCrop(), T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform(size):
    from torchvision import transforms as T
    return T.Compose([FundusCrop(),
                      T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05),
                                          interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.RandomRotation(10),
                      T.ColorJitter(0.15, 0.15, 0.1, 0.02), T.ToTensor(), T.Normalize(MEAN, STD)])


class FundusDataset(Dataset):
    def __init__(self, paths, labels, size, train):
        self.paths = list(paths)
        self.labels = labels.astype(np.float32)
        self.size = size
        self.tf = train_transform(size) if train else eval_transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(open_fundus(self.paths[i], self.size)), torch.from_numpy(self.labels[i]), i


# ------------------------------------------------------------------ labels
def _yes_no(s):
    s = s.astype(str).str.strip().str.lower()
    return s.map({'yes': 1.0, 'no': 0.0}).astype(float)


def load_tables():
    """-> images (one row per image), patients (one row per patient)."""
    df = pd.read_csv(C.MBRSET_LABELS)
    age = df['age'].astype(str).str.replace('&gt;', '>', regex=False)
    df['age_num'] = pd.to_numeric(age.str.replace('>=', '', regex=False).str.strip(), errors='coerce')
    icdr = pd.to_numeric(df['final_icdr'], errors='coerce')
    images = pd.DataFrame({
        'image_id': df['file'].astype(str),
        'patient_id': df['patient'].astype(int),
        'eye': df['laterality'].str.strip().str.lower().map({'right': 'OD', 'left': 'OS'}),
        'path': C.MBRSET_IMAGES + df['file'].astype(str),
        'quality_poor': 1.0 - _yes_no(df['final_quality']),
        'icdr': icdr,
        'dr_referable': np.where(icdr.isna(), np.nan, (icdr >= 2).astype(float)),
        'edema': _yes_no(df['final_edema']),
    })
    first = df.groupby('patient').first()
    patients = pd.DataFrame({'patient_id': first.index.astype(int),
                             'age': df.groupby('patient')['age_num'].median().values,
                             'sex': first['sex'].astype(float).values,           # 0 female / 1 male
                             'dm_time': pd.to_numeric(first['dm_time'], errors='coerce').values,
                             'insulin': first['insulin'].values,
                             'oraltreatment_dm': first['oraltreatment_dm'].values})
    for t in C.SYSTEMIC_TARGETS:
        patients[t] = first[t].astype(float).values
    for t in C.SYSTEMIC_TARGETS:     # image-level copies for the auxiliary heads
        images[t] = images['patient_id'].map(patients.set_index('patient_id')[t])
    return images, patients


def load_split():
    """Frozen patient-grouped 70/10/20 split (OculoMoE protocol P1), copied into the Insight Rx output dir."""
    C.ensure_dirs()
    dst = os.path.join(C.OUTPUT_DIR, 'splits.json')
    if not os.path.exists(dst):
        shutil.copy(C.P1_SPLIT, dst)
    s = json.load(open(dst))['splits']['0']
    split = {k: [int(p) for p in s[k]] for k in ('train', 'val', 'test')}
    if C.SMOKE:
        rng = np.random.RandomState(C.SEED)
        split = {k: sorted(rng.choice(v, min(len(v), C.SMOKE_PATIENTS), replace=False).tolist())
                 for k, v in split.items()}
    return split


def label_matrix(images):
    return np.stack([images[name].values.astype(np.float32) for name, *_ in C.HEADS], 1)


def label_audit(images, patients, split):
    out = {'n_images': int(len(images)), 'n_patients': int(len(patients)),
           'split_patients': {k: len(v) for k, v in split.items()},
           'images': {c: images[c].value_counts(dropna=False).rename(str).to_dict()
                      for c in ['eye', 'quality_poor', 'icdr', 'dr_referable', 'edema']},
           'patients': {c: patients[c].value_counts(dropna=False).rename(str).to_dict()
                        for c in C.SYSTEMIC_TARGETS + ['sex', 'insulin']},
           'patient_missing_fraction': {c: float(patients[c].isna().mean()) for c in patients.columns},
           'coding': {'quality_poor': "final_quality == 'no'", 'dr_referable': 'final_icdr >= 2',
                      'icdr': 'final_icdr 0-4', 'edema': "final_edema == 'yes'"}}
    with open(os.path.join(C.OUTPUT_DIR, 'label_audit.json'), 'w') as f:
        json.dump(out, f, indent=1, default=str)
    return out
