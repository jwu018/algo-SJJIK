# imports
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from selfeeg import dataloading as dl
import mne

# seed
seed = 12
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# partition data
freq = 256
window = 1
overlap = 0.15
data_path = r"C:\Users\inesh\Downloads\EEG Data\flat_bdf_files"

# read EEGs
def loadEEG(path, return_label=False):
    raw = mne.io.read_raw_bdf(path, preload=True)
    data = raw.get_data()
    
    # Extract label from filename
    if 'pd' in os.path.basename(path).lower():
        label = 1
    else:
        label = 0
    
    if return_label:
        return data, label
    else:
        return data

# idk
def transformEEG(EEG):
    EEG = EEG[:,:-64]
    return EEG

# ​​ Number of partitions

num_partitions = dl.get_eeg_partition_number(
    data_path,
    freq,
    window,
    overlap,
    file_format='*.bdf',
    load_function=loadEEG,
    optional_load_fun_args=[False],
    transform_function=transformEEG
)
print(f"Number of partitions available: {num_partitions}")

num_partitions.head()

# ​​ Split data
EEGsplit = dl.get_eeg_split_table(
    num_partitions,
    test_ratio=0.1,
    val_ratio=0.1,
    test_split_mode='file',
    val_split_mode='file',
    exclude_data_id=None,
    stratified=False,
    perseverance = 5000,
    split_tolerance = 0.005,
    seed = seed
)
dl.check_split(num_partitions, EEGsplit)

# Check what EEGsplit actually contains
print("EEGsplit type:", type(EEGsplit))
print("EEGsplit shape:", EEGsplit.shape if hasattr(EEGsplit, 'shape') else "No shape")
print("EEGsplit columns:", EEGsplit.columns.tolist() if hasattr(EEGsplit, 'columns') else "No columns")
print("EEGsplit head:")
print(EEGsplit.head() if hasattr(EEGsplit, 'head') else EEGsplit)

# ​​ Create Dataset
dataset = dl.EEGDataset(
    num_partitions,
    EEGsplit,
    [freq, window, overlap],
    mode = 'train',
    load_function = loadEEG,
    transform_function = transformEEG
)
# get first sample 
sample_1 = dataset[0]
print(sample_1.shape)

# create a basic sampler
sampler_linear = dl.EEGSampler(dataset, Mode=0)

# create the dataloader
Final_Dataloader = DataLoader(
    dataset = dataset,
    batch_size = 16,
    sampler = sampler_linear,
    num_workers = 0
)

# test
for X in Final_Dataloader:
    print(X.shape)
    break
