# imports
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import selfeeg
import selfeeg.dataloading as dl
import mne
import shutil

#takes out all the .edf files
root_folder = r"..\000"

destination = r"..\000_collected"

os.makedirs(destination, exist_ok=True)

for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        # Example: only copy certain file types
        if filename.endswith(".edf"):
            full_path = os.path.join(dirpath, filename)
            dest_path = os.path.join(destination, filename)

            # If you want to preserve uniqueness, handle duplicates
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(destination, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.copy2(full_path, dest_path)
            print(f"Copied {full_path} -> {dest_path}")


# seed
seed = 12
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# partition data
freq = 250
window = 1
overlap = 0.15
batchsize = 16
workers = 0
data_path = destination# data path here

# read EEGs
def loadEEG(path, return_label=False):
    raw = mne.io.read_raw_edf(path, preload=True)
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
    n_channels = EEG.shape[0]
    
    if n_channels >= 32:
        EEG = EEG[:32, :]
    else:
        # If we have fewer than 32 channels, pad with zeros
        padded_EEG = np.zeros((32, EEG.shape[1]))
        padded_EEG[:n_channels, :] = EEG
        EEG = padded_EEG
    
    return EEG

# ​​ Number of partitions

num_partitions = dl.get_eeg_partition_number(
    data_path,
    freq,
    window,
    overlap,
    file_format='*.edf',
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

# ​​ Create Training Dataset
train_dataset = dl.EEGDataset(
    num_partitions,
    EEGsplit,
    [freq, window, overlap],
    mode = 'train',
    load_function = loadEEG,
    transform_function = transformEEG
)

# Create Validation Dataset
val_dataset = dl.EEGDataset(
    num_partitions,
    EEGsplit,
    [freq, window, overlap],
    mode = 'validation',
    load_function = loadEEG,
    transform_function = transformEEG
)

# get first sample 
# train_sample_1 = train_dataset[0]
# print(train_sample_1.shape)

# val_sample_1 = val_dataset[0]
# print(val_sample_1.shape)

# create samplers
train_sampler = dl.EEGSampler(train_dataset, Mode=0)
val_sampler = dl.EEGSampler(val_dataset, Mode=0)

# create the dataloader
train_Dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 16,
    sampler = train_sampler,
    num_workers = 0
)

val_Dataloader = DataLoader(
    dataset = val_dataset,
    batch_size = 16,
    sampler = val_sampler,
    num_workers = 0
)

# # test
# for X in train_Dataloader:
#     print(X.shape)
#     break
#
# for X in val_Dataloader:
#     print(X.shape)
#     break
#

# FINETUNING CODE

# Extract a subset of training data for fine-tuning
filesFT = EEGsplit.loc[EEGsplit['split_set']==0, 'file_name'].values
EEGlenFT = num_partitions.loc[num_partitions['file_name'].isin(filesFT)]
EEGlenFT = EEGlenFT.reset_index().drop(columns=['index'])

def extract_labels_from_files(file_list, data_path):
    labels = []
    for file in file_list:
        full_path = os.path.join(data_path, file)
        try:
            _, label = loadEEG(full_path, return_label=True)
            labels.append(label)
        except Exception as e:
            labels.append(0)
    return np.array(labels)

# Extract labels for the finetuning files
labels = extract_labels_from_files(filesFT, data_path)

EEGsplitFT = dl.get_eeg_split_table(
    partition_table=EEGlenFT,
    test_ratio = 0.2,
    val_ratio= 0.1,
    val_ratio_on_all_data=False,
    stratified=True,
    labels=labels,
    split_tolerance=0.001,
    perseverance=10000,
    seed=seed
)

# TRAINING DATALOADER
trainsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap], 'train', supervised=True,
    label_on_load=True, load_function=loadEEG, optional_load_fun_args=[True]
)
trainsamplerFT = dl.EEGSampler(trainsetFT, batchsize, workers)
trainloaderFT = DataLoader(
    dataset = trainsetFT, batch_size= batchsize, sampler=trainsamplerFT, num_workers=workers)

# VALIDATION DATALOADER
valsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap], 'validation', supervised=True,
    label_on_load=True, load_function=loadEEG, optional_load_fun_args=[True]
)
valloaderFT = DataLoader(
    dataset=valsetFT, batch_size=batchsize, num_workers=workers, shuffle=False)

#TEST DATALOADER
testsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap], 'test', supervised=True,
    label_on_load=True, load_function=loadEEG, optional_load_fun_args=[True]
)
testloaderFT = DataLoader(dataset = testsetFT, batch_size= batchsize, shuffle=False)

dl.check_split(EEGlenFT, EEGsplitFT, labels)
