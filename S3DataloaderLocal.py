import os
import random
import numpy as np
import torch
import selfeeg.dataloading as dl
import mne

# -----------------------------
# Local data paths
# -----------------------------
train_folder = r"../000/train"
val_folder   = r"../000/val"
test_folder  = r"../000/test"

# List all EDF files in each split
def list_edf_files(folder):
    files = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith(".edf"):
                files.append(os.path.join(dirpath, f))
    return files

train_files = list_edf_files(train_folder)
val_files   = list_edf_files(val_folder)
test_files  = list_edf_files(test_folder)

# Combine all files for num_partitions table
all_files = train_files + val_files + test_files

# -----------------------------
# Seed
# -----------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# -----------------------------
# Parameters
# -----------------------------
freq = 250
window = 1       # seconds
overlap = 0.15
batchsize = 64
workers = 0

# -----------------------------
# EDF loader (local)
# -----------------------------
def loadEEG_local(path, return_label=False):
    raw = mne.io.read_raw_edf(path, preload=True)
    data = raw.get_data()
    if return_label:
        label = 1 if 'pd' in os.path.basename(path).lower() else 0
        return data, label
    return data

# -----------------------------
# Transform function (61 channels)
# -----------------------------
def transformEEG(EEG):
    n_channels = EEG.shape[0]
    if n_channels >= 61:
        EEG = EEG[:61, :]
    else:
        padded_EEG = np.zeros((61, EEG.shape[1]))
        padded_EEG[:n_channels, :] = EEG
        EEG = padded_EEG
    return EEG

# -----------------------------
# Partition table
# -----------------------------
num_partitions = dl.get_eeg_partition_number(
    all_files,
    freq,
    window,
    overlap,
    file_format='*.edf',
    load_function=loadEEG_local,
    optional_load_fun_args=[False],
    transform_function=transformEEG
)
print(f"Number of partitions available: {num_partitions}")

# -----------------------------
# Split data table
# -----------------------------
EEGsplit = dl.get_eeg_split_table(
    num_partitions,
    test_ratio=0.1,
    val_ratio=0.1,
    test_split_mode='file',
    val_split_mode='file',
    exclude_data_id=None,
    stratified=False,
    perseverance=5000,
    split_tolerance=0.005,
    seed=seed
)
dl.check_split(num_partitions, EEGsplit)

# -----------------------------
# Create Datasets
# -----------------------------
train_dataset = dl.EEGDataset(
    num_partitions,
    EEGsplit,
    [freq, window, overlap],
    mode='train',
    load_function=loadEEG_local,
    transform_function=transformEEG
)

val_dataset = dl.EEGDataset(
    num_partitions,
    EEGsplit,
    [freq, window, overlap],
    mode='validation',
    load_function=loadEEG_local,
    transform_function=transformEEG
)

# -----------------------------
# Samplers & DataLoaders
# -----------------------------
train_sampler = dl.EEGSampler(train_dataset, Mode=0)
val_sampler   = dl.EEGSampler(val_dataset, Mode=0)

train_Dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batchsize,
    sampler=train_sampler,
    num_workers=workers
)

val_Dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batchsize,
    sampler=val_sampler,
    num_workers=workers
)

# -----------------------------
# Fine-tuning subset
# -----------------------------
filesFT = EEGsplit.loc[EEGsplit['split_set'] == 0, 'file_name'].values
EEGlenFT = num_partitions.loc[num_partitions['file_name'].isin(filesFT)]
EEGlenFT = EEGlenFT.reset_index().drop(columns=['index'])

def extract_labels_from_files(file_list):
    labels = []
    for file in file_list:
        try:
            _, label = loadEEG_local(file, return_label=True)
            labels.append(label)
        except Exception as e:
            labels.append(0)
    return np.array(labels)

labels = extract_labels_from_files(filesFT)

EEGsplitFT = dl.get_eeg_split_table(
    partition_table=EEGlenFT,
    test_ratio=0.2,
    val_ratio=0.1,
    val_ratio_on_all_data=False,
    stratified=True,
    labels=labels,
    split_tolerance=0.001,
    perseverance=10000,
    seed=seed
)

# FT datasets
trainsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap],
    'train', supervised=True, label_on_load=True,
    load_function=loadEEG_local, optional_load_fun_args=[True]
)
trainsamplerFT = dl.EEGSampler(trainsetFT, batchsize, workers)
trainloaderFT = torch.utils.data.DataLoader(
    dataset=trainsetFT, batch_size=batchsize,
    sampler=trainsamplerFT, num_workers=workers
)

valsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap],
    'validation', supervised=True, label_on_load=True,
    load_function=loadEEG_local, optional_load_fun_args=[True]
)
valloaderFT = torch.utils.data.DataLoader(
    dataset=valsetFT, batch_size=batchsize, num_workers=workers,
    shuffle=False
)

testsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap],
    'test', supervised=True, label_on_load=True,
    load_function=loadEEG_local, optional_load_fun_args=[True]
)
testloaderFT = torch.utils.data.DataLoader(
    dataset=testsetFT, batch_size=batchsize, shuffle=False
)

dl.check_split(EEGlenFT, EEGsplitFT, labels)
