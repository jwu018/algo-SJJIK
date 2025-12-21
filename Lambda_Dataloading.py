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

# Lambda Cloud filesystem paths
FILESYSTEM_NAME = "JJIK-EEG"  # TODO: Update this!

root_folder = f"/lambda/nfs/{FILESYSTEM_NAME}/tuh_eeg_data"
destination = f"/lambda/nfs/{FILESYSTEM_NAME}/eeg_collected"
ft_root = f"/lambda/nfs/{FILESYSTEM_NAME}/finetune_data"
ft_flat = f"/lambda/nfs/{FILESYSTEM_NAME}/finetune_collected"

# Create necessary directories
os.makedirs(destination, exist_ok=True)
os.makedirs(ft_flat, exist_ok=True)

# Function to collect EDF files from any directory structure
def collect_files(source_dir, dest_dir, file_extension=".edf"):
    """
    Recursively collect files with specified extension from source directory
    and copy them to a flat destination directory.
    """
    collected_count = 0
    
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} does not exist")
        return collected_count
    
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.lower().endswith(file_extension):
                full_path = os.path.join(dirpath, filename)
                dest_path = os.path.join(dest_dir, filename)
                
                # Handle duplicate filenames
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                        counter += 1
                
                try:
                    shutil.copy2(full_path, dest_path)
                    print(f"Copied {full_path} -> {dest_path}")
                    collected_count += 1
                except Exception as e:
                    print(f"Error copying {full_path}: {e}")
    
    return collected_count

# Collect EDF files for main training
#print("Collecting EDF files for training...")
#Commented out since collected
# edf_count = collect_files(root_folder, destination, ".edf")
# print(f"Collected {edf_count} EDF files")

# seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# partition data
freq = 250
window = 16
overlap = 0.25
batchsize = 64
workers = 0
data_path = destination

# read EEGs
def loadEEG(path, return_label=False):
    """
    Loads EDF files. Assumes path contains 'pd' if it's a Parkinson's case.
    """
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
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
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# Channel standardization
def transformEEG(EEG):
    n_channels = EEG.shape[0]
    Target_Chans = 61
    
    if n_channels >= Target_Chans:
        EEG = EEG[:Target_Chans, :]
    else:
        # If we have fewer than 61 channels, pad with zeros
        padded_EEG = np.zeros((Target_Chans, EEG.shape[1]))
        padded_EEG[:n_channels, :] = EEG
        EEG = padded_EEG
    
    return EEG

# Utility: deterministically select subset of files from a flat folder
def pick_subset_files(folder, ext=".edf", fraction=0.25, seed=42):
    """Select a fraction of files from folder"""
    files = [f for f in os.listdir(folder) if f.lower().endswith(ext)]
    files.sort()
    if len(files) == 0:
        return []
    rng = np.random.default_rng(seed)
    k = max(1, int(len(files) * fraction))
    idx = rng.choice(len(files), size=k, replace=False)
    idx.sort()
    return [files[i] for i in idx]

# Number of partitions
# Use only a subset of the EDF files already collected
subset_files = pick_subset_files(data_path, ext=".edf", fraction=0.05, seed=seed)

if len(subset_files) == 0:
    print("No EDF files found. Exiting.")
    exit(1)

print(f"Processing {len(subset_files)} files for partition counting...")
print("This may take several minutes depending on file sizes...")
num_partitions = dl.get_eeg_partition_number(
    data_path,
    freq,
    window,
    overlap,
    file_format=subset_files,
    load_function=loadEEG,
    optional_load_fun_args=[False],
    transform_function=transformEEG
)
print(f"Number of partitions available: {num_partitions}")

num_partitions.head()

# Split data
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

# Check what EEGsplit actually contains
print("EEGsplit type:", type(EEGsplit))
print("EEGsplit shape:", EEGsplit.shape if hasattr(EEGsplit, 'shape') else "No shape")
print("EEGsplit columns:", EEGsplit.columns.tolist() if hasattr(EEGsplit, 'columns') else "No columns")
print("EEGsplit head:")
print(EEGsplit.head() if hasattr(EEGsplit, 'head') else EEGsplit)

# Create Training Dataset
train_dataset = dl.EEGDataset(
    num_partitions,
    EEGsplit,
    [freq, window, overlap],
    mode='train',
    load_function=loadEEG,
    transform_function=transformEEG
)

# Create Validation Dataset
val_dataset = dl.EEGDataset(
    num_partitions,
    EEGsplit,
    [freq, window, overlap],
    mode='validation',
    load_function=loadEEG,
    transform_function=transformEEG
)

# Create samplers
train_sampler = dl.EEGSampler(train_dataset, Mode=0)
val_sampler = dl.EEGSampler(val_dataset, Mode=0)

# Create the dataloader
# Note: num_workers=0 is safer but slower. On Lambda Cloud, you might increase this (e.g., 4 or 8)
# if you don't encounter multiprocessing errors with MNE/pickling.
train_Dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batchsize,
    sampler=train_sampler,
    num_workers=workers
)

val_Dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batchsize,
    sampler=val_sampler,
    num_workers=workers
)

# FINETUNING CODE
#Test later
# print("\nCollecting BDF files for fine-tuning...")
# bdf_count = collect_files(ft_root, ft_flat, ".bdf")
# print(f"Collected {bdf_count} BDF files")

data_pathFT = ft_flat

# Extract files for fine-tuning
filesFT = [f for f in os.listdir(data_pathFT) if f.lower().endswith(".bdf")]
filesFT.sort()

if len(filesFT) == 0:
    print("No BDF files found for fine-tuning. Skipping fine-tuning setup.")
else:
    # BDF loader for FT data
    def loadEEG_FT(path, return_label=False):
        raw = mne.io.read_raw_bdf(path, preload=True)
        data = raw.get_data()
        label = 1 if 'pd' in os.path.basename(path).lower() else 0
        return (data, label) if return_label else data

    EEGlenFT = dl.get_eeg_partition_number(
        data_pathFT,
        freq,
        window,
        overlap,
        file_format=filesFT,
        load_function=loadEEG_FT,
        optional_load_fun_args=[False],
        transform_function=transformEEG
    )
    EEGlenFT = EEGlenFT.reset_index().drop(columns=['index'])

    def extract_labels_from_files(file_list, data_path):
        labels = []
        for file in file_list:
            full_path = os.path.join(data_path, file)
            try:
                _, label = loadEEG_FT(full_path, return_label=True)
                labels.append(label)
            except Exception as e:
                print(f"Error extracting label from {file}: {e}")
                labels.append(0)
        return np.array(labels)

    # Extract labels for the finetuning files
    labels = extract_labels_from_files(filesFT, data_pathFT)

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

    # TRAINING DATALOADER
    trainsetFT = dl.EEGDataset(
        EEGlenFT, EEGsplitFT, [freq, window, overlap], 'train', supervised=True,
        label_on_load=True, load_function=loadEEG_FT, optional_load_fun_args=[True], 
        transform_function=transformEEG
    )
    trainsamplerFT = dl.EEGSampler(trainsetFT, batchsize, workers)
    trainloaderFT = DataLoader(
        dataset=trainsetFT, batch_size=batchsize, sampler=trainsamplerFT, num_workers=workers
    )

    # VALIDATION DATALOADER
    valsetFT = dl.EEGDataset(
        EEGlenFT, EEGsplitFT, [freq, window, overlap], 'validation', supervised=True,
        label_on_load=True, load_function=loadEEG_FT, optional_load_fun_args=[True], 
        transform_function=transformEEG
    )
    valloaderFT = DataLoader(
        dataset=valsetFT, batch_size=batchsize, num_workers=workers, shuffle=False
    )

    # TEST DATALOADER
    testsetFT = dl.EEGDataset(
        EEGlenFT, EEGsplitFT, [freq, window, overlap], 'test', supervised=True,
        label_on_load=True, load_function=loadEEG_FT, optional_load_fun_args=[True], 
        transform_function=transformEEG
    )
    testloaderFT = DataLoader(
        dataset=testsetFT, batch_size=batchsize, shuffle=False
    )

    dl.check_split(EEGlenFT, EEGsplitFT, labels)
    
print("\nData loading setup complete!")

