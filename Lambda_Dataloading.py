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
from training import loadEEG as loadEEGFT
import split
import pickle
import re
from typing import Optional
from scipy.signal import butter, filtfilt
from scipy.stats import zscore


# Lambda Cloud filesystem paths
FILESYSTEM_NAME = "Algoverse"  # TODO: Update this!

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

def bandpass_and_zscore(data, fs, low=0.5, high=40, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    if data.shape[1] >= 3 * max(len(a), len(b)):
        data = filtfilt(b, a, data, axis=1)
    data = np.nan_to_num(data)
    data = zscore(data, axis=1)
    return data


# read EEGs
def loadEEG(path, return_label=False):
    """
    Loads EDF files. Assumes path contains 'pd' if it's a Parkinson's case.
    """
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        data = raw.get_data()
        fs = int(raw.info['sfreq'])
        data = bandpass_and_zscore(data, fs)
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
subset_files = pick_subset_files(data_path, ext=".edf", fraction=1.0, seed=seed)

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

#data_pathFT = ft_flat

# Extract files for fine-tuning
# filesFT = [f for f in os.listdir(data_pathFT) if f.lower().endswith(".bdf")]
# filesFT.sort()

# if len(filesFT) == 0:
#     print("No BDF files found for fine-tuning. Skipping fine-tuning setup.")
# else:
#     # BDF loader for FT data
#     def loadEEG_FT(path, return_label=False):
#         raw = mne.io.read_raw_bdf(path, preload=True)
#         data = raw.get_data()
#         label = 1 if 'pd' in os.path.basename(path).lower() else 0
#         return (data, label) if return_label else data
#
#     EEGlenFT = dl.get_eeg_partition_number(
#         data_pathFT,
#         freq,
#         window,
#         overlap,
#         file_format=filesFT,
#         load_function=loadEEG_FT,
#         optional_load_fun_args=[False],
#         transform_function=transformEEG
#     )
#     EEGlenFT = EEGlenFT.reset_index().drop(columns=['index'])
#
#     def extract_labels_from_files(file_list, data_path):
#         labels = []
#         for file in file_list:
#             full_path = os.path.join(data_path, file)
#             try:
#                 _, label = loadEEG_FT(full_path, return_label=True)
#                 labels.append(label)
#             except Exception as e:
#                 print(f"Error extracting label from {file}: {e}")
#                 labels.append(0)
#         return np.array(labels)
#
#     # Extract labels for the finetuning files
#     labels = extract_labels_from_files(filesFT, data_pathFT)
#
#     EEGsplitFT = dl.get_eeg_split_table(
#         partition_table=EEGlenFT,
#         test_ratio=0.2,
#         val_ratio=0.1,
#         val_ratio_on_all_data=False,
#         stratified=True,
#         labels=labels,
#         split_tolerance=0.001,
#         perseverance=10000,
#         seed=seed
#     )
#
#     # TRAINING DATALOADER
#     trainsetFT = dl.EEGDataset(
#         EEGlenFT, EEGsplitFT, [freq, window, overlap], 'train', supervised=True,
#         label_on_load=True, load_function=loadEEG_FT, optional_load_fun_args=[True],
#         transform_function=transformEEG
#     )
#     trainsamplerFT = dl.EEGSampler(trainsetFT, batchsize, workers)
#     trainloaderFT = DataLoader(
#         dataset=trainsetFT, batch_size=batchsize, sampler=trainsamplerFT, num_workers=workers
#     )
#
#     # VALIDATION DATALOADER
#     valsetFT = dl.EEGDataset(
#         EEGlenFT, EEGsplitFT, [freq, window, overlap], 'validation', supervised=True,
#         label_on_load=True, load_function=loadEEG_FT, optional_load_fun_args=[True],
#         transform_function=transformEEG
#     )
#     valloaderFT = DataLoader(
#         dataset=valsetFT, batch_size=batchsize, num_workers=workers, shuffle=False
#     )
#
#     # TEST DATALOADER
#     testsetFT = dl.EEGDataset(
#         EEGlenFT, EEGsplitFT, [freq, window, overlap], 'test', supervised=True,
#         label_on_load=True, load_function=loadEEG_FT, optional_load_fun_args=[True],
#         transform_function=transformEEG
#     )
#     testloaderFT = DataLoader(
#         dataset=testsetFT, batch_size=batchsize, shuffle=False
#     )
#
#     dl.check_split(EEGlenFT, EEGsplitFT, labels)

# ==================================
#  loading and renaming data
# ==================================



# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = [
    {
        "name": "3-Stim",
        "root_dir": f"/lambda/nfs/{FILESYSTEM_NAME}/finetuning_datasets/ds003490",
        "output_dir": f"/lambda/nfs/{FILESYSTEM_NAME}/finetune_collected",
        "dataset_id": 5,
    },
    {
        "name": "UCSD",
        "root_dir": f"/lambda/nfs/{FILESYSTEM_NAME}/finetuning_datasets/ds002778",
        "output_dir": f"/lambda/nfs/{FILESYSTEM_NAME}/finetune_collected",
        "dataset_id": 8,
    },
    {
        "name": "Test-Retest",
        "root_dir": f"/lambda/nfs/{FILESYSTEM_NAME}/finetuning_datasets/ds004148",
        "output_dir": f"/lambda/nfs/{FILESYSTEM_NAME}/finetune_collected",
        "dataset_id": 2,
    },
    {
        "name": "PD EO",
        "root_dir": f"/lambda/nfs/{FILESYSTEM_NAME}/finetuning_datasets/ds004584",
        "output_dir": f"/lambda/nfs/{FILESYSTEM_NAME}/finetune_collected",
        "dataset_id": 19,
    },
]

DRY_RUN = True   # set True to test without writing files

SUPPORTED_EXTS = (".bdf", ".bdt", ".set", ".vhdr")

# ============================================================
# REGEX DEFINITIONS
# ============================================================

SUBJECT_RE = re.compile(r"sub-[a-z]*?(\d+)", re.IGNORECASE)
# supports: ses-01  |  ses-session1
SESSION_RE = re.compile(r"ses-(?:session)?(\d+)", re.IGNORECASE)
TRIAL_RE   = re.compile(r"trial[-_]?(\d+)", re.IGNORECASE)

# ============================================================
# HELPERS
# ============================================================

def extract_or_default(regex, text, default: Optional[int]):
    match = regex.search(text)
    return int(match.group(1)) if match else default

def extract_group(text: str) -> Optional[str]:
    text = text.lower()
    if "sub-hc" in text:
        return "hc"
    if "sub-pd" in text:
        return "pd"
    return None

def load_raw_any(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".bdf", ".bdt"]:
        return mne.io.read_raw_bdf(path, preload=True, verbose=False)

    elif ext == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose=False)

    elif ext == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=True, verbose=False)

    else:
        raise ValueError(f"Unsupported EEG format: {ext}")

# ============================================================
# CORE CONVERSION FUNCTION
# ============================================================

def convert_dataset(dataset_cfg):
    root_dir   = dataset_cfg["root_dir"]
    output_dir = dataset_cfg["output_dir"]
    dataset_id = dataset_cfg["dataset_id"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Converting dataset {dataset_cfg['name']} (ID={dataset_id}) ===")

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:

            if not fname.lower().endswith(SUPPORTED_EXTS):
                continue

            # Only load BrainVision via .vhdr
            if fname.lower().endswith((".eeg", ".vmrk")):
                continue

            full_path = os.path.join(dirpath, fname)
            name = fname.lower()

            subject_id = extract_or_default(SUBJECT_RE, name, None)
            if subject_id is None:
                print(f"[SKIP] No subject ID → {full_path}")
                continue

            session_id = extract_or_default(SESSION_RE, name, 1)
            trial_id = extract_or_default(TRIAL_RE, name, 1)
            group = extract_group(name)

            try:
                raw = load_raw_any(full_path)
            except Exception as e:
                print(f"[ERROR] Failed to load {full_path}: {e}")
                continue

            data = raw.get_data()
            sfreq = int(raw.info["sfreq"])
            ch_names = raw.info["ch_names"]

            out = {
                "data": data,
                "sfreq": sfreq,
                "ch_names": ch_names,
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "session_id": session_id,
                "trial_id": trial_id,
                "group": group,
                "source_format": os.path.splitext(fname)[1].lower(),
            }

            out_name = f"{dataset_id}_{subject_id}_{session_id}_{trial_id}.pickle"
            out_path = os.path.join(output_dir, out_name)

            if DRY_RUN:
                print(f"[DRY RUN] {out_name}")
            else:
                with open(out_path, "wb") as f:
                    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"✓ {out_name}")

for ds in DATASETS:
    convert_dataset(ds)

print("\n✓ All datasets processed")

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


#Paper's finetuning code - to be tested
# ==================================
#  Section 3: create partition list
# ==================================
#ds003490 - ID 5 - 3Stim
ctl_id_5 = [i for i in range(28,51)] + [3,5]
pds_id_5 = [i for i in range(6,28)]  + [1,2,4]
part_c = split.create_nested_kfold_subject_split(ctl_id_5, 10, 5)
part_p = split.create_nested_kfold_subject_split(pds_id_5, 10, 5)
partition_list_1 = split.merge_partition_lists(part_c, part_p, 10, 5)

#ds002778 - ID 8 - UCSD
ctl_id_8 = [1, 2, 4, 7,  8, 10, 17, 19, 20, 23, 24, 27, 28, 29, 30, 31]
pds_id_8 = [3, 5, 6, 9, 11, 12, 13, 14, 15, 16, 18, 21, 22, 25, 26]

ctl_id_8 = [i for i in range(1, 17)]
pds_id_8 = [i for i in range(17,32)]

part_c = split.create_nested_kfold_subject_split(ctl_id_8, 10, 5)
part_p = split.create_nested_kfold_subject_split(pds_id_8, 10, 5)
partition_list_2 = split.merge_partition_lists(part_c, part_p, 10, 5)

#ds004148
partition_list_3 = split.create_nested_kfold_subject_split(60,10,5)

#ds004584
pds_id_19 = [i for i in range(1, 101)]
ctl_id_19 = [i for i in range(101, 141)]
part_c = split.create_nested_kfold_subject_split(ctl_id_19, 10, 5)
part_p = split.create_nested_kfold_subject_split(pds_id_19, 10, 5)
partition_list_4 = split.merge_partition_lists(part_c, part_p, 10, 5)
# =============================================================================
# FINE-TUNING DATA
# =============================================================================
print_section("FINE-TUNING DATA SETUP")


data_pathFT = ft_flat

# prepare loadEEG arguments as the scripts do:
loadEEG_args = {
    'return_label': True,
    'downsample': False,
    #'use_only_original': False,
    'apply_zscore': True,
    #TODO: corect Args?
}

# Set the Dataset ID for glob.glob operation in SelfEEG's GetEEGPartitionNumber().
# It is a single number for every dataset
datasetID_1 = '5'  # EEG 3-Stim
datasetID_2 = '8'  # UC SD
datasetID_3 = '2'  # Test_Retest_Rest
datasetID_4 = '19' # PD_EO

glob_input = [
    datasetID_1 + '_*.pickle',   # only off medication
    datasetID_2 + '_*.pickle',   # only off medication
    datasetID_3 + '_*.pickle',   # only eyes open session 1.
    datasetID_4 + '_*.pickle',   # datasetID_4 have only eyes open
]

EEGlenFT = dl.get_eeg_partition_number(
        data_pathFT, freq, window, overlap,
        file_format             = glob_input,
        load_function           = loadEEGFT,
        optional_load_fun_args  = loadEEG_args,
        includePartial          = False if overlap == 0 else True,
        verbose                 = False
)

# Now we also need to load the labels
loadEEG_args['return_label'] = True

# Set functions to retrieve dataset, subject, and session from each filename.
# They will be used by GetEEGSplitTable to perform a subject based split
dataset_id_ex = lambda x: int(x.split(os.sep)[-1].split('_')[0])
subject_id_ex = lambda x: int(x.split(os.sep)[-1].split('_')[1])
session_id_ex = lambda x: int(x.split(os.sep)[-1].split('_')[2])

#Inner fold between 1 and 5
#Outer fold between 1 and 10
#Both have default 1 on paper code
outerFold = 1
innerFold = 1

# fold to eval is the correct index to get the desired train/val/test partition
foldToEval = outerFold*5 + innerFold

# Now call the GetEEGSplitTable. Since Parkinson task merges two datasets
# we need to differentiate between this and other tasks
# Remember: 5 = 3-Stim   &&   8 = UCSD
train_id = {
    5: partition_list_1[foldToEval][0],
    8: partition_list_2[foldToEval][0],
    2: partition_list_3[foldToEval][0],
    19: partition_list_4[foldToEval][0],
}
val_id = {
    5: partition_list_1[foldToEval][1],
    8: partition_list_2[foldToEval][1],
    2: partition_list_3[foldToEval][1],
    19: partition_list_4[foldToEval][1],
}
test_id = {
    5: partition_list_1[foldToEval][2],
    8: partition_list_2[foldToEval][2],
    2: partition_list_3[foldToEval][2],
    19: partition_list_4[foldToEval][2],
}
EEGsplitFT = dl.get_eeg_split_table(
    partition_table=EEGlenFT,
    exclude_data_id=None,
    val_data_id=val_id,
    test_data_id=test_id,
    split_tolerance=0.001,
    dataset_id_extractor=dataset_id_ex,
    subject_id_extractor=subject_id_ex,
    perseverance=10000
)

# EEGlen and EEGsplit come from their split utilities (see RunSingleTraining)
trainsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap], 'train',
    supervised=True,
    label_on_load=True,
    load_function=loadEEGFT,
    optional_load_fun_args=loadEEG_args,
    transform_function=transformEEG
)
trainsetFT.preload_dataset()   # fills trainset.x_preload and trainset.y_preload


# A boolean that set if EEG data should be transformed
# with the common spatial pattern.
#Default in paper is false
# if csp:
#     flag_dir = "csp/"
#     _reset_seed_number(seed)
#     CSP = CSPScaler(Nfilters=Nfilters, device=device)
#     data1 = trainset.x_preload[trainset.y_preload == 0].detach().clone().numpy()
#     data2 = trainset.x_preload[trainset.y_preload == 1].detach().clone().numpy()
#     CSP.fit(data1, data2)
#     del data1, data2
#     Chan = Nfilters * 2
#     CSPval = copy.deepcopy(CSP)
#     CSPval._use_torch = False
#     CSPval.Wcsp = CSPval.Wcsp.detach().cpu().numpy()

valsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap], 'validation',
    supervised=True,
    label_on_load=True,
    load_function=loadEEGFT,
    optional_load_fun_args=loadEEG_args,
    transform_function=transformEEG
)
valsetFT.preload_dataset()

testsetFT = dl.EEGDataset(
    EEGlenFT, EEGsplitFT, [freq, window, overlap], 'test',
    supervised=True,
    label_on_load=True,
    load_function=loadEEGFT,
    optional_load_fun_args=loadEEG_args,
    transform_function=transformEEG
)
testsetFT.preload_dataset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainsetFT.x_preload = trainsetFT.x_preload.to(device=device)
trainsetFT.y_preload = trainsetFT.y_preload.to(device=device)
valsetFT.x_preload = valsetFT.x_preload.to(device=device)
valsetFT.y_preload = valsetFT.y_preload.to(device=device)
testsetFT.x_preload = testsetFT.x_preload.to(device=device)
testsetFT.y_preload = testsetFT.y_preload.to(device=device)

# Finally, Define Dataloaders
# (no need to use more workers in validation and test dataloaders)
trainloaderFT = DataLoader(
    dataset=trainsetFT,
    batch_size=batchsize,
    shuffle=True,
    num_workers=workers
)
valloaderFT = DataLoader(
    dataset=valsetFT,
    batch_size=batchsize,
    shuffle=False,
    num_workers=0
)
testloaderFT = DataLoader(
    dataset=testsetFT,
    batch_size=batchsize,
    shuffle=False,
    num_workers=0
)

print(f"✓ Fine-tuning dataloaders created")
print(f"  Training: {len(trainloaderFT)} batches")
print(f"  Validation: {len(valloaderFT)} batches")
print(f"  Test: {len(testloaderFT)} batches")




