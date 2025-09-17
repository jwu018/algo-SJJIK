import torch
from torch.utils.data import Dataset, DataLoader
import mne
import io
import s3fs

# -------------------------------------
# S3 filesystem
# -------------------------------------
fs = s3fs.S3FileSystem(anon=False)

# -------------------------------------
# EEG Dataset for SSL pretraining
# -------------------------------------
class EEGDatasetS3(Dataset):
    def __init__(self, s3_paths, window_size=16, sfreq=250, overlap=0.25, transform=None):
        """
        Args:
            s3_paths (list): list of full S3 paths to EDF files
            window_size (int): in seconds
            sfreq (int): sampling frequency
            overlap (float): fractional overlap (0.25 = 25%)
            transform: optional transform (Augmenter)
        """
        self.s3_paths = s3_paths
        self.window_size = window_size
        self.sfreq = sfreq
        self.overlap = overlap
        self.transform = transform

        self.samples = []  # [(file_idx, start_sample, stop_sample), ...]

        win_len = int(window_size * sfreq)
        step = int(win_len * (1 - overlap))

        # Precompute all window indices
        for file_idx, path in enumerate(s3_paths):
            with fs.open(path, 'rb') as f:
                raw = mne.io.read_raw_edf(io.BytesIO(f.read()), preload=False)
                n_samples = raw.n_times

            for start in range(0, n_samples - win_len, step):
                stop = start + win_len
                self.samples.append((file_idx, start, stop))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start, stop = self.samples[idx]
        s3_path = self.s3_paths[file_idx]

        # Load EDF from S3
        with fs.open(s3_path, 'rb') as f:
            raw = mne.io.read_raw_edf(io.BytesIO(f.read()), preload=True)
            data, _ = raw[:, start:stop]  # shape (C, T)

        x = torch.tensor(data, dtype=torch.float32)

        # Apply Augmenter (expects [B, C, T])
        if self.transform:
            x = self.transform(x.unsqueeze(0)).squeeze(0)

        return x  # returns (C, T)

# -------------------------------------
# Helper to build DataLoaders
# -------------------------------------
def get_ssl_dataloaders(bucket_name, window_size=16, sfreq=250, overlap=0.25, batch_size=64, augmenter=None):
    # List S3 files
    train_files = fs.ls(f"{bucket_name}/train")
    val_files   = fs.ls(f"{bucket_name}/val")
    test_files  = fs.ls(f"{bucket_name}/test")

    # Create Datasets
    train_dataset = EEGDatasetS3(train_files, window_size, sfreq, overlap, transform=augmenter)
    val_dataset   = EEGDatasetS3(val_files,   window_size, sfreq, overlap, transform=None)  # usually no augmentation for val
    test_dataset  = EEGDatasetS3(test_files,  window_size, sfreq, overlap, transform=None)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader



