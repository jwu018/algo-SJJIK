import torch
import torch.nn as nn
import torch.nn.functional as F


def trial_averaging(x, y):
    return (x+y)/2

def time_slice_mix(x, y, num_slices = 4):
    C, T = x.shape
    slice_len = T // num_slices
    x_new = x.clone()
    for i in range(num_slices):
        if torch.rand(1) < 0.5:
            start = i * slice_len
            end = T if i == num_slices-1 else (i+1) * slice_len
            x_new[:, start:end] = y[:, start:end]
    return x_new

def frequency_mix(x, y, alpha=0.5):
    X1 = torch.fft.fft(x, dim=-1)
    X2 = torch.fft.fft(y, dim=-1)
    X_new = alpha * X1 + (1 - alpha) * X2
    x_new = torch.fft.ifft(X_new, dim=-1).real
    return x_new

def random_crop(x, crop_ratio=0.8):
    C, T = x.shape
    crop_len = int(T * crop_ratio)
    start = torch.randint(0, T - crop_len + 1, (1,)).item()
    x_crop = x[:, start:start+crop_len]  # cropped segment

    # Resize back to original length
    x_resized = F.interpolate(x_crop.unsqueeze(0), size=T, mode='linear', align_corners=False)
    return x_resized.squeeze(0)


class Augmenter(nn.Module):
    def __init__(self, use_trial_avg=True, use_time_mix=True, use_freq_mix=True, use_crop=True, p=0.6):
        super().__init__()
        self.use_trial_avg = use_trial_avg
        self.use_time_mix = use_time_mix
        self.use_freq_mix = use_freq_mix
        self.use_crop = use_crop
        self.p = p

    def forward(self, eeg, eeg_alt=None):
        # eeg in [C, T] format from dataloader
        # eeg_alt needs to be extracted from main pipeline using a shuffled batch

        if self.use_trial_avg and eeg_alt != None and torch.rand(1) < self.p:
            eeg = trial_averaging(eeg, eeg_alt)

        if self.use_time_mix and eeg_alt is not None and torch.rand(1) < self.p:
            eeg = time_slice_mix(eeg, eeg_alt)

        if self.use_freq_mix and eeg_alt is not None and torch.rand(1) < self.p:
            eeg = frequency_mix(eeg, eeg_alt)

        if self.use_crop and torch.rand(1) < self.p:
            eeg = random_crop(eeg)

        return eeg