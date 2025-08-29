import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Augmentation functions (batch-aware)
# -----------------------------

def trial_averaging(x1, x2):
    # x1, x2: [B, C, T]
    return (x1 + x2) / 2

def time_slice_mix(x1, x2, num_slices=4):
    # x1, x2: [B, C, T]
    B, C, T = x1.shape
    slice_len = T // num_slices
    x_new = x1.clone()

    for i in range(num_slices):
        start = i * slice_len
        end = T if i == num_slices - 1 else (i + 1) * slice_len
        # random swap mask for each sample in batch
        swap_mask = (torch.rand(B, 1, 1, device=x1.device) < 0.5)
        x_new[:, :, start:end] = torch.where(
            swap_mask, x2[:, :, start:end], x_new[:, :, start:end]
        )
    return x_new

def frequency_mix(x1, x2, alpha=0.5):
    # x1, x2: [B, C, T]
    X1 = torch.fft.fft(x1, dim=-1)
    X2 = torch.fft.fft(x2, dim=-1)
    X_new = alpha * X1 + (1 - alpha) * X2
    x_new = torch.fft.ifft(X_new, dim=-1).real
    return x_new

def random_crop(x, crop_ratio=0.8):
    # x: [B, C, T]
    B, C, T = x.shape
    crop_len = int(T * crop_ratio)
    x_out = torch.zeros_like(x)

    for b in range(B):
        start = torch.randint(0, T - crop_len + 1, (1,)).item()
        segment = x[b, :, start:start+crop_len]
        # resize back to original T
        segment_resized = F.interpolate(
            segment.unsqueeze(0), size=T, mode="linear", align_corners=False
        )
        x_out[b] = segment_resized.squeeze(0)

    return x_out

# -----------------------------
# Augmenter class
# -----------------------------

class Augmenter(nn.Module):
    def __init__(self, use_trial_avg=True, use_time_mix=True, use_freq_mix=True, use_crop=True, p=0.5):
        super().__init__()
        self.use_trial_avg = use_trial_avg
        self.use_time_mix = use_time_mix
        self.use_freq_mix = use_freq_mix
        self.use_crop = use_crop
        self.p = p  # probability of applying each augmentation

    def forward(self, x, x_alt=None):
        """
        x: [B, C, T] batch of EEG trials
        x_alt: [B, C, T] optional second batch (same shape) for mix-based augmentations
        """
        # 1) Trial averaging
        if self.use_trial_avg and x_alt is not None and torch.rand(1) < self.p:
            x = trial_averaging(x, x_alt)

        # 2) Time slice mixing
        if self.use_time_mix and x_alt is not None and torch.rand(1) < self.p:
            x = time_slice_mix(x, x_alt)

        # 3) Frequency domain mixing
        if self.use_freq_mix and x_alt is not None and torch.rand(1) < self.p:
            x = frequency_mix(x, x_alt)

        # 4) Random cropping
        if self.use_crop and torch.rand(1) < self.p:
            x = random_crop(x)

        return x


