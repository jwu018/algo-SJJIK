import torch
import torch.nn.functional as F

# -----------------------------
# 1) Trial averaging
# -----------------------------
def trial_averaging(x1, x2):
    # x1, x2: [B, C, T]
    return (x1 + x2) / 2

# -----------------------------
# 2) Time slice mixing
# -----------------------------
def time_slice_mix(x1, x2, num_slices=4):
    # x1, x2: [B, C, T]
    B, C, T = x1.shape
    slice_len = T // num_slices
    x_new = x1.clone()

    for i in range(num_slices):
        start = i * slice_len
        end = T if i == num_slices-1 else (i+1) * slice_len
        # mask per batch: decide whether to swap this slice
        swap_mask = torch.rand(B, 1, 1, device=x1.device) < 0.5
        x_new[:, :, start:end] = torch.where(
            swap_mask, x2[:, :, start:end], x_new[:, :, start:end]
        )
    return x_new

# -----------------------------
# 3) Frequency domain mixing
# -----------------------------
def frequency_mix(x1, x2, alpha=0.5):
    # x1, x2: [B, C, T]
    X1 = torch.fft.fft(x1, dim=-1)
    X2 = torch.fft.fft(x2, dim=-1)
    X_new = alpha * X1 + (1 - alpha) * X2
    x_new = torch.fft.ifft(X_new, dim=-1).real
    return x_new

# -----------------------------
# 4) Random cropping (keeps [B, C, T])
# -----------------------------
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
