import selfeeg
import selfeeg.augmentation as aug
import selfeeg.dataloading as dl

# IMPORT CLASSICAL PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import random

# IMPORT TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import TransformEEG
from TF_augmenter_fix import Augmenter
from Dataloading_Test import train_Dataloader, val_Dataloader  # Your dataloaders

# -----------------------------
# SSL Encoder Wrapper
# -----------------------------
class TransformEEGEncoder(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.token_gen = original_model.token_gen
        self.transformer = original_model.transformer
        self.pool_lay = original_model.pool_lay

    def forward(self, x):
        x = self.token_gen(x)  # Token embedding
        x = torch.permute(x, [0, 2, 1])  # (batch, seq_len, feature_dim)
        x = self.transformer(x)
        x = torch.permute(x, [0, 2, 1])
        x = self.pool_lay(x).squeeze(-1)  # Global pooling
        return x  # Latent representation

# -----------------------------
# Initialize Models
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

baseline = TransformEEG(nb_classes=2, Chan=61, Features=128)
ssl_backbone = TransformEEG(nb_classes=2, Chan=61, Features=128)
encoder = TransformEEGEncoder(ssl_backbone)
NNencoder = encoder
NNencoder2 = copy.deepcopy(NNencoder)

# -----------------------------
# SSL Model: SimCLR
# -----------------------------
head_size = [128, 64, 64]
SelfMdl = selfeeg.ssl.SimCLR(
    encoder=NNencoder,
    projection_head=head_size
).to(device=device)

# Loss function
loss = selfeeg.losses.simclr_loss
loss_arg = {'temperature': 0.5}

# Early stopping
earlystop = selfeeg.ssl.EarlyStopping(
    patience=20, min_delta=1e-04, record_best_weights=True
)

# Optimizer & LR scheduler
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=2.5e-5, betas=(0.75,0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# -----------------------------
# SSL Pretraining
# -----------------------------
loss_info = SelfMdl.fit(
    train_dataloader      = train_Dataloader,
    augmenter             = Augmenter(),
    epochs                = 300,
    optimizer             = optimizer,
    loss_func             = loss,
    loss_args             = loss_arg,
    lr_scheduler          = scheduler,
    EarlyStopper          = earlystop,
    validation_dataloader = val_Dataloader,
    verbose               = True,
    device                = device,
    return_loss_info      = True
)

# -----------------------------
# Save Encoder Weights
# -----------------------------
torch.save(SelfMdl.get_encoder().state_dict(), "ssl_pretrained_encoder.pth")

print("SSL Pretraining completed and encoder weights saved.")
