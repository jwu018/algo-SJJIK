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

#set seed to 12

class TransformEEGEncoder(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.token_gen = original_model.token_gen
        self.transformer = original_model.transformer
        self.pool_lay = original_model.pool_lay

    def forward(self, x):
        # Token embedding
        x = self.token_gen(x)

        # Reshape for transformer
        x = torch.permute(x, [0, 2, 1])  # (batch, seq_len, feature_dim)

        # Transformer encoder
        x = self.transformer(x)

        # Reshape back for pooling
        x = torch.permute(x, [0, 2, 1])

        # Global pooling
        x = self.pool_lay(x)  # (batch, features, 1)
        x = x.squeeze(-1)     # (batch, features)

        return x  # Latent representation for SSL

from models import TransformEEG
from TF_augmenter_fix import Augmenter
from Dataloading_Test import train_Dataloader
from Dataloading_Test import val_Dataloader

# Initialize full model
baseline = TransformEEG(nb_classes=2, Chan=32, Features=128)
ssl_backbone = TransformEEG(nb_classes=2, Chan=32, Features=128)
# Wrap encoder
encoder = TransformEEGEncoder(ssl_backbone)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder
NNencoder= encoder
# It's suggested to copy the random initialization for embedding analysis
NNencoder2= copy.deepcopy(NNencoder)

# SSL model
head_size=[ 128, 64, 64]
SelfMdl = selfeeg.ssl.SimCLR(
    encoder=NNencoder, projection_head=head_size).to(device=device)

# loss (fit method has a default loss based on the SSL algorithm
loss=selfeeg.losses.simclr_loss
loss_arg={'temperature': 0.5}

# earlystopper
earlystop = selfeeg.ssl.EarlyStopping(
    patience=25, min_delta=1e-05, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=1e-3)
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

loss_info = SelfMdl.fit(
    train_dataloader      = train_Dataloader,
    augmenter             = Augmenter(),
    epochs                = 5,
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

