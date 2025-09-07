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
from Dataloading_Test import train_Dataloader, val_Dataloader
from Dataloading_Test import trainloaderFT, valloaderFT, testloaderFT

#in paper, dataloading has 125 and 250 Hz supported and 16s windows and 0.25 overlap and batch size is 64

# Initialize full model
baseline = TransformEEG(nb_classes=2, Chan=32, Features=128) #paper has 61 channels
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
    epochs                = 100, #paper has 300
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

#defines the backbone and then sets the pretrained encoder onto the final model
FinalMdl = TransformEEG(nb_classes=2, Chan=32, Features=128)
SelfMdl.train()
SelfMdl.to(device='cpu')
FinalMdl.encoder = SelfMdl.get_encoder()
FinalMdl.train()
FinalMdl.to(device=device)

#defines the loss function
def loss_fineTuning(yhat, ytrue):
    ytrue = ytrue + 0.
    yhat = torch.squeeze(yhat)
    return F.binary_cross_entropy_with_logits(yhat, ytrue)

#defining the early stop for making sure not to overfit
earlystopFT = selfeeg.ssl.EarlyStopping(
    patience=10, min_delta=1e-03, record_best_weights=True) #paper has patience 20 and min_delta 1e-04

#sets the optimizer and the lr scheduler
optimizerFT = torch.optim.Adam(FinalMdl.parameters(), lr=1e-3) #has a lr of 2.5e-5 and also beta1=0.75 while we have 0.9
schedulerFT = torch.optim.lr_scheduler.ExponentialLR(optimizerFT, gamma=0.97) #paper has gamma of 0.99

finetuning_loss=selfeeg.ssl.fine_tune(
    model                 = FinalMdl,
    train_dataloader      = trainloaderFT, # for when dataloader is set
    epochs                = 15,
    optimizer             = optimizerFT,
    loss_func             = loss_fineTuning,
    lr_scheduler          = schedulerFT,
    EarlyStopper          = earlystopFT,
    validation_dataloader = valloaderFT, # for when dataloader is set
    verbose               = True,
    device                = device,
    return_loss_info      = True
)

#saves the finalized model after training
torch.save(FinalMdl.state_dict(), "finetuned_model.pth")


#evaluating the final model code
from sklearn.metrics import classification_report
nb_classes=2
FinalMdl.eval()
ytrue=torch.zeros(len(testloaderFT.dataset))
ypred=torch.zeros_like(ytrue)
cnt=0
for i, (X, Y) in enumerate(testloaderFT):
    X=X.to(device=device)
    ytrue[cnt:cnt+X.shape[0]]= Y
    with torch.no_grad():
        yhat = torch.sigmoid(FinalMdl(X)).to(device='cpu')
        ypred[cnt:cnt+X.shape[0]] = torch.squeeze(yhat)
    cnt += X.shape[0]


# paper's evaulation
#    - Window-level performance at 0.5 threshold and ROC-corrected threshold.
#    - Subject-level aggregation using a learned “win ratio” threshold derived from validation.
#    - Extensive saving of results and model checkpoints per-fold.
#

print('Results of trivial Example\n')
print(classification_report(ytrue,ypred>0.5))