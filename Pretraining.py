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
baseline = TransformEEG(nb_classes=2, Chan=61, Features=128) #paper has 61 channels
ssl_backbone = TransformEEG(nb_classes=2, Chan=61, Features=128)
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
    patience=20, min_delta=1e-04, record_best_weights=True)
# optimizer
optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=2.5e-5, betas=(0.75,0.999))
# lr scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

loss_info = SelfMdl.fit(
    train_dataloader      = train_Dataloader,
    augmenter             = Augmenter(),
    epochs                = 300, #paper has 300
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
FinalMdl = TransformEEG(nb_classes=2, Chan=61, Features=128)
#SelfMdl.train()
#SelfMdl.to(device='cpu') # device moving necessary or not?
#FinalMdl.encoder = SelfMdl.get_encoder()

pretrained_encoder = SelfMdl.get_encoder()
FinalMdl.to(device=device)
pretrained_encoder.to(device=device)

#loading in weights module-wise
FinalMdl.token_gen.load_state_dict(pretrained_encoder.token_gen.state_dict())
FinalMdl.transformer.load_state_dict(pretrained_encoder.transformer.state_dict())
FinalMdl.pool_lay.load_state_dict(pretrained_encoder.pool_lay.state_dict())

FinalMdl.train()

#defines the loss function
def loss_fineTuning(yhat, ytrue):
    ytrue = ytrue.float()
    yhat = torch.squeeze(yhat) #maybe yhat = yhat.view(-1)
    return F.binary_cross_entropy_with_logits(yhat, ytrue)

#defining the early stop for making sure not to overfit
earlystopFT = selfeeg.ssl.EarlyStopping(
    patience=20, min_delta=1e-04, record_best_weights=True) #paper has patience 20 and min_delta 1e-04

#sets the optimizer and the lr scheduler
optimizerFT = torch.optim.Adam(FinalMdl.parameters(), lr=2.5e-5, betas=(0.75,0.999)) #has a lr of 2.5e-5 and also beta1=0.75 while we have 0.9
schedulerFT = torch.optim.lr_scheduler.ExponentialLR(optimizerFT, gamma=0.99) #paper has gamma of 0.99

finetuning_loss=selfeeg.ssl.fine_tune(
    model                 = FinalMdl,
    train_dataloader      = trainloaderFT, # for when dataloader is set
    epochs                = 300,
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

from training import get_performances, WinRatio
FinalMdl.eval()
scores = {}
scores['loss_progression'] = finetuning_loss
verbose = False # double check if we want this
class_labels = [] # check in for
#- turn model probabilities/logits into decisions using a threshold. You can use a fixed 0.5 or pick a better one based on validation via ROC-driven selection.
# Evaluate model on the window level with the standard 0.5 threshold - - You classify as positive if score > 0.5,
scores['th_standard'] = get_performances(
        loader2eval    = testloaderFT,
        Model          = FinalMdl,
        device         = device,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels,
        roc_correction = False,
        plot_roc       = False,
        th             = 0.5,
        subj_ratio      = None,
    )
    #if not verbose:
bal_acc = scores['th_standard']['accuracy_weighted']
print(f'Balanced accuracy on windows with threshold 0.500 --> {bal_acc:.7f}')

#- You find the “best” decision threshold on the validation set (e.g., the threshold that maximizes balanced accuracy or Youden’s J from the ROC curve).
# Then you fix that threshold and evaluate on the test set -  always better than 0.5

    # Evalutate the model on the window level with a roc corrected threshold
th_eval = get_performances(
    loader2eval    = valloaderFT,
    Model          = FinalMdl,
    device         = device,
    class_labels   = class_labels,
    verbose        = False,
    return_scores  = False,
    plot_confusion = False,
    plot_roc       = False,
    roc_correction = True
)
th = th_eval['best_th']
scores['window_threshold'] = th
scores['th_corrected'] = get_performances(
    loader2eval    = testloaderFT,
    Model          = FinalMdl,
    device         = device,
    return_scores  = True,
    verbose        = verbose,
    plot_confusion = False,
    class_labels   = class_labels,
    roc_correction = False,
    plot_roc       = False,
    th             = th,
    subj_ratio      = None,
)
if not verbose:
    bal_acc_roc = scores['th_corrected']['accuracy_weighted']
    print(f'Balanced accuracy on windows with threshold {th:.3f} --> {bal_acc_roc:.3f}')

# Evalutate the model on the subject level
#Maybe take out the subject level on 0.5 threshold
# for window vs subject: - Window-level evaluation: How well the model classifies individual windows/segments.
# - Subject-level evaluation: How well the model decides the final label for a subject by aggregating its windows.
# - the clinical/real label is per subject. Window-level is a proxy.
#The first is on a 0.5 window while second is ROC- optimized
#Metrics to maybe report:
#- Window-level:
#     - AUC (threshold-independent) to show discriminative ability.
#     - Balanced accuracy, sensitivity, specificity at:
#         - threshold = 0.5
#         - ROC-tuned threshold (selected on validation), including the numeric threshold used
#
# - Subject-level:
#     - Balanced accuracy, sensitivity, specificity using:
#         - ratio threshold learned on validation (report the ratio value)
#         - combined with the ROC-tuned window threshold (recommended primary)
#
# - Consider reporting confidence intervals (bootstrap over subjects) for subject-level metrics.

subject_labeler = WinRatio()
subject_labeler.add_data(valloaderFT, FinalMdl, device='cpu', th = 0.5)
subject_labeler.compute_ratio()
subject_thresh = subject_labeler.get_ratio()
scores['subject_threshold_05'] = subject_thresh
scores['subject'] = get_performances(
    loader2eval    = testloaderFT,
    Model          = FinalMdl,
    device         = device,
    return_scores  = True,
    verbose        = verbose,
    plot_confusion = False,
    class_labels   = class_labels,
    roc_correction = False,
    plot_roc       = False,
    th             = 0.5,
    subj_ratio      = subject_thresh,
)
if not verbose:
    bal_acc_sub = scores['subject']['accuracy_weighted']
    print("using the following threshold for subject predictions", subject_thresh)
    print(f'Balanced accuracy on subject with threshold 0.500 --> {bal_acc_sub:.3f}')

subject_labeler = WinRatio()
subject_labeler.add_data(valloaderFT, FinalMdl, device='cpu', th = th)
subject_labeler.compute_ratio()
subject_thresh = subject_labeler.get_ratio()
scores['subject_threshold_th'] = subject_thresh
scores['subject_corrected'] = get_performances(
    loader2eval    = testloaderFT,
    Model          = FinalMdl,
    device         = device,
    return_scores  = True,
    verbose        = verbose,
    plot_confusion = False,
    class_labels   = class_labels,
    roc_correction = False,
    plot_roc       = False,
    th             = th,
    subj_ratio      = subject_thresh,
)
if not verbose:
    bal_acc_sub_roc = scores['subject_corrected']['accuracy_weighted']
    print("using the following threshold for subject predictions", subject_thresh)
    print(f'Balanced accuracy on subject with roc & th  {th:.3f} --> {bal_acc_sub_roc:.3f}')

#evaluating - code from selfeeg
# from sklearn.metrics import classification_report
# nb_classes=2
# FinalMdl.eval()
# ytrue=torch.zeros(len(testloaderFT.dataset))
# ypred=torch.zeros_like(ytrue)
# cnt=0
# for i, (X, Y) in enumerate(testloaderFT):
#     X=X.to(device=device)
#     ytrue[cnt:cnt+X.shape[0]]= Y
#     with torch.no_grad():
#         yhat = torch.sigmoid(FinalMdl(X)).to(device='cpu')
#         ypred[cnt:cnt+X.shape[0]] = torch.squeeze(yhat)
#     cnt += X.shape[0]
# print('Results of trivial Example\n')
# print(classification_report(ytrue,ypred>0.5))

# paper's evaulation
#    - Window-level performance at 0.5 threshold and ROC-corrected threshold.
#    - Subject-level aggregation using a learned “win ratio” threshold derived from validation.
#    - Extensive saving of results and model checkpoints per-fold.
#

