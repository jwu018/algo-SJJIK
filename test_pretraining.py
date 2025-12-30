"""
Test script for SSL pretraining only (no fine-tuning)
This script can be run independently on Lambda Cloud to verify pretraining works
"""

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

        # detect d_model from first transformer layer
        if hasattr(self.transformer, "layers") and len(self.transformer.layers) > 0:
            self.d_model = self.transformer.layers[0].self_attn.embed_dim
        else:
            self.d_model = getattr(self.transformer, "d_model", 128)

        # transformer batch_first flag (default False for nn.TransformerEncoder)
        self.batch_first = getattr(self.transformer, "batch_first", False)

        # lazy projection will be created on first forward if needed
        self._proj = None

    def _maybe_init_proj(self, feat_dim: int):
        if self._proj is None:
            self._proj = nn.Identity() if feat_dim == self.d_model else nn.Linear(feat_dim, self.d_model)

    def forward(self, x):
        # token_gen usually returns [B, F, S]; ensure 3D
        x = self.token_gen(x)
        if x.dim() != 3:
            raise RuntimeError(f"token_gen must return 3D tensor, got {tuple(x.shape)}")

        B, D1, D2 = x.shape
        # assume longer dim is sequence; convert to [B, S, F]
        if D2 >= D1:
            x = x.permute(0, 2, 1)  # [B, S, F]
            feat_dim = D1
        else:
            feat_dim = D2  # already [B, S, F]

        # project to transformer's d_model if needed
        self._maybe_init_proj(feat_dim)
        x = self._proj(x)  # [B, S, d_model]

        # feed transformer with correct layout
        if self.batch_first:
            x = self.transformer(x)  # [B, S, d_model]
        else:
            x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)  # [B, S, d_model]

        # pool expects [B, F, S]; convert and pool
        x = x.permute(0, 2, 1)  # [B, d_model, S]
        x = self.pool_lay(x).squeeze(-1)  # [B, d_model]
        return x  # Latent representation for SSL


def main():
    print("=" * 60)
    print("PRETRAINING TEST SCRIPT")
    print("=" * 60)

    # Import dependencies
    try:
        from models import TransformEEG
        from TF_augmenter_fix import Augmenter
        print("✓ Successfully imported models and augmenter")
    except ImportError as e:
        print(f"✗ Error importing dependencies: {e}")
        print("Make sure models.py and TF_augmenter_fix.py are in the same directory")
        return

    # Import dataloaders (only pretraining ones)
    # Try optimized version first, fallback to regular version
    try:
        from Lambda_Dataloading import train_Dataloader, val_Dataloader
        print("✓ Successfully imported dataloaders (standard version)")
        print(f"  - Training batches: {len(train_Dataloader)}")
        print(f"  - Validation batches: {len(val_Dataloader)}")
    except ImportError as e:
        print(f"✗ Error importing dataloaders: {e}")
        print("Make sure Lambda_Dataloading.py or Lambda_Dataloading_optimized.py is in the same directory")
        print("Also check that FILESYSTEM_NAME is correctly set")
        return

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  ⚠ WARNING: No GPU detected. Training will be slow on CPU.")

    print("\n" + "=" * 60)
    print("INITIALIZING MODELS")
    print("=" * 60)

    # Initialize full model
    ssl_backbone = TransformEEG(nb_classes=2, Chan=61, Features=244)
    print("✓ Created TransformEEG model (61 channels, 244 features)")

    # Wrap encoder
    encoder = TransformEEGEncoder(ssl_backbone)
    print("✓ Created TransformEEGEncoder wrapper")

    # Encoder
    NNencoder = encoder
    # It's suggested to copy the random initialization for embedding analysis
    NNencoder2 = copy.deepcopy(NNencoder)

    # SSL model
    head_size = [244, 122, 122]
    SelfMdl = selfeeg.ssl.SimCLR(
        encoder=NNencoder, projection_head=head_size).to(device=device)
    print(f"✓ Created SimCLR model with projection head {head_size}")

    # Count parameters
    total_params = sum(p.numel() for p in SelfMdl.parameters())
    trainable_params = sum(p.numel() for p in SelfMdl.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("CONFIGURING TRAINING")
    print("=" * 60)

    # loss (fit method has a default loss based on the SSL algorithm)
    loss = selfeeg.losses.simclr_loss
    loss_arg = {'temperature': 0.5}
    print(f"✓ Loss function: SimCLR loss (temperature={loss_arg['temperature']})")

    # earlystopper
    earlystop = selfeeg.ssl.EarlyStopping(
        patience=20, min_delta=1e-04, record_best_weights=True)
    print("✓ Early stopping: patience=20, min_delta=1e-04")

    # optimizer
    optimizer = torch.optim.Adam(SelfMdl.parameters(), lr=2.5e-5, betas=(0.75, 0.999))
    print("✓ Optimizer: Adam (lr=2.5e-5, betas=(0.75, 0.999))")

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    print("✓ LR Scheduler: ExponentialLR (gamma=0.99)")

    print("\n" + "=" * 60)
    print("STARTING PRETRAINING (1 EPOCH FOR TESTING)")
    print("=" * 60)
    print("Note: Set epochs=300 in production training")
    print()

    try:
        loss_info = SelfMdl.fit(
            train_dataloader      = train_Dataloader,
            augmenter             = Augmenter(),
            epochs                = 1,  # 1 epoch for testing, use 300 for full training
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

        print("\n" + "=" * 60)
        print("PRETRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        # Print loss information
        if loss_info is not None:
            print(f"\nFinal training loss: {loss_info['train'][-1]:.6f}")
            if 'validation' in loss_info and len(loss_info['validation']) > 0:
                print(f"Final validation loss: {loss_info['validation'][-1]:.6f}")

        # Save the pretrained encoder
        print("\n" + "=" * 60)
        print("SAVING PRETRAINED MODEL")
        print("=" * 60)

        pretrained_encoder = SelfMdl.get_encoder()
        torch.save(pretrained_encoder.state_dict(), "pretrained_encoder.pth")
        print("✓ Saved pretrained encoder to: pretrained_encoder.pth")

        # Also save the full SSL model for reference
        torch.save(SelfMdl.state_dict(), "simclr_model.pth")
        print("✓ Saved full SimCLR model to: simclr_model.pth")

        print("\n" + "=" * 60)
        print("TEST COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Verify the saved model files exist")
        print("2. Debug and prepare your fine-tuning data")
        print("3. Run full pretraining with epochs=300")
        print("4. Use pretrained_encoder.pth for fine-tuning")

    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR DURING TRAINING")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
