# pipeline.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import selfeeg  # keep consistency
from s3_selfeeg_dataloader import get_s3_dataloaders


# ---------------------------
# Example Augmenter (customize as needed)
# ---------------------------
class Augmenter:
    def __call__(self, x):
        # x: numpy array [C, T]
        # simple augmentation example: random gaussian noise
        noise = 0.01 * torch.randn_like(torch.tensor(x, dtype=torch.float32))
        return torch.tensor(x, dtype=torch.float32) + noise


# ---------------------------
# Example Encoder Backbone
# ---------------------------
class SimpleEEGEncoder(nn.Module):
    def __init__(self, in_channels=21, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        # x: [B, C, T]
        h = self.net(x)
        h = h.squeeze(-1)
        z = self.fc(h)
        return z


# ---------------------------
# SSL Pretraining Step (SimCLR-like)
# ---------------------------
class SSLModel(nn.Module):
    def __init__(self, encoder, emb_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.projection(z)


def ssl_loss(z1, z2, temperature=0.5):
    """NT-Xent loss."""
    batch_size = z1.size(0)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(reps, reps.T) / temperature
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    targets = torch.arange(batch_size, device=z1.device)
    targets = torch.cat([targets + batch_size, targets])
    loss = nn.CrossEntropyLoss()(sim, targets)
    return loss


# ---------------------------
# Fine-tuning Classifier
# ---------------------------
class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)  # freeze encoder
        return self.fc(features)


# ---------------------------
# Training Loops
# ---------------------------
def pretrain_ssl(model, train_loader, val_loader, epochs=10, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch.to(device)  # [B, C, T]

            # two augmented views
            aug = Augmenter()
            x1 = aug(x)
            x2 = aug(x)

            z1 = model(x1)
            z2 = model(x2)

            loss = ssl_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[SSL] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model.encoder  # return pretrained encoder


def finetune(encoder, train_loader, val_loader, test_loader, num_classes=2, epochs=5, device="cuda"):
    model = Classifier(encoder, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch.to(device)
            y = torch.randint(0, num_classes, (x.size(0),), device=device)  # dummy labels for now

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[FT] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch.to(device)
            y = torch.randint(0, num_classes, (x.size(0),), device=device)  # dummy labels
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy (dummy labels): {100*correct/total:.2f}%")
    return model


# ---------------------------
# Main Pipeline
# ---------------------------
if __name__ == "__main__":
    bucket = "your-bucket-name"

    train_loader, val_loader, trainloaderFT, valloaderFT, testloaderFT = get_s3_dataloaders(
        bucket_name=bucket,
        freq=250,
        window=16,
        overlap=0.25,
        batch_size=32,
        seed=42,
        use_train_augment=True,
        train_augmenter=Augmenter()
    )

    encoder = SimpleEEGEncoder(in_channels=21, emb_dim=128)
    ssl_model = SSLModel(encoder, emb_dim=128)

    # Step 1: Pretrain with SSL
    pretrained_encoder = pretrain_ssl(ssl_model, train_loader, val_loader, epochs=5)

    # Step 2: Fine-tune on downstream (dummy labels here, replace with real labels)
    finetune(pretrained_encoder, trainloaderFT, valloaderFT, testloaderFT, num_classes=2, epochs=3)

#    - Extensive saving of results and model checkpoints per-fold.

