import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from datetime import datetime
from astro_resnet_50 import AstroResNet50

DATA_DIR = 'data/finalDataset'
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 4
EPOCHS = 20
PATIENCE = 10
KFOLDS = 5
METRICS_DIR = 'data/metrics'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(METRICS_DIR, exist_ok=True)
csv_path = os.path.join(METRICS_DIR, f'{timestamp}_metrics.csv')

fold_results = []

kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n[Fold {fold+1}/{KFOLDS}]")
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    model = AstroResNet50(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    patience_counter = 0
    train_acc_history, val_acc_history = [], []

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_history.append(train_acc)

        model.eval()
        val_correct = 0
        val_total = 0
        val_preds, val_targets = [], []
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1:03d}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("\nClassification Report:\n")
    print(classification_report(val_targets, val_preds, target_names=class_names, digits=4))
    print("Confusion Matrix:\n")
    print(confusion_matrix(val_targets, val_preds))

    fold_df = pd.DataFrame({
        'epoch': list(range(1, len(train_acc_history) + 1)),
        f'train_acc_fold{fold+1}': train_acc_history,
        f'val_acc_fold{fold+1}': val_acc_history
    })
    fold_results.append(fold_df)

merged_df = fold_results[0]
for df in fold_results[1:]:
    merged_df = pd.merge(merged_df, df, on='epoch', how='outer')

merged_df.to_csv(csv_path, index=False)
print(f"\n[✔] Saved metrics to {csv_path}")
