import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold
import pandas as pd
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR
from astro_net import AstroNet

DATA_DIR = 'data/finalDataset'
MODELS_DIR = 'models'
METRICS_DIR = 'data/metrics'
# PS dont increase this batch size unless you have enough VRAM on your DGPU it as is melts my 3070ti mobile lol
BATCH_SIZE = 32
NUM_CLASSES = 4
# Ideally dont go less than 20 epochs as it just starts to converge in my observation around 15-17th epoch
EPOCHS = 30
PATIENCE = 5
KFOLDS = 5

# FYI set this to cpu if you dont have DGPU on your lappy
#device = torch.device("cpu")
device = torch.device("cuda")

# Apply image transforms enhancements for better model convergence
transform = transforms.Compose([ transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.ToTensor(), transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)), transforms.Normalize([0.5]*3, [0.5]*3)])

# load and prep dataset from the final dataset folder
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(METRICS_DIR, f'{timestamp}_metrics.csv')
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
    print(f"\n[Fold {fold}/{KFOLDS}]")
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = AstroNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.3, anneal_strategy='cos')

    best_val_f1 = 0.0
    patience_counter = 0
    train_acc_hist, val_acc_hist, val_f1_hist = [], [], []

    for epoch in range(1, EPOCHS+1):
        model.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct/total
        train_acc_hist.append(train_acc)

        model.eval()
        val_correct = val_total = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        val_acc = val_correct/val_total
        val_f1  = f1_score(val_targets, val_preds, average='macro')
        val_acc_hist.append(val_acc)
        val_f1_hist.append(val_f1)

        print(f"Epoch {epoch:02d}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")

        # Early stop and save model if needed (PS dont torture the gpu or cpu anymore than it needs to lol)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model_path = os.path.join(MODELS_DIR, f"{timestamp}_fold{fold}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model for fold {fold} (Val F1={val_f1:.4f}) to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Print out best model classification report
    print("\nClassification Report:")
    print(classification_report(val_targets, val_preds, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(val_targets, val_preds))

    fold_results.append(pd.DataFrame({
        'epoch': list(range(1, len(train_acc_hist)+1)),
        f'train_acc_fold{fold}': train_acc_hist,
        f'val_acc_fold{fold}':   val_acc_hist,
        f'val_f1_fold{fold}':    val_f1_hist
    }))

# Save all stats for plotting 
merged = fold_results[0]
for df in fold_results[1:]:
    merged = merged.merge(df, on='epoch', how='outer')
merged.to_csv(csv_path, index=False)
print(f"\nSaved all folds’ metrics to {csv_path}")
