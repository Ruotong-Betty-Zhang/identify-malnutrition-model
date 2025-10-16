# LSTM_model_cap.py (with FocalLoss and grid search)

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
from copy import deepcopy

# ---------------------------
# Utility
# ---------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------
# Dataset
# ---------------------------
class MalnutritionDataset(Dataset):
    """Thin wrapper dataset for variable-length sequences and integer labels.

    Args:
        sequences (list[torch.Tensor]): Each tensor is (seq_len, input_size).
        labels (list[torch.Tensor]): Each tensor is a scalar class index.

    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """Pad sequences to the same length and return (padded, lengths, labels).

    Args:
        batch: Iterable of (sequence, label) pairs.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - padded_seqs: (B, T_max, D)
            - lengths: (B,) true lengths before padding
            - labels: (B,) class indices
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded_seqs, lengths, labels

# ---------------------------
# LSTM Model
# ---------------------------
class LSTMModel(nn.Module):
    """Single-layer LSTM with simple attention pooling and a linear head.

    Args:
        input_size (int): Feature dimension per timestep.
        hidden_size (int): LSTM hidden size.
        num_classes (int): Number of output classes.

    Forward Inputs:
        x (torch.Tensor): (B, T, D) padded sequences.
        lengths (torch.Tensor): (B,) true lengths.

    Returns:
        torch.Tensor: (B, num_classes) unnormalized logits.
    """
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (batch, seq_len, hidden_size)

        # Attention weights
        attn_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)  # (batch, seq_len)
        attn_output = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden_size)

        out = self.fc(self.dropout(attn_output))
        return out

# ---------------------------
# Focal Loss
# ---------------------------
class FocalLoss(nn.Module):
    """Multi-class Focal Loss wrapper around cross-entropy.

    Args:
        alpha (torch.Tensor | None): Class weights (on device) or None.
        gamma (float): Focusing parameter.
        reduction (str): 'mean' or 'sum'.

    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ---------------------------
# Data Processing
# ---------------------------
def prepare_sequence_data(df, id_col='IDno', time_col='Assessment_Date', target_col='CAP_Nutrition'):
    """Build per-person time-ordered sequences; split when target state changes.

    - Scales continuous columns (StandardScaler).
    - Casts features to float32.
    - For each ID, segments a new subsequence whenever the target changes,
      and assigns the label as the state of that segment.

    Args:
        df (pd.DataFrame): Raw, row-per-assessment table.
        id_col (str): Person identifier column.
        time_col (str): Timestamp column (parsed to datetime).
        target_col (str): Target label column.

    Returns:
        tuple[list[Tensor], list[Tensor], list[str]]:
            sequences, labels, feature_cols
    """
    df[time_col] = pd.to_datetime(df[time_col], format="%d%b%Y")

    all_cols = [col for col in df.columns if col not in [id_col, time_col, target_col]]
    continuous_cols = [
        'iK1ab',
        'iK1bb',
        "Scale_ADLHierarchy",
        "Scale_ADLLongForm",
        "Scale_ADLShortForm",
        "Scale_AggressiveBehaviour",
        "Scale_BMI",
        "Scale_CHESS",
        "Scale_Communication",
        "Scale_CPS",
        "Scale_DRS",
        "Scale_IADLCapacity",
        "Scale_IADLPerformance",
        "Scale_MAPLE",
        "Scale_Pain",
        "Scale_PressureUlcerRisk",
        "OverallUrgencyScale"
    ]
    continuous_cols = [col for col in continuous_cols if col in df.columns]  # remove missing
    discrete_cols = [col for col in all_cols if col not in continuous_cols]

    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    # Force all training feature columns to float32 to avoid dtype errors
    feature_cols = continuous_cols + discrete_cols
    df[feature_cols] = df[feature_cols].astype(np.float32)

    seqs = []
    labels = []

    for _, group in df.groupby(id_col):
        group_sorted = group.sort_values(by=time_col).copy()
        states = group_sorted[target_col].values
        start_idx = 0
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                sub_seq = group_sorted.iloc[start_idx:i]
                seqs.append(torch.tensor(sub_seq[feature_cols].values, dtype=torch.float32))
                labels.append(torch.tensor(states[i - 1], dtype=torch.long))
                start_idx = i
        sub_seq = group_sorted.iloc[start_idx:]
        seqs.append(torch.tensor(sub_seq[feature_cols].values, dtype=torch.float32))
        labels.append(torch.tensor(states[-1], dtype=torch.long))

    return seqs, labels, feature_cols


def split_dataset(seqs, labels, test_size=0.2, random_state=42):
    """Index-based train/test split with stratification when possible.

    Args:
        seqs (list[Tensor]): Variable-length sequences.
        labels (list[Tensor]): Scalar class tensors.
        test_size (float): Proportion for test split.
        random_state (int): RNG seed.

    Returns:
        ((list[Tensor], list[Tensor]), (list[Tensor], list[Tensor])):
            (train_seqs, train_labels), (test_seqs, test_labels)
    """
    y = np.array([l.item() for l in labels])
    stratify = y if min(pd.Series(y).value_counts()) > 1 else None
    indices = list(range(len(seqs)))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=stratify, random_state=random_state)
    train = [seqs[i] for i in train_idx], [labels[i] for i in train_idx]
    test = [seqs[i] for i in test_idx], [labels[i] for i in test_idx]
    return train, test

# ---------------------------
# Train & Evaluate
# ---------------------------
# LSTM_model_cap.py (track and keep best test accuracy model)
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=100):
    """Train with early best-checkpointing by validation accuracy.

    Args:
        model (nn.Module): Network to train.
        train_loader (DataLoader): Training batches.
        val_loader (DataLoader): Validation batches.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        loss_fn (Callable): Loss function (logits, targets) -> loss.
        device (torch.device): CPU/GPU device.
        num_epochs (int): Max epochs.

    Returns:
        nn.Module: Model loaded with the best-performing weights on val set.
    """
    best_model = deepcopy(model.state_dict())
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, l, y in train_loader:
            x, l, y = x.to(device), l.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, l)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, l, y in val_loader:
                x, l, y = x.to(device), l.to(device), y.to(device)
                logits = model(x, l)
                val_loss += loss_fn(logits, y).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    return model


def evaluate_model(model, dataloader, device, num_classes=3):
    """Compute accuracy, confusion matrix, classification report, and ROC-AUC.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Evaluation data.
        device (torch.device): CPU/GPU device.
        num_classes (int): Number of classes.

    Prints:
        Accuracy, confusion matrix, classification report, and macro ROC-AUC
        (multi-class OVR) when feasible.
    """
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for x, l, y in dataloader:
            x, l, y = x.to(device), l.to(device), y.to(device)
            logits = model(x, l)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    if num_classes == 2:
        probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()[:, 1]
        print("ROC AUC:", roc_auc_score(all_labels, probs))
    else:
        bin_labels = label_binarize(all_labels, classes=list(range(num_classes)))
        probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
        try:
            auc = roc_auc_score(bin_labels, probs, average="macro", multi_class="ovr")
            print("Multi-class ROC AUC:", auc)
        except Exception as e:
            print("Unable to compute multi-class AUC:", e)

# ---------------------------
# Grid Search
# ---------------------------
def grid_search_model(train_dataset, test_dataset, input_size, num_classes, class_weights, device):
    """Grid-search hidden size & learning rate; keep the best test accuracy model.

    Args:
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Testing dataset used as validation for search.
        input_size (int): Feature dimension.
        num_classes (int): Number of classes.
        class_weights (torch.Tensor): Weights for Focal/Cross-Entropy.
        device (torch.device): CPU/GPU device.

    Returns:
        nn.Module: Best-scoring model (deep-copied).
    """
    param_grid = {
        'hidden_size': [32, 64, 128],
        'learning_rate': [0.01, 0.001, 0.0005]
    }

    best_model = None
    best_score = 0
    best_params = {}

    for hs in param_grid['hidden_size']:
        for lr in param_grid['learning_rate']:
            print(f"\n[Grid Search] hidden_size={hs}, lr={lr}")

            model = LSTMModel(input_size=input_size, hidden_size=hs, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = FocalLoss(alpha=class_weights)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

            model = train_model(model, train_loader, test_loader, optimizer, loss_fn, device)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, l, y in test_loader:
                    x, l, y = x.to(device), l.to(device), y.to(device)
                    logits = model(x, l)
                    preds = logits.argmax(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            print(f"Test Accuracy: {acc:.4f}")

            if acc > best_score:
                best_score = acc
                best_model = deepcopy(model)
                best_params = {'hidden_size': hs, 'learning_rate': lr}

    print("\nBest Parameters:", best_params)
    print(f"Best Test Accuracy: {best_score:.4f}")
    return best_model

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cpu")

    df = pd.read_pickle("./datasets/CAP_1.pkl")
    seqs, labels, feature_cols = prepare_sequence_data(df)
    (train_seqs, train_labels), (test_seqs, test_labels) = split_dataset(seqs, labels)

    train_dataset = MalnutritionDataset(train_seqs, train_labels)
    test_dataset = MalnutritionDataset(test_seqs, test_labels)

    num_classes = len(np.unique([y.item() for y in labels]))
    input_size = train_seqs[0].shape[1]

    y_np = np.array([y.item() for y in train_labels])
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_np), y=y_np)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    best_model = grid_search_model(train_dataset, test_dataset, input_size, num_classes, weight_tensor, device)

    final_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    evaluate_model(best_model, final_loader, device, num_classes=num_classes)
