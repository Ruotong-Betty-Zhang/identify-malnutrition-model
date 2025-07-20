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
# Dataset supporting ID index
# ---------------------------
class MalnutritionDatasetWithID(Dataset):
    def __init__(self, sequences, labels, id_indices):
        self.sequences = sequences
        self.labels = labels
        self.id_indices = id_indices
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.id_indices[idx]

def collate_fn_with_id(batch):
    sequences, labels, id_indices = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    id_indices = torch.tensor(id_indices)
    return padded_seqs, lengths, labels, id_indices

# ---------------------------
# T-LSTM Model with ID embedding
# ---------------------------
class TLSTMWithID(nn.Module):
    def __init__(self, input_size, num_ids, id_embed_dim=16, hidden_size=64, num_classes=3):
        super(TLSTMWithID, self).__init__()
        self.input_size = input_size - 1  # time delta counted in input_size
        self.hidden_size = hidden_size
        self.id_embedding = nn.Embedding(num_ids, id_embed_dim)

        self.input_layer = nn.Linear(self.input_size + id_embed_dim, hidden_size)
        self.decay_layer = nn.Linear(1, hidden_size)

        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths, id_indices):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        id_embeds = self.id_embedding(id_indices)  # (batch, id_embed_dim)
        id_embeds = id_embeds.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, id_embed_dim)

        x = torch.cat([x[:, :, :-1], id_embeds, x[:, :, -1:].clone()], dim=2)  # 拼接特征和id embedding以及time delta

        for t in range(seq_len):
            mask = (t < lengths).float().unsqueeze(1)
            time_delta = x[:, t, -1].unsqueeze(1)
            features = x[:, t, :-1]
            embedded = torch.tanh(self.input_layer(features))
            gamma = torch.sigmoid(self.decay_layer(time_delta))
            c_t = c_t * gamma
            h_t, c_t = self.lstm_cell(embedded, (h_t, c_t))
            h_t = h_t * mask
            c_t = c_t * mask

        out = self.output_layer(self.dropout(h_t))
        return out

# ---------------------------
# Data Processing
# ---------------------------
def prepare_sequence_data_split_state(df, id_col='IDno', time_col='Assessment_Date', target_col='CAP_Nutrition'):
    df[time_col] = pd.to_datetime(df[time_col], format="%d%b%Y")
    base_feature_cols = [col for col in df.columns if col not in [id_col, time_col, target_col]]
    scaler = StandardScaler()
    df[base_feature_cols] = scaler.fit_transform(df[base_feature_cols])

    seqs = []
    labels = []
    ids = []

    for pid, group in df.groupby(id_col):
        group_sorted = group.sort_values(by=time_col).copy()
        group_sorted['Time_Delta'] = (group_sorted[time_col] - group_sorted[time_col].iloc[0]).dt.days
        max_delta = group_sorted['Time_Delta'].max()
        if max_delta > 0:
            group_sorted['Time_Delta'] = group_sorted['Time_Delta'] / max_delta
        else:
            group_sorted['Time_Delta'] = 0

        # Based on the target_col, we can assume it is a categorical state
        states = group_sorted[target_col].values
        start_idx = 0
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                sub_seq = group_sorted.iloc[start_idx:i]
                seqs.append(torch.tensor(sub_seq[base_feature_cols + ['Time_Delta']].values, dtype=torch.float32))
                labels.append(torch.tensor(states[i - 1], dtype=torch.long))
                ids.append(pid)
                start_idx = i
        # Handle the last segment
        sub_seq = group_sorted.iloc[start_idx:]
        seqs.append(torch.tensor(sub_seq[base_feature_cols + ['Time_Delta']].values, dtype=torch.float32))
        labels.append(torch.tensor(states[-1], dtype=torch.long))
        ids.append(pid)

    feature_cols = base_feature_cols + ['Time_Delta']
    return seqs, labels, ids, feature_cols

def split_dataset_by_id_with_id_index(seqs, labels, ids, test_size=0.2, random_state=42):
    ids = [str(x) for x in ids]
    unique_ids = sorted(list(set(ids)))
    id_to_idx = {pid: idx for idx, pid in enumerate(unique_ids)}

    labels_for_ids = []
    for uid in unique_ids:
        idx = ids.index(uid)
        labels_for_ids.append(labels[idx].item() if isinstance(labels[idx], torch.Tensor) else labels[idx])

    stratify = labels_for_ids if min(pd.Series(labels_for_ids).value_counts()) > 1 else None
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state, stratify=stratify)

    def collect(id_list):
        filtered_seqs = []
        filtered_labels = []
        filtered_id_indices = []
        for seq, label, pid in zip(seqs, labels, ids):
            if pid in id_list:
                filtered_seqs.append(seq)
                filtered_labels.append(label)
                filtered_id_indices.append(id_to_idx[pid])
        return filtered_seqs, filtered_labels, filtered_id_indices

    train_seqs, train_labels, train_id_indices = collect(train_ids)
    test_seqs, test_labels, test_id_indices = collect(test_ids)
    return (train_seqs, train_labels, train_id_indices), (test_seqs, test_labels, test_id_indices), id_to_idx

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=200, patience=200):
    best_model = deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, all_preds, all_labels = 0, [], []
        for x, l, y, id_idx in train_loader:
            x, l, y, id_idx = x.to(device), l.to(device), y.to(device), id_idx.to(device)
            optimizer.zero_grad()
            logits = model(x, l, id_idx)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        val_loss = 0
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, l, y, id_idx in val_loader:
                x, l, y, id_idx = x.to(device), l.to(device), y.to(device), id_idx.to(device)
                logits = model(x, l, id_idx)
                val_loss += loss_fn(logits, y).item()
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy_score(val_labels, val_preds):.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break
    model.load_state_dict(best_model)
    return model

def grid_search_model(train_loader, test_loader, input_size, num_classes, y_seqs, device):
    param_grid = {
        'hidden_size': [32, 64, 128],
        'learning_rate': [0.01, 0.001, 0.0005],
    }

    best_model = None
    best_score = 0
    best_params = {}

    y_np = np.array([y.item() for y in y_seqs])
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_np), y=y_np)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    for hs in param_grid['hidden_size']:
        for lr in param_grid['learning_rate']:
            print(f"\n[Grid Search] hidden_size={hs}, lr={lr}")

            model = TLSTMModel(input_size=input_size, hidden_size=hs, num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
            loss_fn = FocalLoss(alpha=weight_tensor)


            model = train_model(model, train_loader, test_loader, optimizer, loss_fn, device)

            # Evaluate
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
    return best_model, best_params



def evaluate_model(model, dataloader, device, num_classes=3):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for x, l, y, id_idx in dataloader:
            x, l, y, id_idx = x.to(device), l.to(device), y.to(device), id_idx.to(device)
            logits = model(x, l, id_idx)
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
# ---------------------------
# Feature Importance (Permutation)
# ---------------------------
def permutation_feature_importance(model, dataset, feature_idx, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_id)

    def get_accuracy(mod):
        mod.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, l, y, id_idx in loader:
                x, l, y, id_idx = x.to(device), l.to(device), y.to(device), id_idx.to(device)
                logits = mod(x, l, id_idx)
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        return accuracy_score(labels, preds)

    baseline = get_accuracy(model)
    shuffled_model = deepcopy(model)

    # Shuffle the specified feature across all sequences in the batch
    with torch.no_grad():
        for x, l, y, id_idx in loader:
            x = x.clone() 
            for b in range(x.size(0)):
                seq_len = l[b]
                perm = torch.randperm(seq_len)
                x[b, :seq_len, feature_idx] = x[b, :seq_len, feature_idx][perm]
    drop = baseline - get_accuracy(shuffled_model)
    return drop

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cpu")

    df = pd.read_pickle("./datasets/cap_data.pkl")
    seqs, labels, ids, feature_cols = prepare_sequence_data_split_state(df)
    (train_seqs, train_labels, train_id_indices), (test_seqs, test_labels, test_id_indices), id_to_idx = split_dataset_by_id_with_id_index(seqs, labels, ids)
    print(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

    train_dataset = MalnutritionDatasetWithID(train_seqs, train_labels, train_id_indices)
    test_dataset = MalnutritionDatasetWithID(test_seqs, test_labels, test_id_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_with_id)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_id)

    num_classes = len(np.unique([y.item() for y in labels]))
    input_size = train_seqs[0].shape[1]


    # best_model, best_params = grid_search_model(
    #     train_loader, test_loader, input_size, num_classes, y_seqs, device
    # )
    # evaluate_model(best_model, test_loader, device, num_classes=num_classes)
    # importances = [permutation_feature_importance(best_model, MalnutritionDataset(test_X, test_y), i, device) for i in range(input_size)]

    model = TLSTMWithID(input_size=input_size, num_ids=len(id_to_idx), id_embed_dim=16, hidden_size=64, num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    y_np = np.array([y.item() for y in train_labels])
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_np), y=y_np)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    # model = train_model(model, train_loader, test_loader, optimizer, loss_fn, device)
    # evaluate_model(model, test_loader, device, num_classes=num_classes)

    # Plot top-10 permutation importance
    # importances = [permutation_feature_importance(model, MalnutritionDataset(test_X, test_y), i, device) for i in range(input_size)]
    # sorted_idx = np.argsort(importances)[::-1]
    # plt.barh([feature_cols[i] for i in sorted_idx[:10]], [importances[i] for i in sorted_idx[:10]])
    # plt.gca().invert_yaxis()
    # plt.title("Top 10 Permutation Importances")
    # plt.xlabel("Accuracy Drop")
    # plt.tight_layout()
    # plt.show()
