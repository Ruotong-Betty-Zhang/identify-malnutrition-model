"""Malnutrition LSTM pipeline.

This script prepares variable-length longitudinal sequences from interRAI LTCF
assessments, trains a bidirectional LSTM classifier with early stopping, 
performs simple grid search over a few hyperparameters, and estimates 
permutation feature importance at the sequence level.

Notes:
- Sequences are built per ID and sorted by assessment date.
- A 'Time_Delta' feature (days since previous assessment) is appended.
- The final label per ID is the last observed 'Malnutrition' state.
- No functional changes from the original code; comments/docstrings only.
"""

import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from copy import deepcopy
import random
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42) -> None:
    """Set RNG seeds for reproducibility across numpy/torch/cuda.

    Args:
        seed: Global seed to apply.

    Note:
        For cuDNN, we turn off benchmarking and set deterministic for stability.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_sequence_data(
    df: pd.DataFrame,
    id_col: str = "IDno",
    time_col: str = "Assessment_Date",
    target_col: str = "Malnutrition",
):
    """Construct per-person time-ordered sequences; append Time_Delta feature.

    Steps:
        1) Parse timestamps and standardize all non-ID/time/target features.
        2) For each person (ID), sort by time and compute Time_Delta in days.
        3) Build a float32 tensor X (T, D) and an integer label y (last state).

    Args:
        df: Row-per-assessment dataframe.
        id_col: Column name for person identifier.
        time_col: Column name for assessment timestamp (string format "%d%b%Y").
        target_col: Column name for classification label.

    Returns:
        X_seqs (list[Tensor]): Each (T, D) sequence (padded later in collate_fn).
        y_seqs (list[Tensor]): Each scalar class index (last observed target).
        lengths (list[int]): True sequence lengths per ID (before padding).
        id_list (list[Any]): ID values aligned with X_seqs/y_seqs.
    """
    df[time_col] = pd.to_datetime(df[time_col], format="%d%b%Y")

    # Select feature columns: exclude ID, time, and target
    base_feature_cols = [c for c in df.columns if c not in [id_col, time_col, target_col]]
    print("The original df shape is:", df.shape)

    # Standardize features (fit on all rows, then transform)
    scaler = StandardScaler()
    df_features = df[base_feature_cols].copy()
    df[base_feature_cols] = scaler.fit_transform(df_features)

    X_seqs, y_seqs, lengths, id_list = [], [], [], []

    for pid, group in df.groupby(id_col):
        group_sorted = group.sort_values(by=time_col).copy()

        # Compute inter-assessment gaps in days; first assessment gets 0
        group_sorted["Time_Delta"] = group_sorted[time_col].diff().dt.days.fillna(0)

        # Compose feature matrix + time delta (float32 tensor)
        feature_data = group_sorted[base_feature_cols].copy()
        feature_data["Time_Delta"] = group_sorted["Time_Delta"]

        X = torch.tensor(feature_data.values, dtype=torch.float32)
        y = torch.tensor(group_sorted[target_col].values[-1], dtype=torch.long)

        X_seqs.append(X)
        y_seqs.append(y)
        lengths.append(X.shape[0])
        id_list.append(pid)

    return X_seqs, y_seqs, lengths, id_list


def split_sequence_dataset_by_id(
    X_seqs, y_seqs, lengths, id_list, test_size: float = 0.2, random_state: int = 42
):
    """Split sequences into train/test by unique IDs (optionally stratified).

    We stratify on per-ID labels when each class has >= 2 IDs; otherwise fall back
    to non-stratified split to avoid scikit-learn errors.

    Args:
        X_seqs: List of (T, D) tensors per ID.
        y_seqs: List of scalar class tensors per ID (last state).
        lengths: List of sequence lengths per ID.
        id_list: List of IDs aligned with X_seqs.
        test_size: Proportion of IDs to allocate to the test set.
        random_state: RNG seed for the split.

    Returns:
        (X_train, y_train, len_train), (X_test, y_test, len_test)
    """
    unique_ids = list(set(id_list))

    # Map each unique ID to a single label (the corresponding per-ID target)
    id_to_label = {}
    for i, pid in enumerate(id_list):
        if pid not in id_to_label:
            id_to_label[pid] = y_seqs[i].item() if isinstance(y_seqs[i], torch.Tensor) else y_seqs[i]

    labels_for_unique_ids = [id_to_label[uid] for uid in unique_ids]

    # Enable stratification only if every class has at least 2 IDs
    from collections import Counter
    label_counts = Counter(labels_for_unique_ids)
    stratify_option = labels_for_unique_ids if min(label_counts.values()) >= 2 else None

    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=random_state, stratify=stratify_option
    )

    X_train, y_train, len_train = [], [], []
    X_test, y_test, len_test = [], [], []

    for i, pid in enumerate(id_list):
        if pid in train_ids:
            X_train.append(X_seqs[i]); y_train.append(y_seqs[i]); len_train.append(lengths[i])
        else:
            X_test.append(X_seqs[i]);  y_test.append(y_seqs[i]);  len_test.append(lengths[i])

    return (X_train, y_train, len_train), (X_test, y_test, len_test)


class MalnutritionDataset(Dataset):
    """Thin wrapper for variable-length sequence tensors and integer labels.

    Args:
        sequences (list[Tensor]): Each (T, D) tensor per ID.
        labels (list[Tensor]): Each scalar class tensor per ID.
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """Pad sequences in a batch; return padded, lengths, and stacked labels.

    Args:
        batch: Iterable of (sequence, label) pairs.

    Returns:
        padded_seqs (Tensor): (B, T_max, D)
        lengths (Tensor): (B,) true lengths before padding
        labels (Tensor): (B,) class indices
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    # Right-pad to max length in batch; LSTM uses lengths to ignore pads
    padded_seqs = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded_seqs, lengths, labels


class LSTMModel(nn.Module):
    """Bidirectional LSTM classifier with LayerNorm, Dropout, and linear head.

    Args:
        input_size (int): Feature dimension per timestep.
        hidden_size (int): LSTM hidden size per direction.
        num_classes (int): Number of output classes.

    Forward:
        x (Tensor): (B, T, D) padded sequences.
        lengths (Tensor): (B,) true lengths.

    Returns:
        logits (Tensor): (B, num_classes) unnormalized scores.
    """
    def __init__(self, input_size, hidden_size: int = 64, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        # Pack to skip padded timesteps; enforce_sorted=False allows arbitrary order
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (ht, ct) = self.lstm(packed)
        # Concatenate last hidden states from both directions
        out = self.fc(self.dropout(self.norm(torch.cat([ht[-2], ht[-1]], dim=1))))
        return out


def train_model_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    num_epochs: int = 100,
    patience: int = 5,
):
    """Train with early stopping based on validation accuracy.

    Tracks best model parameters by highest validation accuracy; stops when
    no improvement is observed for `patience` epochs.

    Returns:
        nn.Module: Model loaded with best-performing weights.
    """
    best_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch_x, batch_len, batch_y in train_loader:
            batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, batch_len)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # ----- validation -----
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_len, batch_y in val_loader:
                batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
                logits = model(batch_x, batch_len)
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Early-stopping bookkeeping
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> None:
    """Evaluate model on a dataloader and print accuracy/report/confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_len, batch_y in dataloader:
            batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
            logits = model(batch_x, batch_len)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nEvaluation Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


def search_best_model(X_seqs, y_seqs, lengths, id_list):
    """Grid search over hidden_size, learning_rate, batch_size; keep best test acc.

    Note:
        We compute class weights on the full label array and reuse them inside
        each run to mitigate class imbalance.

    Returns:
        best_model (nn.Module), best_config (dict)
    """
    hidden_sizes = [32, 64, 128, 256]
    learning_rates = [0.01, 0.001, 0.0005]
    batch_sizes = [16, 32, 64]
    y_np = np.array([y.item() for y in y_seqs])

    best_acc, best_model, best_config = 0.0, None, {}

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"\nTraining with hidden_size={hidden_size}, lr={lr}, batch_size={batch_size}")

                # Split by IDs; test split is held out and used as validation here
                (train_X, train_y, _), (test_X, test_y, _) = split_sequence_dataset_by_id(
                    X_seqs, y_seqs, lengths, id_list
                )

                train_dataset = MalnutritionDataset(train_X, train_y)
                test_dataset = MalnutritionDataset(test_X, test_y)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

                model = LSTMModel(
                    input_size=X_seqs[0].shape[1],
                    hidden_size=hidden_size,
                    num_classes=len(np.unique(y_np)),
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                class_weights = compute_class_weight("balanced", classes=np.unique(y_np), y=y_np)
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
                loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

                # Train (early stop on the test split used as validation)
                model = train_model_with_early_stopping(
                    model, train_loader, test_loader, optimizer, loss_fn, num_epochs=100, patience=15
                )

                # Evaluate on test and train to check generalization
                model.eval()
                all_preds, all_labels = [], []
                train_preds, train_labels = [], []

                with torch.no_grad():
                    for batch_x, batch_len, batch_y in test_loader:
                        batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
                        logits = model(batch_x, batch_len)
                        preds = logits.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch_y.cpu().numpy())

                    for batch_x, batch_len, batch_y in train_loader:
                        batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
                        logits = model(batch_x, batch_len)
                        preds = logits.argmax(dim=1)
                        train_preds.extend(preds.cpu().numpy())
                        train_labels.extend(batch_y.cpu().numpy())

                acc_test = accuracy_score(all_labels, all_preds)
                acc_train = accuracy_score(train_labels, train_preds)

                print(f"Test Accuracy: {acc_test:.4f}")
                print(f"Train Accuracy: {acc_train:.4f}")

                if acc_test > best_acc:
                    best_acc = acc_test
                    best_model = deepcopy(model)
                    best_config = {"hidden_size": hidden_size, "learning_rate": lr, "batch_size": batch_size}

    print("\nBest Config:", best_config)
    print("Best Accuracy:", best_acc)

    # Final evaluation on the last test_loader used above (scope note)
    best_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_len, batch_y in test_loader:
            batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
            logits = best_model(batch_x, batch_len)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    return best_model, best_config


def permutation_feature_importance(
    model: nn.Module,
    dataset: Dataset,
    feature_idx: int,
    batch_size: int = 32,
) -> float:
    """Sequence-aware permutation importance by shuffling one feature across timesteps.

    Approach:
        - Compute baseline accuracy.
        - For each batch, clone inputs and permute a single feature column across the
          entire (B, T) grid, keeping batch/time alignment, then recompute accuracy.
        - Importance = baseline_acc - permuted_acc.

    Args:
        model: Trained classifier to evaluate.
        dataset: MalnutritionDataset to iterate over.
        feature_idx: Zero-based column index in the feature dimension (D).
        batch_size: Dataloader batch size.

    Returns:
        float: Accuracy drop due to permutation (higher => more important).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # ----- baseline -----
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_len, batch_y in loader:
            batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
            logits = model(batch_x, batch_len)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    baseline_acc = accuracy_score(all_labels, all_preds)

    # ----- permute one feature across all timesteps -----
    all_preds_perm, all_labels_perm = [], []
    with torch.no_grad():
        for batch_x, batch_len, batch_y in loader:
            batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
            batch_x_perm = batch_x.clone()

            # Flatten (B, T) slice for the feature, permute globally, then reshape back
            feat_values = batch_x_perm[:, :, feature_idx].flatten()
            permuted = feat_values[torch.randperm(feat_values.size(0))]
            batch_x_perm[:, :, feature_idx] = permuted.view(batch_x_perm.size(0), batch_x_perm.size(1))

            logits = model(batch_x_perm, batch_len)
            preds = logits.argmax(dim=1)
            all_preds_perm.extend(preds.cpu().numpy())
            all_labels_perm.extend(batch_y.cpu().numpy())
    permuted_acc = accuracy_score(all_labels_perm, all_preds_perm)

    importance = baseline_acc - permuted_acc
    return importance


if __name__ == "__main__":
    # Reproducibility and device setup
    set_seed(42)
    device = torch.device("cpu")

    # Example: load preprocessed pickle and build sequences
    df = pd.read_pickle("./datasets/mal_data.pkl")
    X_seqs, y_seqs, lengths, id_list = prepare_sequence_data(df)

    # Further training / search / evaluation examples are provided below
    # and can be uncommented as needed.

    # best_model, best_config = search_best_model(X_seqs, y_seqs, lengths, id_list)
    # (train_X, train_y, train_len), (test_X, test_y, test_len) = split_sequence_dataset_by_id(X_seqs, y_seqs, lengths, id_list)

    # hidden_size = 64
    # learning_rate = 0.0005
    # batch_size = 32

    # train_dataset = MalnutritionDataset(train_X, train_y)
    # test_dataset = MalnutritionDataset(test_X, test_y)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    # y_np = np.array([y.item() for y in y_seqs])
    
    # model = LSTMModel(input_size=X_seqs[0].shape[1], hidden_size=hidden_size, num_classes=len(np.unique(y_np))).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    # class_weights = compute_class_weight('balanced', classes=np.unique(y_np), y=y_np)
    # weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

    # model = train_model_with_early_stopping(model, train_loader, test_loader, optimizer, loss_fn, num_epochs=100, patience=15)

    # evaluate_model(model, test_loader)

    # feature_num = X_seqs[0].shape[1]
    # importances = []
    # for i in range(feature_num):
    #     imp = permutation_feature_importance(model, test_dataset, i)
    #     importances.append(imp)

    # feature_cols = [col for col in df.columns if col not in ['IDno', 'Assessment_Date', 'CAP_Nutrition']]

    # sorted_idx = np.argsort(importances)[::-1]
    # for i in sorted_idx[:10]:
    #     print(f"Feature {feature_cols[i]} importance: {importances[i]:.4f}")


