import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from copy import deepcopy
import random
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.preprocessing import StandardScaler


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_sequence_data(df, id_col='IDno', time_col='Assessment_Date', target_col='CAP_Nutrition'):
    df['Assessment_Date'] = pd.to_datetime(df['Assessment_Date'], format="%d%b%Y")
    feature_cols = [col for col in df.columns if col not in [id_col, time_col, target_col]]
    print("The original df shape is:", df.shape)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X_seqs, y_seqs, lengths, id_list = [], [], [], []

    for pid, group in df.groupby(id_col):
        group_sorted = group.sort_values(by=time_col)
        X = torch.tensor(group_sorted[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(group_sorted[target_col].values[-1], dtype=torch.long)
        X_seqs.append(X)
        y_seqs.append(y)
        lengths.append(X.shape[0])
        id_list.append(pid)

    return X_seqs, y_seqs, lengths, id_list


def split_sequence_dataset_by_id(X_seqs, y_seqs, lengths, id_list, test_size=0.2, random_state=42):
    unique_ids = list(set(id_list))
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state, stratify=y_seqs)

    X_train, y_train, len_train = [], [], []
    X_test, y_test, len_test = [], [], []

    for i, idno in enumerate(id_list):
        if idno in train_ids:
            X_train.append(X_seqs[i])
            y_train.append(y_seqs[i])
            len_train.append(lengths[i])
        else:
            X_test.append(X_seqs[i])
            y_test.append(y_seqs[i])
            len_test.append(lengths[i])

    return (X_train, y_train, len_train), (X_test, y_test, len_test)

class MalnutritionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True)  # shape: (batch, max_seq, features)
    labels = torch.stack(labels)
    return padded_seqs, lengths, labels

class MalnutritionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True)  # shape: (batch, max_seq, features)
    labels = torch.stack(labels)
    return padded_seqs, lengths, labels


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (ht, ct) = self.lstm(packed)
        out = self.fc(self.dropout(self.norm(torch.cat([ht[-2], ht[-1]], dim=1))))
        return out  


def train_model_with_early_stopping(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=100, patience=5):
    best_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch_x, batch_len, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x, batch_len)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_len, batch_y in val_loader:
                logits = model(batch_x, batch_len)
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

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

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_len, batch_y in dataloader:
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
    hidden_sizes = [32, 64, 128, 256]
    learning_rates = [0.01, 0.001, 0.0005]
    batch_sizes = [16, 32, 64]


    best_acc = 0
    best_model = None
    best_config = {}

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"\nTraining with hidden_size={hidden_size}, lr={lr}, batch_size={batch_size}")

                # Split into train and test
                (train_X_full, train_y_full, _), (test_X, test_y, _) = split_sequence_dataset_by_id(X_seqs, y_seqs, lengths, id_list)

                # Further split train into train + val
                train_X, val_X, train_y, val_y = train_test_split(train_X_full, train_y_full, test_size=0.2, random_state=42, stratify=train_y_full)

                train_dataset = MalnutritionDataset(train_X, train_y)
                val_dataset = MalnutritionDataset(val_X, val_y)
                test_dataset = MalnutritionDataset(test_X, test_y)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

                # Create model, optimizer, and loss function
                model = LSTMModel(input_size=X_seqs[0].shape[1], hidden_size=hidden_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                y_np = np.array([y.item() for y in y_seqs])
                class_weights = compute_class_weight('balanced', classes=np.unique(y_np), y=y_np)
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
                loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

                # Train with early stopping
                model = train_model_with_early_stopping(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=100, patience=15)

                # Evaluate on test set
                model.eval()
                all_preds, all_labels = [], []
                train_preds, train_labels = [], []

                with torch.no_grad():
                    for batch_x, batch_len, batch_y in test_loader:
                        logits = model(batch_x, batch_len)
                        preds = logits.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch_y.cpu().numpy())

                    for batch_x, batch_len, batch_y in train_loader:
                        logits = model(batch_x, batch_len)
                        preds = logits.argmax(dim=1)
                        train_preds.extend(preds.cpu().numpy())
                        train_labels.extend(batch_y.cpu().numpy())

                acc_test = accuracy_score(all_labels, all_preds)
                acc_train = accuracy_score(train_labels, train_preds)

                print(f"Test Accuracy: {acc_test:.4f}")
                print(f"Train Accuracy: {acc_train:.4f}")

                # Save best model
                if acc_test > best_acc:
                    best_acc = acc_test
                    best_model = deepcopy(model)
                    best_config = {
                        'hidden_size': hidden_size,
                        'learning_rate': lr,
                        'batch_size': batch_size
                    }

    print("\nBest Config:", best_config)
    print("Best Accuracy:", best_acc)
    best_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_len, batch_y in test_loader:
            logits = best_model(batch_x, batch_len)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # print final evaluation metrics
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    return best_model, best_config



if __name__ == "__main__":
    set_seed(42)
    df = pd.read_pickle("./datasets/cap_data.pkl")
    X_seqs, y_seqs, lengths, id_list = prepare_sequence_data(df)
    best_model, best_config = search_best_model(X_seqs, y_seqs, lengths, id_list)
    # (train_X, train_y, train_len), (test_X, test_y, test_len) = split_sequence_dataset_by_id(X_seqs, y_seqs, lengths, id_list)

    # train_dataset = MalnutritionDataset(train_X, train_y)
    # test_dataset = MalnutritionDataset(test_X, test_y)

    # train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)
    
    # model = LSTMModel(input_size=X_seqs[0].shape[1])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = nn.CrossEntropyLoss()
    # train_model(model, train_loader, optimizer, loss_fn, num_epochs=100)

    # evaluate_model(model, test_loader)

