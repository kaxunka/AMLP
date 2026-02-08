import argparse
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from typing import List, Tuple
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             average_precision_score)
from sklearn.model_selection import train_test_split


# ========================= 1. Configuration =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


# ========================= 2. Dataset Logic =========================

class AntimicrobialDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int], tokenizer, max_seq_len: int = 100):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        # ProtBERT requires space-separated amino acid sequences
        sequence = ' '.join(list(self.sequences[idx]))
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ========================= 3. Data Pipeline =========================

def load_and_preprocess_data(data_path: str, tokenizer, args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    df = pd.read_csv(data_path)

    # Validation
    required_cols = ['Sequence', 'Label', 'Length']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV missing required columns: {required_cols}")

    # Stratified split based on Label and sequence length binning
    bins = list(range(0, 200, 10))
    df["Length_Bin"] = pd.cut(df["Length"], bins=bins, right=False)

    # 70/15/15 Split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df[["Label", "Length_Bin"]], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[["Label", "Length_Bin"]], random_state=42
    )

    def create_loader(d, shuffle=False):
        ds = AntimicrobialDataset(d["Sequence"].tolist(), d["Label"].tolist(), tokenizer, args.max_seq_len)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    return create_loader(train_df, True), create_loader(val_df), create_loader(test_df)


# ========================= 4. Model Architecture =========================

class ProtBERTClassifier(nn.Module):
    def __init__(self, model_name: str = "Rostlab/prot_bert", unfreeze_layers: int = 8):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._set_trainable_layers(unfreeze_layers)

    def _set_trainable_layers(self, unfreeze_layers):
        total_layers = self.model.config.num_hidden_layers

        # Freeze all backbone parameters
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        # Unfreeze top N transformer layers
        layers_to_train = range(total_layers - unfreeze_layers, total_layers)
        for i in layers_to_train:
            for param in self.model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Always train Pooler and Classifier head
        for param in self.model.bert.pooler.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model initialized. Trainable parameters: {trainable:,}")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# ========================= 5. Metrics =========================

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    probs = torch.softmax(torch.tensor(preds), dim=1)[:, 1].numpy()
    pred_labels = np.argmax(preds, axis=1)

    return {
        "acc": accuracy_score(labels, pred_labels),
        "mcc": matthews_corrcoef(labels, pred_labels),
        "f1": f1_score(labels, pred_labels),
        "auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "recall": recall_score(labels, pred_labels),
        "precision": precision_score(labels, pred_labels)
    }


# ========================= 6. Training Engine =========================

def run_training(args):
    set_seed()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    model = ProtBERTClassifier(unfreeze_layers=args.unfreeze_layers).to(device)
    train_loader, val_loader, test_loader = load_and_preprocess_data(args.data_path, model.tokenizer, args)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    best_mcc = -1.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_preds, train_labels = 0, [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch[
                'label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += outputs.loss.item()
            train_preds.append(outputs.logits.detach().cpu().numpy())
            train_labels.append(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        train_metrics = compute_metrics(np.concatenate(train_preds), np.concatenate(train_labels))

        # --- Validation Phase ---
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch[
                    'label'].to(device)
                outputs = model(input_ids, mask, labels=labels)
                val_loss += outputs.loss.item()
                val_preds.append(outputs.logits.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_metrics = compute_metrics(np.concatenate(val_preds), np.concatenate(val_labels))

        # Logging & Checkpointing
        print(
            f"\nEpoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val MCC: {val_metrics['mcc']:.4f}")

        writer.add_scalar("MCC/Val", val_metrics['mcc'], epoch)
        scheduler.step(val_metrics['mcc'])

        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            torch.save(model.model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # --- Final Evaluation ---
    model.model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            test_preds.append(outputs.logits.cpu().numpy())
            test_labels.append(batch['label'].cpu().numpy())

    test_metrics = compute_metrics(np.concatenate(test_preds), np.concatenate(test_labels))
    pd.DataFrame([test_metrics]).to_csv(os.path.join(args.output_dir, "test_results.csv"), index=False)
    print(f"\nTest Set Results | MCC: {test_metrics['mcc']:.4f} | AUC: {test_metrics['auc']:.4f}")
    writer.close()


# ========================= 7. Entry Point =========================

def parse_args():
    parser = argparse.ArgumentParser(description="ProtBERT Fine-tuning for Linear AMPs")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--unfreeze_layers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())