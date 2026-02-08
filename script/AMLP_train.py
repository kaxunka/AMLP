import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, roc_auc_score, matthews_corrcoef,
                             average_precision_score, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer


# ========================= 1. Base Configuration =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= 2. Dataset Definition =========================

class MixedPeptideDataset(Dataset):
    """Dataset handling both lipopeptides (with fatty acid chains) and linear peptides."""

    def __init__(self, lipo_df, linear_df, tokenizer, chain_vocab, max_len=40):
        self.lipo_data = lipo_df.reset_index(drop=True)
        self.linear_data = linear_df.reset_index(drop=True)
        self.lipo_len = len(self.lipo_data)
        self.total_len = self.lipo_len + len(self.linear_data)

        self.tokenizer = tokenizer
        self.chain_vocab = chain_vocab
        self.max_len = max_len
        self.unk_idx = chain_vocab.get('<UNK>', 0)
        self.null_idx = chain_vocab.get('NULL', 0)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.lipo_len:
            # Lipopeptide sample
            row = self.lipo_data.iloc[idx]
            chain_str = row['Fatty_acid_chain']
            chain_idx = self.chain_vocab.get(chain_str, self.unk_idx)
            is_lipo = 1.0
        else:
            # Linear peptide sample
            row = self.linear_data.iloc[idx - self.lipo_len]
            chain_idx = self.null_idx
            is_lipo = 0.0

        seq = ' '.join(list(row['Sequence']))
        label = int(row['Label'])

        encoding = self.tokenizer(
            seq, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'chain_index': torch.tensor(chain_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'is_lipo': torch.tensor(is_lipo, dtype=torch.float)
        }


def create_chain_vocab(df):
    """Creates a vocabulary for fatty acid chains, adding NULL for linear peptides."""
    chains = sorted(df['Fatty_acid_chain'].unique())
    vocab = {c: i for i, c in enumerate(chains)}
    vocab['<UNK>'] = len(vocab)
    vocab['NULL'] = len(vocab)
    return vocab


# ========================= 3. Model Architecture =========================

class LipopeptideConsistencyModel(nn.Module):
    def __init__(self, pretrain_path, chain_vocab_size,
                 model_name="Rostlab/prot_bert",
                 embedding_dim=128,
                 consistency_weight=3.0,
                 noise_level=0.1):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.noise_level = noise_level

        # Load backbone (Stage 2 fine-tuned model)
        self.peptide_encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.peptide_encoder.config.hidden_size
        self._load_pretrained_weights(pretrain_path)

        # Freeze Backbone to preserve general AMP features
        for param in self.peptide_encoder.parameters():
            param.requires_grad = False
        print("[Model] Backbone Frozen.")

        # Chain-as-Prompt (CaP) components
        self.chain_embedding = nn.Embedding(chain_vocab_size, embedding_dim)
        self.chain_projector = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        self.null_prompt = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        # Bottleneck Adapter
        self.adapter_down = nn.Linear(self.hidden_size, 256)
        self.adapter_act = nn.GELU()
        self.adapter_up = nn.Linear(256, self.hidden_size)
        self.adapter_norm = nn.LayerNorm(self.hidden_size)

        # Head components
        self.attention_pool = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def _load_pretrained_weights(self, path):
        if os.path.exists(path):
            state = torch.load(path, map_location='cpu')
            if hasattr(state, 'state_dict'): state = state.state_dict()
            new_state = {k.replace('model.', '').replace('base_model.', '').replace('peptide_encoder.', ''): v
                         for k, v in state.items()}
            self.peptide_encoder.load_state_dict(new_state, strict=False)
            print(f"[Model] Loaded Stage-2 weights from {path}")
        else:
            print(f"[Model] Warning: Pretrain path {path} not found.")

    def _forward_adapter(self, x):
        return self.adapter_norm(self.adapter_up(self.adapter_act(self.adapter_down(x))) + x)

    def _get_pooled_feature(self, features, mask):
        attn_scores = self.attention_pool(features)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        return torch.sum(features * attn_weights, dim=1)

    def forward(self, input_ids, attention_mask, chain_index, labels=None, is_lipo=None):
        batch_size = input_ids.size(0)

        with torch.no_grad():
            bert_out = self.peptide_encoder(input_ids, attention_mask).last_hidden_state

        if self.training and self.noise_level > 0:
            bert_out = bert_out + torch.randn_like(bert_out) * self.noise_level

        # Stream 1: Full Input (Prompt + Sequence)
        real_prompt = self.chain_projector(self.chain_embedding(chain_index)).unsqueeze(1)
        prompt_mask = torch.ones((batch_size, 1), device=input_ids.device)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        full_input = torch.cat([real_prompt, bert_out], dim=1)
        z_full = self._get_pooled_feature(self._forward_adapter(full_input), extended_mask)
        logits = self.classifier(z_full)

        # Stream 2: Consistency Loss (calculated for lipopeptides only)
        const_loss = torch.tensor(0.0, device=input_ids.device)
        if self.training and is_lipo is not None and is_lipo.sum() > 0:
            null_prompt_expand = self.null_prompt.expand(batch_size, -1, -1)
            seq_input = torch.cat([null_prompt_expand, bert_out], dim=1)
            z_seq = self._get_pooled_feature(self._forward_adapter(seq_input), extended_mask)

            raw_mse = F.mse_loss(z_full, z_seq, reduction='none').mean(dim=1)
            const_loss = (raw_mse * is_lipo).sum() / (is_lipo.sum() + 1e-8)
            const_loss = const_loss * self.consistency_weight

        return logits, const_loss


# ========================= 4. Training Utilities =========================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_cls = 0, 0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids, mask, chain = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch[
            'chain_index'].to(device)
        label, is_lipo = batch['label'].to(device), batch['is_lipo'].to(device)

        optimizer.zero_grad()
        logits, const_loss = model(input_ids, mask, chain, labels=label, is_lipo=is_lipo)
        cls_loss = criterion(logits[:, 1], label)
        loss = cls_loss + const_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
    return total_loss / len(loader), total_cls / len(loader)


def evaluate(model, loader, device, only_lipo=True):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            if only_lipo and batch['is_lipo'].sum() == 0: continue

            logits, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device),
                              batch['chain_index'].to(device))
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            if only_lipo:
                mask = batch['is_lipo'].bool()
                if mask.sum() > 0:
                    preds.extend(pred[mask].cpu().numpy());
                    labels.extend(batch['label'][mask].cpu().numpy());
                    probs.extend(prob[mask].cpu().numpy())
            else:
                preds.extend(pred.cpu().numpy());
                labels.extend(batch['label'].cpu().numpy());
                probs.extend(prob.cpu().numpy())

    if not labels: return {"mcc": 0, "acc": 0}
    return {
        "mcc": matthews_corrcoef(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    }


# ========================= 5. Main Execution =========================

def main(args):
    set_seed()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    print(">>> Loading Data...")
    lipo_df = pd.read_csv(args.lipo_train_path)
    linear_df = pd.read_csv(args.linear_train_path)
    chain_vocab = create_chain_vocab(pd.read_csv(args.lipo_full_path))
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

    lipo_df['stratify_key'] = lipo_df['Fatty_acid_chain'] + '_' + lipo_df['Label'].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(lipo_df, lipo_df['stratify_key'])):
        print(f"\n>>> Fold {fold + 1}/5")
        lipo_train = lipo_df.iloc[train_idx].copy()
        lipo_val = lipo_df.iloc[val_idx]

        # Weight balancing
        lipo_train['sample_group'] = lipo_train['stratify_key']
        linear_copy = linear_df.copy();
        linear_copy['sample_group'] = 'Linear_' + linear_copy['Label'].astype(str)
        concat_df = pd.concat([lipo_train, linear_copy], ignore_index=True)
        counts = concat_df['sample_group'].value_counts().to_dict()
        weights = [1.0 / counts[g] for g in concat_df['sample_group']]

        sampler = WeightedRandomSampler(weights, num_samples=len(concat_df), replacement=True)
        train_loader = DataLoader(MixedPeptideDataset(lipo_train, linear_copy, tokenizer, chain_vocab),
                                  batch_size=args.batch_size, sampler=sampler)
        val_loader = DataLoader(MixedPeptideDataset(lipo_val, pd.DataFrame(), tokenizer, chain_vocab),
                                batch_size=args.batch_size)

        model = LipopeptideConsistencyModel(args.pretrain_model_path, len(chain_vocab),
                                            consistency_weight=args.consistency_weight,
                                            noise_level=args.noise_level).to(device)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        criterion = FocalLoss(alpha=0.7)

        best_mcc = -1.0
        patience_cnt = 0
        for epoch in range(args.epochs):
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, device)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Ep {epoch + 1}: Loss={train_loss:.4f} | Val MCC={val_metrics['mcc']:.4f}")

            scheduler.step(val_metrics['mcc'])
            if val_metrics['mcc'] > best_mcc:
                best_mcc = val_metrics['mcc']
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_fold_{fold}.pth"))
                patience_cnt = 0
            elif (patience_cnt := patience_cnt + 1) >= args.patience:
                break
        fold_results.append(best_mcc)

    print(f"\nCross-Validation Average MCC: {np.mean(fold_results):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lipo_full_path", type=str, required=True, help="Path to full 477 lipopeptide file for vocab")
    parser.add_argument("--lipo_train_path", type=str, required=True, help="Path to lipo_train_val.csv")
    parser.add_argument("--linear_train_path", type=str, required=True, help="Path to 50k linear peptide dataset")
    parser.add_argument("--pretrain_model_path", type=str, required=True, help="Path to Stage-2 pre-trained pth")
    parser.add_argument("--output_dir", type=str, default="results_final")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--consistency_weight", type=float, default=3.0)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    main(parser.parse_args())