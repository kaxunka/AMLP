import argparse, os, random, sys, torch, torch.nn as nn, pandas as pd, numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.tensorboard import SummaryWriter
from sklearn.isotonic import IsotonicRegression
import joblib

# ---------- Basic Configuration ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): pass

# Vocabulary
def create_chain_vocab():
    chains = ["C6","C8","C10","C12","C14","C16","C18"]
    vocab = {c:i for i,c in enumerate(chains)}
    vocab['<UNK>'] = len(vocab)
    return vocab

# ---------- Chain grouping (for stratification) ----------
def get_chain_group(chain_str):
    try:
        num = int(chain_str[1:])
    except:
        return 'mid'
    if num <= 10: return 'short'
    elif num <= 14: return 'mid'
    else: return 'long'

# ---------- Data augmentation: conservative amino acid substitutions ----------
CONSERVATIVE_SUB = {
    'K': ['R'], 'R': ['K'], 'L': ['I'], 'I': ['L'],
    'A': ['V'], 'V': ['A'], 'F': ['Y'], 'Y': ['F'],
    'E': ['D'], 'D': ['E'], 'W': ['F']
}
def conservative_augment(seq, prob=0.05):
    if random.random() > prob: return seq
    seq = list(seq)
    idx = random.randint(0, len(seq)-1)
    aa = seq[idx]
    if aa in CONSERVATIVE_SUB:
        seq[idx] = random.choice(CONSERVATIVE_SUB[aa])
    return ''.join(seq)

# ---------- Lipopeptide dataset (with augmentation) ----------
class LipopeptideDataset(Dataset):
    def __init__(self, df, tokenizer, chain_vocab, max_len=40, augment=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer; self.chain_vocab = chain_vocab
        self.max_len = max_len; self.unk = chain_vocab['<UNK>']
        self.augment = augment
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['Sequence']
        if self.augment:
            seq = conservative_augment(seq, prob=0.05)
        seq = ' '.join(list(seq))
        chain_idx = self.chain_vocab.get(row['Fatty_acid_chain'], self.unk)
        label = int(row['Label'])
        enc = self.tokenizer(seq, truncation=True, padding='max_length',
                             max_length=self.max_len, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'chain_index': torch.tensor(chain_idx, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)}

class StrictChainAsPromptModel(nn.Module):
    def __init__(self, pretrain_path, chain_vocab_size, reduced_dim=32, dropout=0.5,
                 model_name="Rostlab/prot_bert"):
        super().__init__()
        # Fix ProtBERT configuration for Transformers >= 4.51
        config = AutoConfig.from_pretrained(model_name)
        if not hasattr(config, "model_type") or config.model_type is None:
            config.model_type = "bert"
        self.peptide_encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = self.peptide_encoder.config.hidden_size
        self._load_pretrained_weights(pretrain_path)
        for p in self.peptide_encoder.parameters():
            p.requires_grad = False
        self.seq_projector = nn.Sequential(
            nn.Linear(self.hidden_size, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.Dropout(dropout)
        )
        self.chain_embed = nn.Embedding(chain_vocab_size, reduced_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=reduced_dim, nhead=2, dim_feedforward=reduced_dim*2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.interaction = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, 2)
        )

    def _load_pretrained_weights(self, path):
        if path and os.path.exists(path):
            state = torch.load(path, map_location='cpu')
            if hasattr(state, 'state_dict'): state = state.state_dict()
            new_state = {k.replace('model.','').replace('base_model.','').replace('peptide_encoder.',''): v
                         for k, v in state.items()}
            self.peptide_encoder.load_state_dict(new_state, strict=False)
            print(f"[Model] Loaded LinAMP-BERT weights from {path}")
        else:
            print("[Model] Weights not found, using original ProtBERT.")

    def forward(self, input_ids, attention_mask, chain_index):
        bsz = input_ids.size(0)
        with torch.no_grad():
            bert_out = self.peptide_encoder(input_ids, attention_mask).last_hidden_state
        seq = self.seq_projector(bert_out)
        prompt = self.chain_embed(chain_index).unsqueeze(1)
        x = torch.cat([prompt, seq], dim=1)
        pmask = torch.ones((bsz,1), device=input_ids.device, dtype=attention_mask.dtype)
        mask = torch.cat([pmask, attention_mask], dim=1)
        key_pad = (mask==0)
        x = self.interaction(x, src_key_padding_mask=key_pad)
        cls = x[:,0,:]
        return self.classifier(cls)

# ---------- Evaluation (AUC) ----------
def evaluate_auc(model, loader, device):
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(device),
                           batch['attention_mask'].to(device),
                           batch['chain_index'].to(device))
            prob = torch.softmax(logits, dim=1)[:,1]
            labels.extend(batch['label'].cpu().numpy())
            probs.extend(prob.cpu().numpy())
    labels = np.array(labels); probs = np.array(probs)
    if len(np.unique(labels))<2: return 0.5
    return roc_auc_score(labels, probs)


def main(args):
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training.log")
    sys.stdout = Logger(log_file)

    lipo_df = pd.read_csv(args.lipo_path)
    print(f"Total lipopeptides: {len(lipo_df)}")

    chain_vocab = create_chain_vocab()
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

    # Stratification key: chain_group + Label
    lipo_df['chain_group'] = lipo_df['Fatty_acid_chain'].apply(get_chain_group)
    lipo_df['stratify_key'] = lipo_df['chain_group'] + '_' + lipo_df['Label'].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_aucs = []
    all_val_probs, all_val_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(lipo_df, lipo_df['stratify_key'])):
        print(f"\n{'='*40}\n>>> Fold {fold+1}/5")
        train_df = lipo_df.iloc[train_idx].copy()
        val_df = lipo_df.iloc[val_idx].copy()

        # Enable augmentation on the training set only
        train_loader = DataLoader(
            LipopeptideDataset(train_df, tokenizer, chain_vocab, augment=True),
            batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(
            LipopeptideDataset(val_df, tokenizer, chain_vocab, augment=False),
            batch_size=args.batch_size, shuffle=False)

        # Initialize the Chain-as-Prompt model
        model = StrictChainAsPromptModel(
            pretrain_path=args.pretrain_model,
            chain_vocab_size=len(chain_vocab),
            reduced_dim=args.reduced_dim,
            dropout=args.dropout
        ).to(device)

        # Class weights to handle imbalance (approx. 2:1)
        counts = train_df['Label'].value_counts().to_dict()
        n_pos = counts.get(1,1); n_neg = counts.get(0,1)
        weight_neg = n_pos / n_neg
        weight_pos = 1.0
        class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, weight_decay=args.weight_decay)

        # Cosine annealing with warm restarts scheduler
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

        best_auc = 0.0; best_state = None; patience_cnt = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                chain = batch['chain_index'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                logits = model(ids, mask, chain)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            scheduler.step(epoch)

            val_auc = evaluate_auc(model, val_loader, device)
            train_auc = evaluate_auc(model, DataLoader(
                LipopeptideDataset(train_df, tokenizer, chain_vocab, augment=False),
                batch_size=args.batch_size, shuffle=False), device)

            print(f"Ep {epoch+1:02d} | Loss: {avg_loss:.4f} | Tr AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(args.output_dir, f"model_fold{fold}.pth"))
        fold_aucs.append(best_auc)

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['input_ids'].to(device),
                               batch['attention_mask'].to(device),
                               batch['chain_index'].to(device))
                prob = torch.softmax(logits, dim=1)[:,1]
                all_val_probs.extend(prob.cpu().numpy())
                all_val_labels.extend(batch['label'].cpu().numpy())

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"\n{'='*40}\nCross-validation AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    all_val_labels = np.array(all_val_labels)
    all_val_probs = np.array(all_val_probs)

    best_thr, best_mcc = 0.5, -1.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (all_val_probs >= thr).astype(int)
        mcc = matthews_corrcoef(all_val_labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc; best_thr = thr


    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lipo_path", required=True, help="Path to lipopeptide training set")
    parser.add_argument("--pretrain_model", required=True, help="Path to LinAMP-BERT weights")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reduced_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    main(args)
