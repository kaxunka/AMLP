import argparse, os, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

# ---------- Basic configuration ----------
def get_device():
    """Return the available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_chain_vocab():
    """Create a vocabulary for fatty acid chain types."""
    chains = ["C6", "C8", "C10", "C12", "C14", "C16", "C18"]
    vocab = {c: i for i, c in enumerate(chains)}
    vocab['<UNK>'] = len(vocab)
    return vocab

# ---------- Dataset for prediction (no labels required) ----------
class LipopeptideDataset(Dataset):
    def __init__(self, df, tokenizer, chain_vocab, max_len=40):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.chain_vocab = chain_vocab
        self.max_len = max_len
        self.unk = chain_vocab['<UNK>']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = ' '.join(list(row['Sequence']))
        chain_idx = self.chain_vocab.get(row['Fatty_acid_chain'], self.unk)
        enc = self.tokenizer(seq, truncation=True, padding='max_length',
                             max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'chain_index': torch.tensor(chain_idx, dtype=torch.long)
        }

# ---------- Model definition (identical to training) ----------
class StrictChainAsPromptModel(nn.Module):
    def __init__(self, chain_vocab_size, reduced_dim=32, dropout=0.3,
                 model_name="Rostlab/prot_bert"):
        super().__init__()
        self.peptide_encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.peptide_encoder.config.hidden_size
        for p in self.peptide_encoder.parameters():
            p.requires_grad = False
        self.seq_projector = nn.Sequential(
            nn.Linear(self.hidden_size, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.Dropout(dropout)
        )
        self.chain_embed = nn.Embedding(chain_vocab_size, reduced_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=reduced_dim, nhead=2, dim_feedforward=reduced_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.interaction = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.classifier = nn.Linear(reduced_dim, 2)

    def forward(self, input_ids, attention_mask, chain_index):
        bsz = input_ids.size(0)
        with torch.no_grad():
            bert_out = self.peptide_encoder(input_ids, attention_mask).last_hidden_state
        seq = self.seq_projector(bert_out)
        prompt = self.chain_embed(chain_index).unsqueeze(1)
        x = torch.cat([prompt, seq], dim=1)
        pmask = torch.ones((bsz, 1), device=input_ids.device, dtype=attention_mask.dtype)
        mask = torch.cat([pmask, attention_mask], dim=1)
        key_pad = (mask == 0)
        x = self.interaction(x, src_key_padding_mask=key_pad)
        cls = x[:, 0, :]
        return self.classifier(cls)

# ---------- Load weights with backward compatibility ----------
def load_model_with_fix(model, path, device):
    state = torch.load(path, map_location=device)
    fixed = {}
    for k, v in state.items():
        k = k.replace("chain_prompt_embedding", "chain_embed")
        k = k.replace("interaction_layer", "interaction")
        if k.startswith("classifier.1."):
            k = k.replace("classifier.1.", "classifier.")
        fixed[k] = v
    model.load_state_dict(fixed, strict=False)
    return model

# ---------- Ensemble inference ----------
def ensemble_predict(models, loader, device):
    all_probs = []
    with torch.no_grad():
        for model in models:
            model.eval()
            probs = []
            for batch in loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                chain = batch['chain_index'].to(device)
                logits = model(ids, mask, chain)
                prob = torch.softmax(logits, dim=1)[:, 1]
                probs.extend(prob.cpu().numpy())
            all_probs.append(np.array(probs))
    avg_prob = np.mean(all_probs, axis=0)
    return avg_prob

# ---------- Main prediction function ----------
def main(args):
    device = get_device()
    # Read input file (CSV or Excel)
    input_path = args.input
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    # Verify required columns
    if 'Sequence' not in df.columns or 'Fatty_acid_chain' not in df.columns:
        raise ValueError("Input file must contain 'Sequence' and 'Fatty_acid_chain' columns.")

    chain_vocab = create_chain_vocab()
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    dataset = LipopeptideDataset(df, tokenizer, chain_vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load 5-fold ensemble models
    models = []
    for fold in range(5):
        path = os.path.join(args.model_dir, f"model_fold{fold}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model = StrictChainAsPromptModel(
            chain_vocab_size=len(chain_vocab),
            reduced_dim=args.reduced_dim,
            dropout=args.dropout
        ).to(device)
        load_model_with_fix(model, path, device)
        models.append(model)
        print(f"Loaded fold {fold} model.")

    # Run ensemble prediction
    avg_prob = ensemble_predict(models, loader, device)

    # Add predicted probability and binary label (threshold = 0.5)
    df['Predicted Probability'] = avg_prob
    df['Predicted Label'] = (avg_prob >= 0.5).astype(int)  # 1: antimicrobial, 0: non-antimicrobial

    # Save output (same format as input if not specified)
    output_path = args.output
    if not output_path:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base}_predictions.csv" if input_path.endswith('.csv') else f"{base}_predictions.xlsx"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)

    print(f"Prediction finished. Output saved to {output_path}")
    print(f"Number of sequences predicted as antimicrobial (≥0.5): {(df['Predicted Label'] == 1).sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMLP prediction script")
    parser.add_argument("--input", required=True, help="Path to input CSV or Excel file (must contain 'Sequence' and 'Fatty_acid_chain')")
    parser.add_argument("--model_dir", required=True, help="Directory containing best_model_fold0.pth ~ best_model_fold4.pth")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--reduced_dim", type=int, default=32, help="Reduced dimension used during training")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate used during training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()
    main(args)
