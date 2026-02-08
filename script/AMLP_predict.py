import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import glob
import warnings

warnings.filterwarnings("ignore")

# ========================= 1. Configuration =========================

VALID_CHAINS = ['C0', 'C6', 'C8', 'C10', 'C12', 'C14', 'C16', 'C18']


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_standard_chain_vocab():
    """Mapping logic consistent with training: C0 is treated as NULL."""
    chains = ['C6', 'C8', 'C10', 'C12', 'C14', 'C16', 'C18']
    sorted_chains = sorted(chains)
    vocab = {c: i for i, c in enumerate(sorted_chains)}
    vocab['<UNK>'] = len(vocab)
    vocab['NULL'] = len(vocab)
    return vocab


# ========================= 2. Dataset Logic =========================

class LipopeptideInferenceDataset(Dataset):
    def __init__(self, df, tokenizer, chain_vocab, max_len=100):
        self.tokenizer = tokenizer
        self.chain_vocab = chain_vocab
        self.max_len = max_len
        self.unk_idx = chain_vocab.get('<UNK>', 0)
        self.samples = []

        for idx, row in df.iterrows():
            seq = str(row['Sequence']).strip()
            # Updated from 'Chain' to 'Fatty_acid_chain'
            chain = str(row['Fatty_acid_chain']).strip().upper()

            if chain not in VALID_CHAINS:
                print(f"Warning: Row {idx} has invalid fatty acid chain '{chain}', skipping.")
                continue

            # C0 (linear peptide behavior) maps to NULL
            real_chain_str = 'NULL' if chain == 'C0' else chain

            self.samples.append({
                'Sequence': seq,
                'Input_Chain': chain,
                'Real_Chain_Str': real_chain_str,
                'Original_Index': idx
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        fmt_seq = ' '.join(list(item['Sequence']))
        encoding = self.tokenizer(
            fmt_seq, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        chain_idx = self.chain_vocab.get(item['Real_Chain_Str'], self.unk_idx)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'chain_index': torch.tensor(chain_idx, dtype=torch.long),
            'Original_Index': item['Original_Index']
        }


# ========================= 3. Model Architecture =========================

class LipopeptideConsistencyModel(nn.Module):
    """Inference version of the consistency model."""

    def __init__(self, chain_vocab_size, model_name="Rostlab/prot_bert", embedding_dim=128):
        super().__init__()
        self.peptide_encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.peptide_encoder.config.hidden_size

        self.chain_embedding = nn.Embedding(chain_vocab_size, embedding_dim)
        self.chain_projector = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        self.adapter_down = nn.Linear(self.hidden_size, 256)
        self.adapter_act = nn.GELU()
        self.adapter_up = nn.Linear(256, self.hidden_size)
        self.adapter_norm = nn.LayerNorm(self.hidden_size)

        self.attention_pool = nn.Sequential(nn.Linear(self.hidden_size, 1), nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def _forward_adapter(self, x):
        return self.adapter_norm(self.adapter_up(self.adapter_act(self.adapter_down(x))) + x)

    def _get_pooled_feature(self, features, mask):
        attn_scores = self.attention_pool(features)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        return torch.sum(features * attn_weights, dim=1)

    def forward(self, input_ids, attention_mask, chain_index):
        bert_out = self.peptide_encoder(input_ids, attention_mask).last_hidden_state
        prompt = self.chain_projector(self.chain_embedding(chain_index)).unsqueeze(1)

        prompt_mask = torch.ones((input_ids.size(0), 1), device=input_ids.device)
        extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        full_input = torch.cat([prompt, bert_out], dim=1)
        z = self._get_pooled_feature(self._forward_adapter(full_input), extended_mask)
        return self.classifier(z)


# ========================= 4. Inference Engine =========================

def main(args):
    device = get_device()
    chain_vocab = get_standard_chain_vocab()
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

    print(f"Loading input: {args.input_file}")
    df = pd.read_csv(args.input_file) if args.input_file.endswith('.csv') else pd.read_excel(args.input_file)

    # Validate columns
    required = ['Sequence', 'Fatty_acid_chain']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from input file.")

    dataset = LipopeptideInferenceDataset(df, tokenizer, chain_vocab, max_len=100)
    if not dataset:
        print("No valid sequences to process.")
        return
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Locate fold models
    model_paths = sorted(glob.glob(os.path.join(args.model_dir, "model_fold_*.pth")))
    if not model_paths:
        raise FileNotFoundError(f"No .pth files found in {args.model_dir}")

    all_fold_probs = np.zeros((len(dataset), len(model_paths)))

    # Ensemble Inference
    for i, path in enumerate(model_paths):
        print(f"Running Fold {i + 1}/{len(model_paths)}: {os.path.basename(path)}")
        model = LipopeptideConsistencyModel(len(chain_vocab)).to(device)

        # Load weights
        state = torch.load(path, map_location=device)
        # Handle different saving formats (Stage 2/3 variants)
        new_state = {k.replace('model.', '').replace('base_model.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                logits = model(batch['input_ids'].to(device),
                               batch['attention_mask'].to(device),
                               batch['chain_index'].to(device))
                fold_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        all_fold_probs[:, i] = fold_probs
        del model
        torch.cuda.empty_cache()

    # Aggregate results (Mean ensemble)
    mean_probs = np.mean(all_fold_probs, axis=1)

    results = []
    for idx, prob in enumerate(mean_probs):
        orig_info = dataset.samples[idx]
        row_dict = df.iloc[orig_info['Original_Index']].to_dict()
        row_dict['Probability'] = round(float(prob), 4)
        row_dict['Prediction'] = 1 if prob >= 0.5 else 0
        results.append(row_dict)

    output_df = pd.DataFrame(results)
    save_path = args.output_file if args.output_file else "predictions.xlsx"

    if save_path.endswith('.csv'):
        output_df.to_csv(save_path, index=False)
    else:
        output_df.to_excel(save_path, index=False)
    print(f"Process complete. Results saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMLP Model Ensemble Inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model_fold_*.pth files")
    parser.add_argument("--input_file", type=str, required=True,
                        help="CSV/Excel file with Sequence and Fatty_acid_chain")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    main(parser.parse_args())