import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ========================= 1. Device =========================
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ========================= 2. Model =========================
class ProtBERTPredictor:
    def __init__(self, model_path, model_name="Rostlab/prot_bert"):
        self.device = get_device()

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

        print("Loading trained weights...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()


# ========================= 3. Sequence Encoding =========================
def encode_sequences(sequences, tokenizer, max_len=100):
    encoded = tokenizer(
        [' '.join(list(seq)) for seq in sequences],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return encoded


# ========================= 4. Prediction =========================
def predict(model, tokenizer, sequences, batch_size=32, max_len=100):
    all_probs = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
        batch_seqs = sequences[i:i + batch_size]

        inputs = encode_sequences(batch_seqs, tokenizer, max_len)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)[:, 1]  # AMP probability
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


# ========================= 5. Load Data =========================
def load_data(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV or Excel supported")

    if "Sequence" not in df.columns:
        raise ValueError("Input file must contain 'Sequence' column")

    return df


# ========================= 6. Save Result =========================
def save_result(df, output_path):
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)


# ========================= 7. Main =========================
def main(args):
    predictor = ProtBERTPredictor(args.model_path)

    df = load_data(args.input_file)
    sequences = df["Sequence"].astype(str).tolist()

    probs = predict(
        predictor.model,
        predictor.tokenizer,
        sequences,
        batch_size=args.batch_size,
        max_len=args.max_seq_len
    )

    # add column "predicted probability"
    df["predicted probability"] = probs

    save_result(df, args.output_file)

    print(f"\nPrediction finished. Saved to: {args.output_file}")


# ========================= 8. Args =========================
def parse_args():
    parser = argparse.ArgumentParser(description="LinAMP-BERT Predictor")

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="prediction_output.csv")
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

