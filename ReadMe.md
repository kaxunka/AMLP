# AMLP: Antimicrobial Lipopeptide Activity Prediction

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

This repository provides the official implementation of the **AMLP (Antimicrobial Lipopeptide)** deep learning framework. It is designed for the *de novo* design and activity prediction of lipopeptides by integrating a **"Chain‑as‑Prompt"** mechanism with a two‑stage domain‑adaptive transfer learning strategy.


## 🚀 Key Features

- **Two‑Stage Transfer Learning**: A pre‑trained protein language model (ProtBERT) is first fine‑tuned on a large‑scale linear AMP dataset (LinAMP‑BERT), and a lightweight downstream module is then trained on the lipopeptide dataset to learn the modulation effect of N‑terminal fatty acid chains.
- **Chain‑as‑Prompt (CaP)**: Rather than being treated as a static auxiliary input, the fatty acid chain type (C6–C18) is encoded as a learnable *prompt token*. This token is prepended to the peptide sequence and jointly processed by a single‑layer Transformer encoder, which dynamically re‑weights the amino acid features according to the lipid environment.
- **Parameter‑Efficient Design**: By freezing the backbone and using an extremely compact bottleneck (32 dimensions), the downstream module contains only approximately 42,000 trainable parameters, effectively preventing overfitting on the small lipopeptide dataset.
- **Ensemble Inference**: Built‑in support for 5‑fold cross‑validation ensemble to ensure robust and reliable predictions.

## 📦 Model Weights

The pre‑trained weights for each stage are publicly available on Hugging Face. Please download them and place them in the corresponding `model/` directory.

| Model Stage | Description | Download Link |
| :--- | :--- | :--- |
| **LinAMP‑BERT** | ProtBERT fine‑tuned on 50,310 linear antimicrobial and non‑antimicrobial peptides. | [Frankie1116/LinAMP-BERT-weights](https://huggingface.co/Frankie1116/LinAMP-BERT-weights) |
| **AMLP** | Final 5‑fold ensemble models for lipopeptide activity prediction. | [Frankie1116/AMLP-weights](https://huggingface.co/Frankie1116/AMLP-weights) |

## 📂 Project Structure

```text
├── scripts/
│   ├── LinAMP_BERT_train.py  # Stage 1: Fine‑tuning ProtBERT for Linear AMPs
│   ├── AMLP_train.py         # Stage 2: Chain‑as‑Prompt Training on Lipopeptides
│   └── AMLP_predict.py       # 5‑fold Ensemble Inference
├── model/
│   ├── LinAMP-BERT/          # Weights for the pre‑trained LinAMP‑BERT backbone
│   └── AMLP/                 # 5‑fold .pth files for the final ensemble model
├── prediction_dataset/       # Directory for input/output files
├── environment.yml           # Complete software dependency configuration
└── README.md
```
## 🛠️ Requirements

All experiments were conducted using the following environment:

- Python: 3.12
- PyTorch: 2.5.1
- CUDA: 12.4
- Transformers: 4.30+
- Other dependencies: scikit‑learn, pandas, numpy, tqdm, openpyxl, joblib, tensorboard
  
A complete and detailed list of all dependencies is provided in the environment.yml file. To replicate the exact software environment, you can install all packages by running:

```Bash
conda env create -f environment.yml
conda activate amlp-env
```

## 📖 Data Specification

### Input File Format (`.xlsx` or `.csv`)

The input file for prediction **must contain** the following columns:

| Column Name | Description | Example |
|------------|------------|------------|
| `Sequence` | The amino acid sequence of the core peptide. | GLLKIIKKLL |
| `Fatty_acid_chain` | The type of N‑terminal fatty acid chain. |  `C12` |

#### Supported chain types: `C6`, `C8`, `C10`, `C12`, `C14`, `C16`, `C18`.


## 💻 Usage

###  Inference (Prediction)
Predict antimicrobial activity of linear peptides using LinAMP‑BERT：
```Bash
python scripts/LinAMP_BERT_predict.py \
    --model_dir model/LinAMP-BERT \
    --input_file prediction_dataset/linear_peptides.xlsx \
    --output_file prediction_dataset/linear_predictions.xlsx
```
To run predictions using the pre-trained 5-fold ensemble:
```Bash
python scripts/AMLP_predict.py \
    --model_dir model/AMLP \
    --input_file prediction_dataset/your_input.xlsx \
    --output_file prediction_dataset/prediction_results.xlsx
```
Example Command (Test Run):
```Bash
python scripts/AMLP_predict.py \
    --model_dir model/AMLP \
    --input_file prediction_dataset/your_lipopeptide.xlsx \
    --output_file prediction_dataset/prediction_output.xlsx
```
## 📊 Methodology Summary

- **Backbone**: A frozen LinAMP‑BERT model (ProtBERT fine‑tuned on 50,310 linear peptides) serves as the universal feature extractor.
- **Chain‑as‑Prompt (CaP)**: The fatty acid chain type is embedded as a prompt token and prepended to the projected peptide sequence. A single‑layer Transformer encoder then processes this joint input, allowing the model to dynamically re‑weight peptide features based on the lipid environment.
- **Classification**: The output of the prompt token after self‑attention is directly fed into a lightweight classifier to predict antimicrobial activity.
- **Inference**: Predictions are made by averaging the softmax probabilities from 5 independently trained models (5‑fold cross‑validation). The final prediction is a direct probability score; a threshold of $\ge 0.5$ is applied to convert it to a class label.
