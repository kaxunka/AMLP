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

AMLP is a deep learning framework designed for predicting the antimicrobial activity of **Linear Antimicrobial Peptides (Linear AMPs)** and **Lipopeptides**. By leveraging the **ProtBERT** architecture, the project introduces a **Chain-as-Prompt (CaP)** mechanism and **Consistency Regularization** to specifically model peptides modified with fatty acid chains.

## 🚀 Key Features

- **Multi-Stage Fine-tuning**: Progressive training from general protein representation to Linear AMPs and Lipopeptide-specific tasks.
- **Chain-as-Prompt (CaP)**: Fatty acid chain information is encoded as learnable prompts injected into the model.
- **Parameter‑Efficient Design**: By freezing the backbone and using an extremely compact bottleneck (32 dimensions), the downstream module contains only approximately 42,000 trainable parameters, effectively preventing overfitting on the small lipopeptide dataset.
- **Ensemble Inference**: Built-in support for 5-fold cross-validation ensemble to ensure robust and reliable predictions.

## 📦 Model Weights

The pre-trained weights for each stage are hosted on Hugging Face. Please download them and place them in the corresponding `model/` directory:

| Model Stage | Description | Download Link |
| :--- | :--- | :--- |
| **LinAMP-BERT** | Fine-tuned ProtBERT for Linear AMPs | [Frankie1116/LinAMP-BERT-weights](https://huggingface.co/Frankie1116/LinAMP-BERT-weights) |
| **AMLP** | Final Ensemble Models for Lipopeptides | [Frankie1116/AMLP-weights](https://huggingface.co/Frankie1116/AMLP-weights) |

## 📂 Project Structure

```text
├── scripts/
│   ├── LinAMP_BERT_train.py  # Stage 1: Linear AMP fine-tuning
│   ├── AMLP_train.py         # Stage 2: Lipopeptide consistency training
│   └── AMLP_predict.py       # Ensemble inference and prediction
├── data/
│   ├── Linear_data/          # Datasets for Linear AMPs
│   │   ├── train_dataset.csv
│   │   ├── val_dataset.csv
│   │   └── test_dataset.csv
│   └── Lipo_data/            # Datasets for Lipopeptides
│       ├── lipo_train_val.csv
│       └── lipo_test_independent.csv
├── model/
│   ├── LinAMP-BERT/          # Store linearAMP.pth files here
│   └── AMLP/                 # Store 5-fold .pth files here
└── prediction_dataset/       # Input/Output directory for inference
```
## 🛠️ Requirements

All experiments were conducted using the following environment:

- Python: 3.12
- PyTorch: 2.5.1
- CUDA: 12.4
- Transformers: 4.30+
- Other dependencies: scikit‑learn, pandas, numpy, tqdm, openpyxl, joblib, tensorboard
  
A complete and detailed list of all dependencies is provided in the environment.yml file. To replicate the exact software environment, you can install all packages by running:

conda env create -f environment.yml
conda activate amlp-env

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- transformers (Hugging Face)  
- scikit-learn  
- pandas  
- numpy  
- tqdm  
- openpyxl  

Install dependencies via pip:
```bash
pip install torch transformers scikit-learn pandas numpy tqdm openpyxl
```

## 📖 Data Specification

### Input File Format (`.xlsx` or `.csv`)

The input file for prediction **must contain** the following columns:

| Column Name | Description |
|------------|------------|
| `Sequence` | Amino acid sequence (e.g. `KLLKLLKKLLK`) |
| `Fatty_acid_chain` | Modification type: `C0` (Linear), `C6`, `C8`, `C10`, `C12`, `C14`, `C16`, `C18` |
## 💻 Usage
### 1. Training (Optional)
If you wish to train the models from scratch:
Fine-tune Linear AMPs:
code
```Bash
python scripts/LinAMP_BERT_train.py --data_path data/Linear_data/train_dataset.csv --output_dir results/linamp_bert
```
Train AMLP with Consistency Loss:
code
```Bash
python scripts/AMLP_train.py \
    --lipo_full_path data/Lipo_data/lipo_full_data.csv \
    --lipo_train_path data/Lipo_data/lipo_train_val.csv \
    --linear_train_path data/Linear_data/linear_full_dataset.CSV \
    --pretrain_model_path results/linamp_bert/best_model.pth \
    --output_dir model/AMLP
```
### 2. Inference (Prediction)
To run predictions using the pre-trained 5-fold ensemble:
code
```Bash
python scripts/AMLP_predict.py \
    --model_dir model/AMLP \
    --input_file prediction_dataset/your_input.xlsx \
    --output_file prediction_dataset/prediction_results.xlsx
```
Example Command (Test Run):
code
```Bash
python scripts/AMLP_predict.py \
    --model_dir model/AMLP \
    --input_file prediction_dataset/your_lipopeptide.xlsx \
    --output_file prediction_dataset/prediction_output.xlsx
```
## 📊 Methodology Summary
Backbone: ProtBERT (Rostlab/prot_bert) serves as the sequence encoder.
CaP Module: Maps fatty acid chains into an embedding space as learnable prefix prompts.
Adapter: Lightweight bottleneck adapters are used to adapt features to the lipopeptide domain without losing general knowledge.
Consistency Loss: Minimizes the representation distance between a peptide with and without its fatty acid chain to enhance biological feature extraction.
