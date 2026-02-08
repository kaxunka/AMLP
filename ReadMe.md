# AMLP: Antimicrobial Lipopeptide Activity Prediction

AMLP is a deep learning framework designed for predicting the antimicrobial activity of **Linear Antimicrobial Peptides (Linear AMPs)** and **Lipopeptides**. By leveraging the **ProtBERT** architecture, the project introduces a **Chain-as-Prompt (CaP)** mechanism and **Consistency Regularization** to specifically model peptides modified with fatty acid chains.

## 🚀 Key Features

- **Multi-Stage Fine-tuning**: Progressive training from general protein representation to Linear AMPs and Lipopeptide-specific tasks.
- **Chain-as-Prompt (CaP)**: Fatty acid chain information is encoded as learnable prompts injected into the model.
- **Consistency Regularization**: A specialized loss function to maintain representation stability between lipopeptides and linear peptides.
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
│   ├── LinAMP_BERT_train.py  # Stage 2: Linear AMP fine-tuning
│   ├── AMLP_train.py         # Stage 3: Lipopeptide consistency training
│   └── AMLP_predict.py       # Ensemble inference and prediction
├── data/
│   ├── Linear_data/          # Datasets for Linear AMPscd
│   │   ├── linear_full_dataset.CSV
│   │   ├── train_dataset.csv
│   │   ├── val_dataset.csv
│   │   └── test_dataset.csv
│   └── Lipo_data/            # Datasets for Lipopeptides
│       ├── lipo_full_data.csv
│       ├── lipo_train_val.csv
│       └── lipo_test_independent.csv
├── model/
│   └── AMLP/                 # Store 5-fold .pth files here
└── prediction_dataset/       # Input/Output directory for inference


🛠️ Requirements
Python 3.8+
PyTorch 1.12+
Transformers (Hugging Face)
Scikit-learn, Pandas, Numpy, tqdm, Openpyxl
Install all dependencies via pip:
code
Bash
pip install torch transformers scikit-learn pandas numpy tqdm openpyxl
📖 Data Specification
Input File Format (.xlsx or .csv)
The input file for prediction must contain the following columns:
Sequence: The amino acid sequence (e.g., KLLKLLKKLLK).
Fatty_acid_chain: The modification type. Supported values: C0 (Linear), C6, C8, C10, C12, C14, C16, C18.
💻 Usage
1. Training (Optional)
If you wish to train the models from scratch:
Fine-tune Linear AMPs:
code
Bash
python scripts/LinAMP_BERT_train.py --data_path data/linear_data.csv --output_dir results/linamp_bert
Train AMLP with Consistency Loss:
code
Bash
python scripts/AMLP_train.py \
    --lipo_full_path data/lipo_all.csv \
    --lipo_train_path data/lipo_train.csv \
    --linear_train_path data/linear_50k.csv \
    --pretrain_model_path results/linamp_bert/best_model.pth \
    --output_dir model/AMLP
2. Inference (Prediction)
To run predictions using the pre-trained 5-fold ensemble:
code
Bash
python scripts/AMLP_predict.py \
    --model_dir model/AMLP \
    --input_file prediction_dataset/your_input.xlsx \
    --output_file prediction_dataset/prediction_results.xlsx
Example Command (Test Run):
code
Bash
python scripts/AMLP_predict.py \
    --model_dir model/AMLP \
    --input_file prediction_dataset/your_file.xlsx \
    --output_file prediction_dataset/your_file_output.xlsx
📊 Methodology Summary
Backbone: ProtBERT (Rostlab/prot_bert) serves as the sequence encoder.
CaP Module: Maps fatty acid chains into an embedding space as learnable prefix prompts.
Adapter: Lightweight bottleneck adapters are used to adapt features to the lipopeptide domain without losing general knowledge.
Consistency Loss: Minimizes the distance between representations of a peptide with and without its fatty acid chain to enhance biological feature extraction.