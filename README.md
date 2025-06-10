---
license: other
license_name: ngen-3-community-license
license_link: https://tnsaai-builds.framer.website/community/licenses/ngen3
library_name: transformers
model-index:
- name: NGen-3-7B
  results:
  - task:
      type: text-generation
    dataset:
      name: TCorpus5
      type: WebCorpus
    metrics:
    - name: MMLU
      type: accuracy
      value: 60.24
    - name: PIQA
      type: accuracy
      value: 79.12
    - name: Hellaswag
      type: accuracy
      value: 52.87
    - name: Winogrande
      type: accuracy
      value: 68.35
pipeline_tag: text2text-generation
datasets:
- HuggingFaceH4/no_robots
- Open-Orca/SlimOrca
- Skylion007/openwebtext
- HuggingFaceFW/fineweb
- HuggingFaceTB/smoltalk
---



# NGen 3 - 7B-Instruct

NGen3 is a production-level foundational language model inspired by state-of-the-art architectures such as GPT-4, Claude-3, and Llama 2. It is designed for both research and production and supports model variants ranging from 7M to 1B parameters. The model is built with a modular transformer decoder architecture and provides a comprehensive command-line interface (CLI) for tokenization, training, sampling, exporting, knowledge distillation, and fine-tuning on conversational data.

![alt text](https://raw.githubusercontent.com/TnsaAi/images-urls/refs/heads/main/TV%20-%201%20(24).png)

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Tokenization](#tokenization)
  - [Training](#training)
  - [Sampling](#sampling)
  - [Exporting](#exporting)
  - [Knowledge Distillation](#knowledge-distillation)
  - [Fine-Tuning](#fine-tuning)
    - [Local Fine-Tuning](#local-fine-tuning)
    - [Hugging Face Fine-Tuning](#hugging-face-fine-tuning)
- [Hyperparameters](#hyperparameters)
- [Acknowledgements](#acknowledgements)

---

## Overview

NGen3 is a flexible, self-contained implementation of a foundational language model built on a transformer decoder architecture. It enables users to:

- **Tokenize** text from local files, URLs, or directly from Hugging Face datasets.
- **Train** the model on tokenized datasets.
- **Generate** text samples from trained models.
- **Export** models (with minimal tokenizer configurations) to formats compatible with Hugging Face.
- **Distill** knowledge from larger teacher models into smaller student models.
- **Fine-Tune** on conversational datasets (using local files or datasets from Hugging Face).

---

## Model Architecture

NGen3 uses a decoder-only transformer design with the following components:

- **Token & Positional Embeddings:** Learnable embeddings for tokens and their positions.
- **Transformer Blocks:** A stack of blocks, each containing:
  - **Causal Self-Attention:** Multi-head attention with a lower-triangular mask to prevent attention to future tokens.
  - **Feed-Forward Network (MLP):** With GELU activation.
  - **Residual Connections & Layer Normalization:** To stabilize training.
- **Final Projection Layer:** Projects the hidden states to logits over the vocabulary.

The model comes in several variants:
- **7M Variant:** 4 layers, 4 heads, 128-dimensional embeddings.
- **120M Variant:** 12 layers, 8 heads, 512-dimensional embeddings.
- **300M, 500M, 700M, and 1B Variants:** Increasing in depth and width.

---

## Evaluation Results

![alt text](https://raw.githubusercontent.com/TnsaAi/images-urls/refs/heads/main/ngen3-7b-bench1.png)
![alt text](https://raw.githubusercontent.com/TnsaAi/images-urls/refs/heads/main/ngne3-7b-bench2.png)
![alt text](https://raw.githubusercontent.com/TnsaAi/images-urls/refs/heads/main/ngen3-7b-bench3.png)

## Installation

Ensure you have Python 3.8+ installed and install the necessary dependencies:

```bash
pip install torch transformers datasets tqdm safetensors
```
## Usage

NGen3 is fully managed via a CLI. Below are examples for each command.
Tokenization
Local Text File or URL:
```bash
python _model_.py tokenize --dataset tinyshakespeare --txt "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
```

Hugging Face Dataset:
```bash
python _model_.py hf_tokenize --hf_dataset roskoN/dailydialog --hf_split train --hf_text_column utterances --dataset dailydialog_train
```

## Training
Train a model variant (e.g., 7M):
```bash
python _model_.py train --variant 7M --data _data_tinyshakespeare_/data.bin
```

## Sampling
Generate text samples from a trained model:
```bash
python _model_.py sample --variant 7M --model_checkpoint 7M_model.pt --prompt "To be, or not to be" --length 100 --temperature 1.0
```
## Exporting
Export a trained model (and its tokenizer configuration) for Hugging Face:

```bash
python _model_.py export --variant 7M --model_path 7M_model.pt --output_dir exported_7M
```

## Knowledge Distillation
Distill a larger teacher model (e.g., GPT-2 120M from HF) into a smaller student model (e.g., 7M):

```bash
python _model_.py distill --teacher_model_path hf --teacher_variant 120M --student_variant 7M --data _data_tinyshakespeare_/data.bin --temperature 2.0 --alpha 0.5
```

## Fine-Tuning
Local Fine-Tuning on Conversational Data
Fine-tune a distilled model using local conversation data:

```bash

python _model_.py finetune --variant 120M --model_checkpoint distilled_120M_model.pt --data _data_conversations_/data.bin --finetune_iters 1000 --prompt "Hello, how are you?" --sample_length 100 --sample_temperature 1.0
```
Hugging Face Fine-Tuning on a Conversational Dataset
Fine-tune on a conversational dataset from Hugging Face (e.g., roskoN/dailydialog):

```bash

python _model_.py hf_finetune --variant 120M --model_checkpoint distilled_120M_model.pt --hf_dataset roskoN/dailydialog --hf_split train --hf_text_column utterances --finetune_iters 1000 --prompt "Hello, how are you?" --sample_length 100 --sample_temperature 1.0
```

## Sampling and Exporting Fine-Tuned Models
After fine-tuning, you can sample from or export the fine-tuned model just as with any checkpoint. For example, if your fine-tuned model is saved as finetuned_120M_model.pt:

Sampling:

```bash
python _model_.py sample --variant 120M --model_checkpoint finetuned_120M_model.pt --prompt "What do you think about AI?" --length 100 --temperature 1.0
```
Exporting:

```bash
python _model_.py export --variant 120M --model_path finetuned_120M_model.pt --output_dir exported_finetuned_120M
```
## Hyperparameters
Each model variant comes with predefined hyperparameters. For example:

7M Variant:

Layers: 4, Heads: 4, Embedding Dimension: 128
Block Size: 128, Batch Size: 16, Learning Rate: 3e-4
120M Variant:

Layers: 12, Heads: 8, Embedding Dimension: 512
Block Size: 256, Batch Size: 32, Learning Rate: 3e-4
300M, 500M, 700M, 1B Variants:
Increasing layers, heads, and embedding dimensions for better performance.

Adjust ```max_iters```, ```log_interval```, and ```eval_interval``` to suit your dataset size and computational resources.


## Acknowledgements
NGen3 is inspired by leading models including GPT-4, Claude-3, and Llama 2. Special thanks to the open-source community for:

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- safetensors
