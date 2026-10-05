# MiniGPT AI Model

MiniGPT is a small GPT-style language model built from scratch with Python and PyTorch.

The project is an experimental implementation focused on building and training a compact transformer language model locally, including its tokenizer, training pipeline, inference system, model checkpoints, and GitHub-based update mechanism.

## Project Status

MiniGPT is an experimental project under active development.

The repository currently contains several model and training experiments, with the latest custom-data experiment using the V14 checkpoint.

## Features

### Custom Transformer Model

The model is implemented directly in `src/model.py` using PyTorch.

The current architecture contains:

* Multi-head causal self-attention
* Feed-forward layers
* Layer normalization
* Residual connections
* Token embeddings
* Positional embeddings
* Autoregressive next-token prediction

The current experimental configuration uses:

| Component          |      Value |
| ------------------ | ---------: |
| Parameters         |  1,892,000 |
| Embedding size     |         96 |
| Attention heads    |          4 |
| Transformer layers |          3 |
| Context length     | 128 tokens |

### Byte-Level BPE Tokenizer

MiniGPT uses a Byte-Level BPE tokenizer trained with the Hugging Face `tokenizers` library.

The current tokenizer has an 8,000-token vocabulary and is stored in:

```text
data/v12_tokenizer.json
```

The tokenizer is designed to handle previously unseen words more effectively than the original word-level tokenizer.

The tokenizer training script is:

```text
src/train_bpe_tokenizer.py
```

### Local Training

The project includes multiple training scripts for different experiments.

The current custom-dataset training script is:

```text
src/train_v14_custom.py
```

It trains the transformer directly on the project's custom conversational dataset and saves the resulting checkpoint to:

```text
checkpoints/v14_custom_best_model.pt
```

The project also contains earlier training pipelines used to experiment with larger external datasets and different training approaches.

### Conversational Training Format

Training examples use a consistent conversational structure:

```text
User: What is Python?
Assistant: Python is a programming language...
```

The training pipeline separates the user prompt from the assistant response and can calculate the training loss primarily on the response portion.

### GPU Training

The training scripts automatically use CUDA when a compatible NVIDIA GPU is available.

The current training code includes:

* CUDA support
* Automatic mixed precision
* TF32 support
* GPU data transfers
* Fused AdamW on CUDA
* cuDNN benchmarking

CPU execution is also supported.

### Model Checkpoints

Different experiments are stored as separate checkpoint files rather than overwriting previous experiments.

Example:

```text
checkpoints/
├── v10_best_model.pt
├── v11_best_model.pt
├── v12_best_model.pt
├── v13_best_model.pt
└── v14_custom_best_model.pt
```

This makes it possible to compare and return to earlier model versions.

### Command-Line Inference

The model can be run locally from the terminal using:

```powershell
python -m src.generate
```

The inference program loads the tokenizer and checkpoint and provides an interactive command-line chat interface.

Example:

```text
You: Hello
AI: ...
```

### GitHub Release Update System

MiniGPT includes an update system in:

```text
src/updater.py
```

The updater is designed to check the latest GitHub Release and compare it with the locally installed version.

When a newer release is available, the client can prompt the user to update.

The update process is designed to:

1. Check the latest GitHub Release.
2. Compare the installed and latest versions.
3. Ask the user whether to update.
4. Download the model and tokenizer assets.
5. Verify their SHA-256 digests.
6. Replace the local files.
7. Store the new installed version.

The updater is currently configured as an implementation that requires the repository name and release configuration to be set before distribution.

## Repository Structure

The repository currently follows this general structure:

```text
AI Model/
│
├── checkpoints/
│   ├── v10_best_model.pt
│   ├── v11_best_model.pt
│   ├── v12_best_model.pt
│   ├── v13_best_model.pt
│   └── v14_custom_best_model.pt
│
├── data/
│   ├── chat_training.txt
│   ├── chat_training_dolly.txt
│   ├── chat_training_v11.txt
│   ├── chat_training_v13.txt
│   ├── v12_tokenizer.json
│   └── version.json
│
└── src/
    ├── model.py
    ├── small_tokenizer.py
    ├── train.py
    ├── train_bpe_tokenizer.py
    ├── prepare_dolly.py
    ├── make_v11_dataset.py
    ├── prepare_v13_dataset.py
    ├── train_v14_custom.py
    ├── generate.py
    └── updater.py
```

## Installation

Clone the repository:

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the Python packages used by the project:

```powershell
pip install torch tqdm tokenizers
```

## Running the Model

After the model checkpoint and tokenizer are available, run:

```powershell
python -m src.generate
```

The program starts an interactive terminal session.

Type:

```text
exit
```

to close the program.

## Training

### Train the BPE Tokenizer

The tokenizer can be trained with:

```powershell
python -m src.train_bpe_tokenizer
```

The resulting tokenizer is saved to:

```text
data/v12_tokenizer.json
```

### Train the V14 Custom Model

The current custom-data experiment can be trained with:

```powershell
python -m src.train_v14_custom
```

The checkpoint is saved to:

```text
checkpoints/v14_custom_best_model.pt
```

### Dataset Preparation

The repository also contains scripts used during earlier experiments, including:

```text
src/prepare_dolly.py
src/make_v11_dataset.py
src/prepare_v13_dataset.py
```

These scripts were used to prepare and transform different training datasets during development.

## Current Experimental Result

The V14 custom-data experiment successfully reduced its training loss from approximately `9.06` at the beginning of training to approximately `0.033` after 100 epochs on the custom training set.

This result demonstrates that the current model and tokenizer pipeline can strongly fit the small custom dataset.

It does not mean that the model has the general knowledge or response quality of a large language model. The V14 experiment is primarily a controlled training test.

## Limitations

MiniGPT is intentionally small and is still experimental.

The current model can:

* Produce incorrect information
* Struggle with questions outside its training data
* Repeat or truncate responses
* Overfit to a small dataset
* Perform poorly on general language tasks

Model quality depends heavily on the dataset, model size, tokenizer, training configuration, and number of training examples.

## Development History

The repository contains multiple versions because MiniGPT has been developed through repeated experiments.

The project has explored:

* Small word-level tokenization
* Byte-Level BPE tokenization
* Custom conversational datasets
* External instruction datasets
* Different model configurations
* Different training objectives
* Answer-focused training
* EOS-based response termination
* GPU-accelerated training
* Versioned model checkpoints
* GitHub-based model updates

Earlier experiments have intentionally been preserved so future versions can be compared against previous approaches.

## Roadmap

The project is still being developed.

Planned areas of development include:

* Improving the training dataset
* Increasing model capacity
* Improving conversational response quality
* Improving context handling
* Improving evaluation
* Improving inference performance
* Completing the GitHub update workflow for public releases
* Adding stronger update compatibility and rollback protections
* Publishing versioned model releases

## License

Add a `LICENSE` file to this repository before publishing the project publicly.

The license should match the terms you intend to use for the source code, model files, and any included datasets.

## Contributing

Contributions and experiments are welcome.

Useful areas include model architecture, tokenizer improvements, dataset preparation, training performance, inference, evaluation, and update-system development.

## About

MiniGPT is a small-scale language-model project built to explore how transformer-based language models can be implemented, trained, and distributed locally.

The project is intentionally developed one experiment at a time, with previous model versions preserved for comparison.
