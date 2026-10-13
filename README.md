# MiniGPT AI Model

MiniGPT is an experimental local AI project by R0sinc. It combines custom PyTorch transformer training experiments with a conversational client built around a pretrained foundation model.

**Current version:** 15.0.0

## Features

- Local command-line AI conversation
- Pretrained foundation-model inference
- Streaming text generation
- Conversation history
- 4-bit quantization on compatible CUDA systems
- Custom PyTorch transformer implementation and training experiments
- Dataset preparation and training scripts
- Legacy checkpoint updater

## Requirements

Before installing, make sure you have:

- Python 3.10 or newer
- Git
- Internet access for installing dependencies and downloading the pretrained model
- Enough disk space for the model files and Python dependencies
- Additional system RAM and GPU VRAM appropriate for the selected model

An NVIDIA GPU with compatible CUDA support is recommended for GPU acceleration. CPU inference may be possible but can be significantly slower.

## Installation

### 1. Download the repository

Install Git if you do not already have it.

Clone the repository:

```bash
git clone https://github.com/flintlock435/MiniGPT-AI-Model.git
cd MiniGPT-AI-Model
```

Alternatively, download the repository ZIP from GitHub and extract it.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

With the virtual environment activated, run:

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

Install the project's Python dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch installation and CUDA compatibility can vary by operating system and hardware. If GPU support is important, verify that your installed PyTorch build supports your CUDA setup.

### 5. Launch MiniGPT

From the repository root, run:

```bash
python -m src.gpt_chat
```

On the first launch, the chat client may download the configured pretrained model from Hugging Face. The initial download can take time and requires sufficient storage.

Keep the internet connection active until the model download finishes. Later launches may reuse the cached model files.

## Model information

The conversational client uses the pretrained model configured in `src/gpt_chat.py`.

Check that file for the exact model identifier and any options controlling device placement, quantization, context length, and generation.

The pretrained model is separate from the custom transformer training experiments included in this repository. Downloading the source-code ZIP does not necessarily include pretrained model weights.

Pretrained model files remain subject to their original publisher's license and terms.

## Troubleshooting

### Python is not recognized

Install Python and ensure it is available through your terminal. On Windows, try:

```powershell
py --version
```

### Dependency installation fails

Upgrade pip and review the error message:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some packages, especially PyTorch and quantization libraries, may require operating-system-specific or hardware-specific installation steps.

### CUDA or GPU errors

Verify that PyTorch detects your GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is unavailable, check your PyTorch installation and GPU compatibility. Remove or adjust GPU-specific options only if the client supports CPU execution.

### The first launch is slow

Downloading a large pretrained model can take considerable time. Download speed, model size, and available storage affect startup time.

### Out-of-memory errors

The model may require more system RAM or GPU VRAM than your machine has available. Actual requirements depend on model size, quantization, context length, and generation settings.

## Custom training experiments

The `src/` directory also contains scripts for custom transformer training and dataset preparation.

These scripts are separate from the pretrained-model chat client. Review each script's configuration and dataset requirements before running it. Training can require substantial memory, compute, and disk space.

## Project structure

```text
MiniGPT-AI-Model/
├── data/
├── src/
├── .gitignore
├── LICENSE
├── NOTICE
├── README.md
└── requirements.txt
```

The precise contents of `data/` and `src/` may change between versions.

## License and attribution

The original project source code is distributed under the Apache License 2.0, subject to the repository's license and notice files.

Pretrained models and third-party dependencies have their own licenses and terms. Review the relevant upstream licenses before redistribution or commercial use.

## Disclaimer

MiniGPT is an experimental project provided for educational and development purposes. Performance and compatibility vary with hardware, dependencies, and model configuration.
