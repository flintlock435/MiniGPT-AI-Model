# MiniGPT AI Model

MiniGPT is an experimental local AI project by R0sinc. It includes custom transformer training experiments and a conversational client built around a pretrained language model.

## Version 15.0.0

## Features

- Local command-line conversation
- Pretrained foundation-model inference
- 4-bit model loading on compatible CUDA systems
- Streaming text generation
- Conversation history
- Custom PyTorch transformer implementation and training experiments
- Versioned source code and legacy checkpoint update utilities

Four-bit loading reduces the model's memory requirements. Actual performance depends on the GPU, available VRAM, system RAM, installed PyTorch/CUDA components, and generation settings.

## Installation

Clone the repository:

```powershell
git clone https://github.com/flintlock435/MiniGPT-AI-Model.git
cd MiniGPT-AI-Model```
