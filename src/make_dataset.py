
from pathlib import Path
import random


# ============================================================
# V9 DATASET
# ============================================================

examples = [
    ("Hello", "Hello! How can I help you?"),
    ("Hi", "Hi! What would you like to learn?"),

    ("What is AI?",
     "AI is the field of creating computer systems that can perform tasks involving learning, reasoning, and problem solving."),

    ("What is artificial intelligence?",
     "Artificial intelligence is the field of creating computer systems that can perform tasks that normally require intelligent behavior."),

    ("What is machine learning?",
     "Machine learning is a method where computers learn patterns from examples and use those patterns to make predictions."),

    ("What is deep learning?",
     "Deep learning uses neural networks with multiple layers to learn complex patterns from data."),

    ("What is a neural network?",
     "A neural network is a system of connected mathematical operations whose parameters are adjusted during training."),

    ("Why do neural networks need training?",
     "Training adjusts neural network parameters so the model can make better predictions."),

    ("What is a parameter?",
     "A parameter is a numerical value that controls how a neural network processes information."),

    ("Why do neural networks use parameters?",
     "Parameters allow a neural network to store patterns learned during training."),

    ("What is a transformer?",
     "A transformer is a neural network architecture that uses attention to process relationships between tokens."),

    ("What is attention?",
     "Attention allows a model to focus on relevant parts of a sequence when producing an output."),

    ("Why do transformers use attention?",
     "Attention helps transformers model relationships between tokens in a sequence."),

    ("What is a token?",
     "A token is a numerical representation of a piece of text that a language model can process."),

    ("What is tokenization?",
     "Tokenization converts text into smaller units called tokens."),

    ("What is a tokenizer?",
     "A tokenizer converts text into tokens that can be represented by numerical identifiers."),

    ("What is a vocabulary?",
     "A vocabulary is the collection of tokens that a tokenizer can represent."),

    ("What is an embedding?",
     "An embedding represents a token as a vector of numerical values."),

    ("Why are embeddings useful?",
     "Embeddings convert tokens into numerical vectors that neural networks can process."),

    ("What is a language model?",
     "A language model learns patterns in text and predicts likely next tokens."),

    ("How does a language model work?",
     "A language model uses previous tokens to predict the next token."),

    ("How does text generation work?",
     "Text generation repeatedly predicts the next token and adds it to the sequence."),

    ("What is training?",
     "Training is the process of adjusting model parameters so predictions become more accurate."),

    ("What is an epoch?",
     "An epoch is one complete pass through the training dataset."),

    ("What is a batch?",
     "A batch is a group of training examples processed together during an optimization step."),

    ("What is loss?",
     "Loss measures how different a model's predictions are from the expected targets."),

    ("What is an optimizer?",
     "An optimizer updates neural network parameters using gradients."),

    ("What is gradient descent?",
     "Gradient descent changes parameters in directions that reduce loss."),

    ("What is backpropagation?",
     "Backpropagation calculates gradients that describe how parameters affect loss."),

    ("What is overfitting?",
     "Overfitting occurs when a model performs well on training data but poorly on unseen data."),

    ("How can you recognize overfitting?",
     "Overfitting often appears when training loss decreases while validation loss increases."),

    ("What is generalization?",
     "Generalization is the ability of a model to perform well on unseen examples."),

    ("What is validation?",
     "Validation measures model performance on data that was not used to update its parameters."),

    ("What is validation loss?",
     "Validation loss measures prediction error on validation data."),

    ("Why are GPUs useful for AI?",
     "GPUs can perform many numerical operations in parallel, which helps neural network computation."),

    ("Why do AI models use GPUs?",
     "AI models perform many mathematical operations that GPUs can execute in parallel."),

    ("What is a GPU?",
     "A GPU is a processor designed to perform many calculations in parallel."),

    ("What is CUDA?",
     "CUDA is NVIDIA technology that allows compatible GPUs to perform general purpose computations."),

    ("What is PyTorch?",
     "PyTorch is a machine learning framework that provides tensors, automatic differentiation, neural network modules, and GPU support."),

    ("Why is PyTorch useful?",
     "PyTorch provides tools for building and training neural networks."),

    ("What is a PyTorch tensor?",
     "A PyTorch tensor is a multidimensional numerical data structure used for machine learning."),

    ("What is a dataset?",
     "A dataset is a collection of examples used to train or evaluate a machine learning system."),

    ("What is inference?",
     "Inference is using a trained model to produce predictions without changing its parameters."),

    ("What is a checkpoint?",
     "A checkpoint is a saved version of a model that can be restored later."),

    ("What is sampling?",
     "Sampling chooses the next token from a probability distribution produced by a language model."),

    ("What is temperature?",
     "Temperature controls how concentrated or varied token probabilities are during sampling."),

    ("What is context?",
     "Context is the preceding sequence of tokens used when predicting the next token."),

    ("Why does context matter?",
     "Context provides information that helps a language model choose an appropriate continuation."),

    ("Can a small model learn language?",
     "A small model can learn simple statistical patterns, but its abilities are limited by its data and architecture."),

    ("Why can generated text be incorrect?",
     "A language model can generate incorrect text because it predicts from learned statistical patterns."),

    ("How can a model improve?",
     "A model can improve through better data, suitable architecture, optimization, and evaluation."),
]


# ============================================================
# Shuffle
# ============================================================

random.seed(42)
random.shuffle(examples)


# ============================================================
# Create dataset
# ============================================================

output = []

for question, answer in examples:

    output.append(
        "User: "
        + question
        + "\n"
        + "Assistant: "
        + answer
        + "\n\n"
    )


text = "".join(output)


# ============================================================
# Save
# ============================================================

path = Path(
    "data/chat_training.txt"
)

path.parent.mkdir(
    parents=True,
    exist_ok=True
)

path.write_text(
    text,
    encoding="utf-8"
)


print("V9 dataset created.")
print("Characters:", len(text))
print("Examples:", len(examples))
print("Vocabulary:", len(set(text)))
print("Saved to:", path)