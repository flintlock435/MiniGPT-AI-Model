from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/chat_training_v11.txt"

OUTPUT_PATH = "data/v12_tokenizer.json"

VOCAB_SIZE = 8000

MIN_FREQUENCY = 2


# ============================================================
# CHECK DATASET
# ============================================================

data_path = Path(DATA_PATH)

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


text = data_path.read_text(
    encoding="utf-8"
)


print(
    "Dataset loaded."
)

print(
    "Characters:",
    len(text)
)


# ============================================================
# CREATE BPE TOKENIZER
# ============================================================

tokenizer = Tokenizer(
    BPE(
        unk_token="<UNK>"
    )
)


tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)


tokenizer.decoder = ByteLevelDecoder()


# ============================================================
# TRAINER
# ============================================================

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=[
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>"
    ],
    initial_alphabet=ByteLevel.alphabet(),
    show_progress=True
)


# ============================================================
# TRAIN
# ============================================================

print()
print(
    "Training Byte-Level BPE tokenizer..."
)

lines = text.splitlines()

tokenizer.train_from_iterator(
    lines,
    trainer=trainer
)


# ============================================================
# SAVE
# ============================================================

output_path = Path(
    OUTPUT_PATH
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


tokenizer.save(
    str(output_path)
)


# ============================================================
# TEST
# ============================================================

print()
print(
    "Tokenizer trained successfully."
)

print(
    "Vocabulary:",
    tokenizer.get_vocab_size()
)

print(
    "Saved to:",
    OUTPUT_PATH
)


test_strings = [
    "Hello",
    "PyTorch",
    "pytorch",
    "Pytorch",
    "What is a neural network?",
    "What is overfitting?"
]


print()
print(
    "Testing tokenizer:"
)


for text_value in test_strings:

    encoded = tokenizer.encode(
        text_value
    )

    decoded = tokenizer.decode(
        encoded.ids
    )

    print()
    print(
        "Input:",
        text_value
    )

    print(
        "Tokens:",
        encoded.tokens
    )

    print(
        "IDs:",
        encoded.ids
    )

    print(
        "Decoded:",
        decoded
    )

    if "<UNK>" in encoded.tokens:

        print(
            "WARNING: <UNK> detected"
        )

    else:

        print(
            "OK: no <UNK>"
        )


print()
print(
    "Done."
)