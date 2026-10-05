import os
import re

import torch
from tokenizers import Tokenizer

from src.model import MiniGPT


# ============================================================
# V13 SETTINGS
# ============================================================

TOKENIZER_PATH = "data/v12_tokenizer.json"

CHECKPOINT_PATH = "checkpoints/v13_best_model.pt"

MAX_NEW_TOKENS = 80

TEMPERATURE = 0.25

TOP_K = 20

REPETITION_PENALTY = 1.05

USE_SAMPLING = False


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    "Using device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# FILE CHECKS
# ============================================================

if not os.path.exists(TOKENIZER_PATH):

    raise FileNotFoundError(
        f"Tokenizer not found: {TOKENIZER_PATH}"
    )


if not os.path.exists(CHECKPOINT_PATH):

    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

vocab_size = tokenizer.get_vocab_size()

print(
    "Tokenizer vocabulary:",
    vocab_size
)


# ============================================================
# SPECIAL TOKENS
# ============================================================

eos_id = tokenizer.token_to_id(
    "<EOS>"
)

bos_id = tokenizer.token_to_id(
    "<BOS>"
)

pad_id = tokenizer.token_to_id(
    "<PAD>"
)

unk_id = tokenizer.token_to_id(
    "<UNK>"
)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)


block_size = checkpoint.get(
    "block_size",
    128
)

embed_size = checkpoint.get(
    "embed_size",
    96
)

num_heads = checkpoint.get(
    "num_heads",
    4
)

num_layers = checkpoint.get(
    "num_layers",
    3
)


# ============================================================
# MODEL
# ============================================================

model = MiniGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_size=embed_size,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)


model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


print(
    "V13 model loaded."
)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print(
    "Validation loss:",
    checkpoint.get(
        "best_val_loss",
        "unknown"
    )
)


# ============================================================
# CLEAN INPUT
# ============================================================

def clean_question(text):

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    lower = text.lower()


    replacements = {

        "whats ": "what is ",

        "what's ": "what is ",

        "dont ": "don't ",

        "doesnt ": "doesn't ",

        "cant ": "can't ",

        "isnt ": "isn't ",

        "wont ": "won't "
    }


    for old, new in replacements.items():

        if lower.startswith(old):

            text = (
                new
                + text[len(old):]
            )

            break


    return text.strip()


# ============================================================
# REPETITION PENALTY
# ============================================================

def apply_repetition_penalty(
    logits,
    generated_ids,
    penalty
):

    if penalty <= 1.0:

        return logits


    recent_ids = generated_ids[
        -30:
    ]


    for token_id in set(
        recent_ids
    ):

        if (
            token_id < 0
            or token_id >= logits.shape[-1]
        ):

            continue


        if logits[token_id] > 0:

            logits[token_id] /= penalty

        else:

            logits[token_id] *= penalty


    return logits


# ============================================================
# TOKEN SELECTION
# ============================================================

def select_next_token(
    logits,
    generated_ids
):

    logits = logits.clone()


    # --------------------------------------------------------
    # Repetition penalty
    # --------------------------------------------------------

    logits = apply_repetition_penalty(
        logits,
        generated_ids,
        REPETITION_PENALTY
    )


    # --------------------------------------------------------
    # Block special tokens
    # --------------------------------------------------------

    for token_id in [
        pad_id,
        bos_id,
        unk_id
    ]:

        if (
            token_id is not None
            and 0 <= token_id < logits.shape[-1]
        ):

            logits[token_id] = float(
                "-inf"
            )


    # --------------------------------------------------------
    # Greedy decoding
    # --------------------------------------------------------

    if not USE_SAMPLING:

        return int(
            torch.argmax(
                logits
            ).item()
        )


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    logits = (
        logits
        / max(
            TEMPERATURE,
            0.05
        )
    )


    if TOP_K > 0:

        k = min(
            TOP_K,
            logits.shape[-1]
        )


        values, indices = torch.topk(
            logits,
            k
        )


        filtered = torch.full_like(
            logits,
            float("-inf")
        )


        filtered.scatter_(
            0,
            indices,
            values
        )


        logits = filtered


    probabilities = torch.softmax(
        logits,
        dim=-1
    )


    return int(
        torch.multinomial(
            probabilities,
            1
        ).item()
    )


# ============================================================
# GENERATE
# ============================================================

@torch.no_grad()
def generate(question):

    question = clean_question(
        question
    )


    # --------------------------------------------------------
    # EXACT TRAINING FORMAT
    # --------------------------------------------------------

    prompt = (
        "User: "
        + question
        + "\nAssistant:"
    )


    prompt_ids = tokenizer.encode(
        prompt
    ).ids


    if not prompt_ids:

        return ""


    # --------------------------------------------------------
    # Context limit
    # --------------------------------------------------------

    prompt_ids = list(
        prompt_ids[
            -block_size:
        ]
    )


    generated_ids = list(
        prompt_ids
    )


    # --------------------------------------------------------
    # Generate tokens
    # --------------------------------------------------------

    for _ in range(
        MAX_NEW_TOKENS
    ):

        context = generated_ids[
            -block_size:
        ]


        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )


        logits, _ = model(
            x
        )


        next_logits = logits[
            0,
            -1
        ]


        next_token = select_next_token(
            next_logits,
            generated_ids
        )


        generated_ids.append(
            next_token
        )


        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if (
            eos_id is not None
            and next_token == eos_id
        ):

            break


        # ----------------------------------------------------
        # Stop on another conversation turn.
        # ----------------------------------------------------

        answer_ids = generated_ids[
            len(prompt_ids):
        ]


        partial = tokenizer.decode(
            answer_ids,
            skip_special_tokens=True
        )


        if "\nUser:" in partial:

            break


        if "\nAssistant:" in partial:

            break


    # ========================================================
    # DECODE ANSWER
    # ========================================================

    answer_ids = generated_ids[
        len(prompt_ids):
    ]


    answer = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True
    )


    # ========================================================
    # CLEAN
    # ========================================================

    answer = re.split(
        r"\nUser:",
        answer,
        maxsplit=1
    )[0]


    answer = re.split(
        r"\nAssistant:",
        answer,
        maxsplit=1
    )[0]


    answer = re.sub(
        r"^[\s:,\-]+",
        "",
        answer
    )


    answer = re.sub(
        r"\s+",
        " ",
        answer
    )


    return answer.strip()


# ============================================================
# CHAT
# ============================================================

print()

print(
    "=========================================="
)

print(
    "V13 AI CHAT"
)

print(
    "Type 'exit' to quit."
)

print(
    "=========================================="
)

print()


while True:

    try:

        question = input(
            "You: "
        )

    except (
        KeyboardInterrupt,
        EOFError
    ):

        print()

        break


    question = question.strip()


    if not question:

        continue


    if question.lower() in {
        "exit",
        "quit",
        "q"
    }:

        print(
            "Goodbye."
        )

        break


    answer = generate(
        question
    )


    if not answer:

        answer = (
            "I don't know how to answer that yet."
        )


    print(
        "AI:",
        answer
    )

    print()