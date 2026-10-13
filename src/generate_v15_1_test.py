import os
import re

import torch
from tokenizers import Tokenizer

from src.model import MiniGPT


# ============================================================
# V15.1 TEST SETTINGS
# ============================================================

TOKENIZER_PATH = "data/v12_tokenizer.json"

CHECKPOINT_PATH = "checkpoints/v15_1_best_model.pt"

MAX_NEW_TOKENS = 80

REPETITION_PENALTY = 1.02


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

if not os.path.exists(
    TOKENIZER_PATH
):

    raise FileNotFoundError(
        f"Tokenizer not found: {TOKENIZER_PATH}"
    )


if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(
        f"V15.1 checkpoint not found: {CHECKPOINT_PATH}"
    )


# ============================================================
# LOAD TOKENIZER
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

pad_id = tokenizer.token_to_id(
    "<PAD>"
)

bos_id = tokenizer.token_to_id(
    "<BOS>"
)

unk_id = tokenizer.token_to_id(
    "<UNK>"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)


block_size = checkpoint.get(
    "block_size",
    256
)

embed_size = checkpoint.get(
    "embed_size",
    128
)

num_heads = checkpoint.get(
    "num_heads",
    4
)

num_layers = checkpoint.get(
    "num_layers",
    4
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
    "V15.1 model loaded successfully."
)


print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


print(
    "Best validation loss:",
    checkpoint.get(
        "best_val_loss",
        "unknown"
    )
)


print(
    "Model version:",
    checkpoint.get(
        "version",
        "unknown"
    )
)


# ============================================================
# CLEAN QUESTION
# ============================================================

def clean_question(text):

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# REPETITION PENALTY
# ============================================================

def apply_repetition_penalty(
    logits,
    generated_ids
):

    if REPETITION_PENALTY <= 1.0:

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

            logits[token_id] /= (
                REPETITION_PENALTY
            )

        else:

            logits[token_id] *= (
                REPETITION_PENALTY
            )


    return logits


# ============================================================
# GENERATE
# ============================================================

@torch.no_grad()
def generate(question):

    question = clean_question(
        question
    )


    # --------------------------------------------------------
    # Exact format used during training.
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
    # Context limit.
    # --------------------------------------------------------

    if len(prompt_ids) > block_size:

        prompt_ids = prompt_ids[
            -block_size:
        ]


    generated_ids = list(
        prompt_ids
    )


    # --------------------------------------------------------
    # Generate.
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
        ].clone()


        # ----------------------------------------------------
        # Repetition penalty.
        # ----------------------------------------------------

        next_logits = apply_repetition_penalty(
            next_logits,
            generated_ids
        )


        # ----------------------------------------------------
        # Block special tokens.
        # ----------------------------------------------------

        for token_id in [
            pad_id,
            bos_id,
            unk_id
        ]:

            if (
                token_id is not None
                and 0 <= token_id < next_logits.shape[-1]
            ):

                next_logits[token_id] = float(
                    "-inf"
                )


        # ----------------------------------------------------
        # Greedy decoding.
        # ----------------------------------------------------

        next_token = int(
            torch.argmax(
                next_logits
            ).item()
        )


        generated_ids.append(
            next_token
        )


        # ----------------------------------------------------
        # EOS.
        # ----------------------------------------------------

        if (
            eos_id is not None
            and next_token == eos_id
        ):

            break


        # ----------------------------------------------------
        # Detect another conversation turn.
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
    # CLEAN ANSWER
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
    "MiniGPT V15.1 TEST"
)

print(
    "=========================================="
)

print(
    "Type 'exit' to quit."
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
            "I don't know yet."
        )


    print(
        "AI:",
        answer
    )

    print()