import json
import os
import re

import torch
from tokenizers import Tokenizer

from src.model import MiniGPT
from src.updater import check_for_update


# ============================================================
# STARTUP UPDATE CHECK
# ============================================================

print()

print(
    "Checking for MiniGPT updates..."
)

try:

    updated = check_for_update()

    if updated:

        print(
            "Update installed."
        )

        print(
            "Loading the updated model..."
        )

except Exception as error:

    print(
        "Update checker error:",
        error
    )


print()


# ============================================================
# V14 SETTINGS
# ============================================================

TOKENIZER_PATH = "data/v12_tokenizer.json"

CHECKPOINT_PATH = "checkpoints/v14_custom_best_model.pt"

MAX_NEW_TOKENS = 80

REPETITION_PENALTY = 1.02

BLOCK_REPEATED_SENTENCES = True


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
# CHECK FILES
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
        f"Checkpoint not found: {CHECKPOINT_PATH}"
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
# CREATE MODEL
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


# ============================================================
# VERSION
# ============================================================

installed_version = "unknown"

version_file = "data/version.json"


if os.path.exists(
    version_file
):

    try:

        with open(
            version_file,
            "r",
            encoding="utf-8"
        ) as file:

            version_data = json.load(
                file
            )


        installed_version = str(
            version_data.get(
                "version",
                "unknown"
            )
        )

    except Exception:

        installed_version = "unknown"


else:

    installed_version = "14.0.0"


print(
    "MiniGPT version:",
    installed_version
)


print(
    "Model parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# INPUT CLEANUP
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
        -20:
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
    # EXACT FORMAT USED DURING V14 TRAINING
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

    if len(prompt_ids) > block_size:

        prompt_ids = prompt_ids[
            -block_size:
        ]


    generated_ids = list(
        prompt_ids
    )


    # --------------------------------------------------------
    # Generate token by token
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
        # Repetition penalty
        # ----------------------------------------------------

        next_logits = apply_repetition_penalty(
            next_logits,
            generated_ids
        )


        # ----------------------------------------------------
        # Block special tokens
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
        # Greedy decoding
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
        # EOS
        # ----------------------------------------------------

        if (
            eos_id is not None
            and next_token == eos_id
        ):

            break


        # ----------------------------------------------------
        # Decode partial answer
        # ----------------------------------------------------

        answer_ids = generated_ids[
            len(prompt_ids):
        ]


        partial = tokenizer.decode(
            answer_ids,
            skip_special_tokens=True
        )


        # ----------------------------------------------------
        # Stop at another conversation turn
        # ----------------------------------------------------

        if "\nUser:" in partial:

            break


        if "\nAssistant:" in partial:

            break


        # ----------------------------------------------------
        # Stop repeated sentences
        # ----------------------------------------------------

        if BLOCK_REPEATED_SENTENCES:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                partial.strip()
            )


            if len(sentences) >= 3:

                last = sentences[-1].strip()

                previous = sentences[-2].strip()


                if (
                    last
                    and last.lower()
                    == previous.lower()
                ):

                    break


    # ========================================================
    # DECODE ANSWER ONLY
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
    "MiniGPT V14"
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