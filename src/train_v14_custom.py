import math
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer

from src.model import MiniGPT


# ============================================================
# V14 CUSTOM QA TEST
# ============================================================

DATA_PATH = "data/chat_training.txt"

TOKENIZER_PATH = "data/v12_tokenizer.json"

CHECKPOINT_PATH = "checkpoints/v14_custom_best_model.pt"

BLOCK_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 0.0

EMBED_SIZE = 96

NUM_HEADS = 4

NUM_LAYERS = 3

SEED = 42


# ============================================================
# SEED
# ============================================================

random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# CUDA
# ============================================================

if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True

    torch.set_float32_matmul_precision(
        "high"
    )


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
    DATA_PATH
):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


if not os.path.exists(
    TOKENIZER_PATH
):

    raise FileNotFoundError(
        f"Tokenizer not found: {TOKENIZER_PATH}"
    )


os.makedirs(
    "checkpoints",
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print(
    "Characters:",
    len(text)
)


# ============================================================
# PARSE CUSTOM CHAT
# ============================================================

raw_examples = [

    block.strip()

    for block in text.split("\n\n")

    if block.strip()
]


examples = []


for block in raw_examples:

    marker = "\nAssistant:"


    position = block.find(
        marker
    )


    if position == -1:

        continue


    user_text = block[
        len("User: "):
        position
    ].strip()


    answer_text = block[
        position + len(marker):
    ].strip()


    if not user_text:

        continue


    if not answer_text:

        continue


    examples.append(
        (
            user_text,
            answer_text
        )
    )


print(
    "Usable custom examples:",
    len(examples)
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


vocab_size = tokenizer.get_vocab_size()


PAD_ID = tokenizer.token_to_id(
    "<PAD>"
)

EOS_ID = tokenizer.token_to_id(
    "<EOS>"
)


if PAD_ID is None:

    raise RuntimeError(
        "Tokenizer is missing <PAD>."
    )


if EOS_ID is None:

    raise RuntimeError(
        "Tokenizer is missing <EOS>."
    )


print(
    "Tokenizer vocabulary:",
    vocab_size
)


# ============================================================
# DATASET
# ============================================================

class CustomChatDataset(Dataset):

    def __init__(
        self,
        examples
    ):

        self.samples = []

        self.questions = []


        for user_text, answer_text in examples:

            prompt = (
                "User: "
                + user_text
                + "\nAssistant:"
            )


            prompt_ids = tokenizer.encode(
                prompt
            ).ids


            answer_ids = tokenizer.encode(
                " " + answer_text
            ).ids


            ids = (
                list(prompt_ids)
                + list(answer_ids)
                + [EOS_ID]
            )


            # ------------------------------------------------
            # Skip examples larger than context.
            # ------------------------------------------------

            if len(ids) > BLOCK_SIZE + 1:

                continue


            x = list(
                ids[:-1]
            )

            y = list(
                ids[1:]
            )


            answer_start = len(
                prompt_ids
            )


            # ------------------------------------------------
            # Mask every target belonging to the prompt.
            # ------------------------------------------------

            for i in range(
                len(y)
            ):

                absolute_position = (
                    i + 1
                )


                if (
                    absolute_position
                    < answer_start
                ):

                    y[i] = -100


            # ------------------------------------------------
            # Pad.
            # ------------------------------------------------

            while len(x) < BLOCK_SIZE:

                x.append(
                    PAD_ID
                )

                y.append(
                    -100
                )


            self.samples.append(
                (
                    torch.tensor(
                        x,
                        dtype=torch.long
                    ),

                    torch.tensor(
                        y,
                        dtype=torch.long
                    )
                )
            )


            self.questions.append(
                user_text
            )


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        return self.samples[
            index
        ]


# ============================================================
# BUILD DATASET
# ============================================================

dataset = CustomChatDataset(
    examples
)


print(
    "Training samples:",
    len(dataset)
)


if len(dataset) == 0:

    raise RuntimeError(
        "No usable training samples."
    )


# ============================================================
# DATA LOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


print(
    "Batches per epoch:",
    len(loader)
)


# ============================================================
# MODEL
# ============================================================

model = MiniGPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(device)


print(
    "Model parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

if torch.cuda.is_available():

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True
    )

else:

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


# ============================================================
# AMP
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAINING
# ============================================================

best_loss = float(
    "inf"
)


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    batches = 0


    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        ncols=70,
        leave=True,
        bar_format=(
            "{desc} "
            "{bar}| "
            "{n_fmt}/{total_fmt}"
        )
    )


    for x, y in progress:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                _, loss = model(
                    x,
                    y
                )

        else:

            _, loss = model(
                x,
                y
            )


        if loss is None:

            continue


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        if use_amp:

            scaler.scale(
                loss
            ).backward()


            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )


            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )


            optimizer.step()


        total_loss += loss.item()

        batches += 1


        progress.set_postfix(
            loss=f"{loss.item():.3f}"
        )


    if batches == 0:

        raise RuntimeError(
            "No training batches were processed."
        )


    train_loss = (
        total_loss
        / batches
    )


    # --------------------------------------------------------
    # Save whenever loss improves.
    # --------------------------------------------------------

    if train_loss < best_loss:

        best_loss = train_loss


        checkpoint = {

            "model_state":
                model.state_dict(),

            "vocab_size":
                vocab_size,

            "block_size":
                BLOCK_SIZE,

            "embed_size":
                EMBED_SIZE,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "tokenizer_path":
                TOKENIZER_PATH,

            "best_loss":
                best_loss,

            "version":
                "V14_CUSTOM"
        }


        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )


        improved = True

    else:

        improved = False


    print()

    print(
        f"Train Loss: {train_loss:.4f}"
    )


    if improved:

        print(
            "New best model!"
        )


# ============================================================
# FINISHED
# ============================================================

print()

print(
    "=========================================="
)

print(
    "V14 CUSTOM TRAINING COMPLETE"
)

print(
    "Best training loss:",
    best_loss
)

print(
    "Saved:"
)

print(
    CHECKPOINT_PATH
)

print(
    "=========================================="
)