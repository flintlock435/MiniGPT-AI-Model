import math
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer

from src.model import MiniGPT


# ============================================================
# V13 SETTINGS
# ============================================================

DATA_PATH = "data/chat_training_v13.txt"

TOKENIZER_PATH = "data/v12_tokenizer.json"

CHECKPOINT_PATH = "checkpoints/v13_best_model.pt"

BLOCK_SIZE = 128

BATCH_SIZE = 256

EPOCHS = 15

LEARNING_RATE = 3e-4

MIN_LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

TRAIN_RATIO = 0.90

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
# CHECK FILES
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
# EXAMPLES
# ============================================================

examples = [

    block.strip()

    for block in text.split("\n\n")

    if block.strip()
]


print(
    "Total examples:",
    len(examples)
)


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    examples
)


# ============================================================
# SPLIT
# ============================================================

split_index = int(
    len(examples)
    * TRAIN_RATIO
)


train_examples = examples[
    :split_index
]

val_examples = examples[
    split_index:
]


print(
    "Training examples:",
    len(train_examples)
)

print(
    "Validation examples:",
    len(val_examples)
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

class ChatDataset(Dataset):

    def __init__(
        self,
        examples,
        tokenizer,
        block_size
    ):

        self.samples = []


        for example in examples:

            # ------------------------------------------------
            # Find exact training format.
            # ------------------------------------------------

            marker = "\nAssistant:"


            position = example.find(
                marker
            )


            if position == -1:

                continue


            user_text = example[
                len("User: "):
                position
            ].strip()


            answer_text = example[
                position + len(marker):
            ].strip()


            if not user_text:

                continue


            if not answer_text:

                continue


            # ------------------------------------------------
            # Encode separately.
            # ------------------------------------------------

            prompt = (
                "User: "
                + user_text
                + "\nAssistant:"
            )


            prompt_ids = tokenizer.encode(
                prompt
            ).ids


            answer_ids = tokenizer.encode(
                " "
                + answer_text
            ).ids


            ids = (
                list(prompt_ids)
                + list(answer_ids)
                + [EOS_ID]
            )


            answer_start = len(
                prompt_ids
            )


            # ------------------------------------------------
            # Create windows.
            # ------------------------------------------------

            start = 0


            while start < len(ids) - 1:

                chunk = ids[
                    start:
                    start + block_size + 1
                ]


                if len(chunk) < 2:

                    break


                x = list(
                    chunk[:-1]
                )

                y = list(
                    chunk[1:]
                )


                # ------------------------------------------------
                # Mask prompt targets.
                # ------------------------------------------------

                for local_index in range(
                    len(y)
                ):

                    absolute_position = (
                        start
                        + local_index
                        + 1
                    )


                    if (
                        absolute_position
                        < answer_start
                    ):

                        y[local_index] = -100


                # ------------------------------------------------
                # Skip windows with no answer loss.
                # ------------------------------------------------

                if not any(
                    value != -100
                    for value in y
                ):

                    start += block_size

                    continue


                # ------------------------------------------------
                # Padding.
                # ------------------------------------------------

                while len(x) < block_size:

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


                start += block_size


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
# DATASETS
# ============================================================

print()

print(
    "Building training dataset..."
)


train_dataset = ChatDataset(
    train_examples,
    tokenizer,
    BLOCK_SIZE
)


print(
    "Building validation dataset..."
)


val_dataset = ChatDataset(
    val_examples,
    tokenizer,
    BLOCK_SIZE
)


print(
    "Training windows:",
    len(train_dataset)
)

print(
    "Validation windows:",
    len(val_dataset)
)


# ============================================================
# LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


print(
    "Batches per epoch:",
    len(train_loader)
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
# SCHEDULE
# ============================================================

total_steps = (
    EPOCHS
    * len(train_loader)
)


warmup_steps = max(
    20,
    total_steps // 20
)


def lr_lambda(
    step
):

    if step < warmup_steps:

        return (
            step + 1
        ) / warmup_steps


    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps - warmup_steps
    )


    progress = min(
        1.0,
        max(
            0.0,
            progress
        )
    )


    minimum_ratio = (
        MIN_LEARNING_RATE
        / LEARNING_RATE
    )


    return (
        minimum_ratio
        + (
            1.0
            - minimum_ratio
        )
        * 0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda
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
# VALIDATE
# ============================================================

def evaluate():

    model.eval()

    total_loss = 0.0

    batches = 0


    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )


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


            if loss is not None:

                total_loss += loss.item()

                batches += 1


    model.train()


    if batches == 0:

        return float("inf")


    return (
        total_loss
        / batches
    )


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float(
    "inf"
)


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    batches = 0


    progress = tqdm(
        train_loader,
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


        scheduler.step()


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


    val_loss = evaluate()


    current_lr = scheduler.get_last_lr()[0]


    print()

    print(
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"LR: {current_lr:.7f}"
    )


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss


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

            "best_val_loss":
                best_val_loss,

            "version":
                "V13"
        }


        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )


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
    "V13 TRAINING COMPLETE"
)

print(
    "Best validation loss:",
    best_val_loss
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