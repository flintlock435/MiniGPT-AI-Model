import math
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer

from src.model import MiniGPT


# ============================================================
# V15.1 SETTINGS
# ============================================================

DATA_PATH = "data/chat_training_v15_1.txt"

TOKENIZER_PATH = "data/v12_tokenizer.json"

CHECKPOINT_PATH = "checkpoints/v15_1_best_model.pt"

BLOCK_SIZE = 256

BATCH_SIZE = 64

EPOCHS = 40

LEARNING_RATE = 3e-4

MIN_LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

TRAIN_RATIO = 0.90

EMBED_SIZE = 128

NUM_HEADS = 4

NUM_LAYERS = 4

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
# CUDA OPTIMIZATION
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
# LOAD EXAMPLES
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
# TRAIN / VALIDATION SPLIT
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

print(
    "PAD ID:",
    PAD_ID
)

print(
    "EOS ID:",
    EOS_ID
)


# ============================================================
# DATASET
# ============================================================

class V15_1_Dataset(Dataset):

    def __init__(
        self,
        examples
    ):

        self.samples = []

        self.skipped = 0

        self.skipped_too_long = 0

        self.skipped_bad_format = 0


        for example in examples:

            # ------------------------------------------------
            # Find the exact separator.
            # ------------------------------------------------

            marker = "\nAssistant:"


            marker_position = example.find(
                marker
            )


            if marker_position == -1:

                self.skipped += 1

                self.skipped_bad_format += 1

                continue


            # ------------------------------------------------
            # Extract user text.
            # ------------------------------------------------

            user_text = example[
                len("User: "):
                marker_position
            ].strip()


            # ------------------------------------------------
            # Extract answer.
            # ------------------------------------------------

            answer_text = example[
                marker_position
                + len(marker):
            ].strip()


            if not user_text:

                self.skipped += 1

                self.skipped_bad_format += 1

                continue


            if not answer_text:

                self.skipped += 1

                self.skipped_bad_format += 1

                continue


            # ------------------------------------------------
            # Prompt.
            # ------------------------------------------------

            prompt = (
                "User: "
                + user_text
                + "\nAssistant:"
            )


            prompt_ids = tokenizer.encode(
                prompt
            ).ids


            # ------------------------------------------------
            # Answer.
            # ------------------------------------------------

            answer_ids = tokenizer.encode(
                " "
                + answer_text
            ).ids


            # ------------------------------------------------
            # Full sequence.
            # ------------------------------------------------

            ids = (
                list(prompt_ids)
                + list(answer_ids)
                + [EOS_ID]
            )


            # ------------------------------------------------
            # Every training example must fit in the context.
            # ------------------------------------------------

            if len(ids) > BLOCK_SIZE + 1:

                self.skipped += 1

                self.skipped_too_long += 1

                continue


            # ------------------------------------------------
            # Inputs and targets.
            # ------------------------------------------------

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
            # Only learn from answer tokens.
            # ------------------------------------------------

            for i in range(
                len(y)
            ):

                target_position = (
                    i + 1
                )


                if target_position < answer_start:

                    y[i] = -100


            # ------------------------------------------------
            # Pad to fixed context length.
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
# BUILD DATASETS
# ============================================================

print()

print(
    "Building training dataset..."
)


train_dataset = V15_1_Dataset(
    train_examples
)


print(
    "Building validation dataset..."
)


val_dataset = V15_1_Dataset(
    val_examples
)


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset
)

)

print(
    "Skipped training:",
    train_dataset.skipped
)

print(
    "  Bad format:",
    train_dataset.skipped_bad_format
)

print(
    "  Too long:",
    train_dataset.skipped_too_long
)

print(
    "Skipped validation:",
    val_dataset.skipped
)

print(
    "  Bad format:",
    val_dataset.skipped_bad_format
)

print(
    "  Too long:",
    val_dataset.skipped_too_long
)


if len(train_dataset) == 0:

    raise RuntimeError(
        "No usable training samples."
    )


# ============================================================
# DATA LOADERS
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
# LEARNING RATE SCHEDULE
# ============================================================

total_steps = (
    EPOCHS
    * len(train_loader)
)


warmup_steps = max(
    50,
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
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# VALIDATION
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
# TRAINING
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
        # Forward pass.
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
        # Backward pass.
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


        total_loss += (
            loss.item()
        )

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
    # Validation.
    # --------------------------------------------------------

    val_loss = evaluate()


    current_lr = (
        scheduler.get_last_lr()[0]
    )


    print()

    print(
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"LR: {current_lr:.7f}"
    )


    # --------------------------------------------------------
    # Save best checkpoint.
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
                "15.1.0"
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
    "V15.1 TRAINING COMPLETE"
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