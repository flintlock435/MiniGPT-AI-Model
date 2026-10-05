import random
from pathlib import Path


CUSTOM_PATH = "data/chat_training.txt"
DOLLY_PATH = "data/chat_training_dolly.txt"
OUTPUT_PATH = "data/chat_training_v11.txt"

RANDOM_SEED = 42


# ============================================================
# Load files
# ============================================================

custom_path = Path(CUSTOM_PATH)
dolly_path = Path(DOLLY_PATH)

if not custom_path.exists():
    raise FileNotFoundError(
        f"Missing custom dataset: {CUSTOM_PATH}"
    )

if not dolly_path.exists():
    raise FileNotFoundError(
        f"Missing Dolly dataset: {DOLLY_PATH}"
    )


custom_text = custom_path.read_text(
    encoding="utf-8"
)

dolly_text = dolly_path.read_text(
    encoding="utf-8"
)


# ============================================================
# Split into complete examples
# ============================================================

custom_examples = [
    block.strip()
    for block in custom_text.split("\n\n")
    if block.strip()
]

dolly_examples = [
    block.strip()
    for block in dolly_text.split("\n\n")
    if block.strip()
]


print(
    "Custom examples:",
    len(custom_examples)
)

print(
    "Dolly examples:",
    len(dolly_examples)
)


# ============================================================
# Deduplicate
# ============================================================

seen = set()
combined = []


def add_example(example):

    key = example.lower()

    if key in seen:
        return

    seen.add(key)

    combined.append(example)


for example in custom_examples:
    add_example(example)

for example in dolly_examples:
    add_example(example)


# ============================================================
# Shuffle
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(
    combined
)


# ============================================================
# Save
# ============================================================

output_text = (
    "\n\n".join(combined)
    + "\n\n"
)


output_path = Path(
    OUTPUT_PATH
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_path.write_text(
    output_text,
    encoding="utf-8"
)


# ============================================================
# Statistics
# ============================================================

print()
print(
    "V11 dataset created."
)

print(
    "Total examples:",
    len(combined)
)

print(
    "Total characters:",
    len(output_text)
)

print(
    "Saved to:",
    OUTPUT_PATH
)