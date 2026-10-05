import json
import os
import random
import re


# ============================================================
# V13 DATASET SETTINGS
# ============================================================

CUSTOM_PATH = "data/chat_training.txt"

DOLLY_PATH = "data/dolly/databricks-dolly-15k.jsonl"

OUTPUT_PATH = "data/chat_training_v13.txt"

SEED = 42

CUSTOM_REPEAT = 12

MAX_DOLLY_EXAMPLES = 3500

MIN_RESPONSE_LENGTH = 2

MAX_RESPONSE_LENGTH = 1200


# ============================================================
# SEED
# ============================================================

random.seed(
    SEED
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    "data",
    exist_ok=True
)


# ============================================================
# NORMALIZE WHITESPACE
# ============================================================

def clean_text(text):

    if text is None:

        return ""

    text = str(text)

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.strip()

    return text


# ============================================================
# EXTRACT CUSTOM CHAT
# ============================================================

def extract_custom_examples(text):

    examples = []


    blocks = [
        block.strip()
        for block in text.split("\n\n")
        if block.strip()
    ]


    for block in blocks:

        # ----------------------------------------------------
        # User / Assistant
        # ----------------------------------------------------

        match = re.search(
            r"User:\s*(.*?)\s*Assistant:\s*(.*)",
            block,
            flags=re.DOTALL
        )


        if match:

            user = clean_text(
                match.group(1)
            )

            answer = clean_text(
                match.group(2)
            )


            if user and answer:

                examples.append(
                    (
                        user,
                        answer
                    )
                )

                continue


        # ----------------------------------------------------
        # Instruction / Response
        # ----------------------------------------------------

        match = re.search(
            r"Instruction:\s*(.*?)\s*Response:\s*(.*)",
            block,
            flags=re.DOTALL
        )


        if match:

            user = clean_text(
                match.group(1)
            )

            answer = clean_text(
                match.group(2)
            )


            if user and answer:

                examples.append(
                    (
                        user,
                        answer
                    )
                )

                continue


        # ----------------------------------------------------
        # Question / Answer
        # ----------------------------------------------------

        match = re.search(
            r"Question:\s*(.*?)\s*Answer:\s*(.*)",
            block,
            flags=re.DOTALL
        )


        if match:

            user = clean_text(
                match.group(1)
            )

            answer = clean_text(
                match.group(2)
            )


            if user and answer:

                examples.append(
                    (
                        user,
                        answer
                    )
                )


    return examples


# ============================================================
# LOAD CUSTOM DATA
# ============================================================

custom_examples = []


if os.path.exists(
    CUSTOM_PATH
):

    with open(
        CUSTOM_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        custom_text = file.read()


    custom_examples = extract_custom_examples(
        custom_text
    )


print(
    "Custom examples:",
    len(custom_examples)
)


# ============================================================
# LOAD DOLLY
# ============================================================

selected_categories = {
    "open_qa",
    "closed_qa",
    "brainstorming"
}


dolly_examples = []

category_counts = {}


if not os.path.exists(
    DOLLY_PATH
):

    raise FileNotFoundError(
        f"Dolly dataset not found: {DOLLY_PATH}"
    )


with open(
    DOLLY_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()


        if not line:

            continue


        try:

            item = json.loads(
                line
            )

        except json.JSONDecodeError:

            continue


        category = clean_text(
            item.get(
                "category",
                ""
            )
        )


        if category not in selected_categories:

            continue


        instruction = clean_text(
            item.get(
                "instruction",
                ""
            )
        )


        context = clean_text(
            item.get(
                "context",
                ""
            )
        )


        response = clean_text(
            item.get(
                "response",
                ""
            )
        )


        if not instruction or not response:

            continue


        if len(response) < MIN_RESPONSE_LENGTH:

            continue


        if len(response) > MAX_RESPONSE_LENGTH:

            continue


        # ----------------------------------------------------
        # Put context inside the user message.
        # ----------------------------------------------------

        if context:

            user_text = (
                instruction
                + "\n\nContext:\n"
                + context
            )

        else:

            user_text = instruction


        dolly_examples.append(
            (
                user_text,
                response
            )
        )


        category_counts[category] = (
            category_counts.get(
                category,
                0
            )
            + 1
        )


print(
    "Dolly selected before cap:",
    len(dolly_examples)
)


print(
    "Dolly categories:"
)


for category, count in sorted(
    category_counts.items()
):

    print(
        f"  {category}: {count}"
    )


# ============================================================
# CAP DOLLY
# ============================================================

random.shuffle(
    dolly_examples
)


dolly_examples = dolly_examples[
    :MAX_DOLLY_EXAMPLES
]


print(
    "Dolly selected:",
    len(dolly_examples)
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

unique_examples = []

seen = set()


for user_text, answer in (
    custom_examples
    + dolly_examples
):

    key = (
        user_text.lower().strip(),
        answer.lower().strip()
    )


    if key in seen:

        continue


    seen.add(
        key
    )


    unique_examples.append(
        (
            user_text,
            answer
        )
    )


print(
    "Unique examples:",
    len(unique_examples)
)


# ============================================================
# SEPARATE CUSTOM DATA AGAIN
# ============================================================

custom_set = set(
    (
        user.lower().strip(),
        answer.lower().strip()
    )
    for user, answer in custom_examples
)


final_examples = []


# ------------------------------------------------------------
# Repeat custom examples heavily.
# ------------------------------------------------------------

for _ in range(
    CUSTOM_REPEAT
):

    for user_text, answer in custom_examples:

        final_examples.append(
            (
                user_text,
                answer
            )
        )


# ------------------------------------------------------------
# Add Dolly once.
# ------------------------------------------------------------

for user_text, answer in dolly_examples:

    final_examples.append(
        (
            user_text,
            answer
        )
    )


# ============================================================
# FINAL SHUFFLE
# ============================================================

random.shuffle(
    final_examples
)


# ============================================================
# WRITE DATASET
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    for user_text, answer in final_examples:

        file.write(
            "User: "
            + user_text
            + "\nAssistant: "
            + answer
            + "\n\n"
        )


# ============================================================
# STATS
# ============================================================

character_count = sum(
    len(user) + len(answer)
    for user, answer in final_examples
)


print()

print(
    "=========================================="
)

print(
    "V13 DATASET READY"
)

print(
    "Final examples:",
    len(final_examples)
)

print(
    "Characters:",
    character_count
)

print(
    "Custom examples:",
    len(custom_examples)
)

print(
    "Custom repetitions:",
    CUSTOM_REPEAT
)

print(
    "Dolly examples:",
    len(dolly_examples)
)

print(
    "Saved:"
)

print(
    OUTPUT_PATH
)

print(
    "=========================================="
)