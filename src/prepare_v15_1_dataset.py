import json
import os
import random
import re


# ============================================================
# V15.1 DATASET SETTINGS
# ============================================================

CUSTOM_PATH = "data/chat_training.txt"

DOLLY_PATH = "data/dolly/databricks-dolly-15k.jsonl"

OUTPUT_PATH = "data/chat_training_v15_1.txt"

SEED = 42

CUSTOM_REPEAT = 20

MAX_DOLLY_EXAMPLES = 3000

MAX_USER_CHARS = 500

MAX_ANSWER_CHARS = 800

MAX_CONTEXT_CHARS = 600


# ============================================================
# SEED
# ============================================================

random.seed(
    SEED
)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if text is None:

        return ""

    text = str(
        text
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD CUSTOM CHAT DATA
# ============================================================

def load_custom_examples():

    examples = []


    if not os.path.exists(
        CUSTOM_PATH
    ):

        raise FileNotFoundError(
            f"Custom dataset not found: {CUSTOM_PATH}"
        )


    with open(
        CUSTOM_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


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

custom_examples = load_custom_examples()


print(
    "Custom examples:",
    len(custom_examples)
)


# ============================================================
# LOAD DOLLY
# ============================================================

if not os.path.exists(
    DOLLY_PATH
):

    raise FileNotFoundError(
        f"Dolly dataset not found: {DOLLY_PATH}"
    )


dolly_examples = []

seen_dolly = set()

allowed_categories = {
    "open_qa",
    "closed_qa"
}

category_counts = {}


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


        if category not in allowed_categories:

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


        if not instruction:

            continue


        if not response:

            continue


        if len(instruction) > MAX_USER_CHARS:

            continue


        if len(response) > MAX_ANSWER_CHARS:

            continue


        if (
            context
            and len(context) > MAX_CONTEXT_CHARS
        ):

            continue


        if response.count(
            "\n"
        ) > 8:

            continue


        # ----------------------------------------------------
        # Build user message.
        # ----------------------------------------------------

        if context:

            user_text = (
                instruction
                + "\nContext: "
                + context
            )

        else:

            user_text = instruction


        if len(user_text) > MAX_USER_CHARS:

            continue


        # ----------------------------------------------------
        # Deduplicate.
        # ----------------------------------------------------

        key = (
            user_text.lower(),
            response.lower()
        )


        if key in seen_dolly:

            continue


        seen_dolly.add(
            key
        )


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


# ============================================================
# SHUFFLE AND LIMIT DOLLY
# ============================================================

random.shuffle(
    dolly_examples
)


dolly_examples = dolly_examples[
    :MAX_DOLLY_EXAMPLES
]


print(
    "Dolly examples:",
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
# BUILD FINAL DATASET
# ============================================================

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
# SHUFFLE
# ============================================================

random.shuffle(
    final_examples
)


# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_PATH
    ),
    exist_ok=True
)


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
# STATISTICS
# ============================================================

characters = sum(
    len(user_text)
    + len(answer)

    for user_text, answer in final_examples
)


print()

print(
    "=========================================="
)

print(
    "V15.1 DATASET READY"
)

print(
    "=========================================="
)

print(
    "Custom examples:",
    len(custom_examples)
)

print(
    "Custom repeat:",
    CUSTOM_REPEAT
)

print(
    "Dolly examples:",
    len(dolly_examples)
)

print(
    "Final examples:",
    len(final_examples)
)

print(
    "Characters:",
    characters
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