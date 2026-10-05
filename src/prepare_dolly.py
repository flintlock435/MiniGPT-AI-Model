import json
import re
from pathlib import Path


INPUT_PATH = "data/dolly/databricks-dolly-15k.jsonl"

OUTPUT_PATH = "data/chat_training_dolly.txt"

MAX_INSTRUCTION_LENGTH = 1000
MAX_CONTEXT_LENGTH = 1500
MAX_RESPONSE_LENGTH = 1500


input_path = Path(INPUT_PATH)

if not input_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_PATH}"
    )


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


examples = []
seen = set()

with input_path.open(
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        instruction = clean_text(
            record.get("instruction", "")
        )

        context = clean_text(
            record.get("context", "")
        )

        response = clean_text(
            record.get("response", "")
        )

        if not instruction or not response:
            continue

        if len(instruction) > MAX_INSTRUCTION_LENGTH:
            continue

        if len(context) > MAX_CONTEXT_LENGTH:
            continue

        if len(response) > MAX_RESPONSE_LENGTH:
            continue

        if context:

            question = (
                instruction
                + "\n\nContext: "
                + context
            )

        else:

            question = instruction

        key = (
            question.lower(),
            response.lower()
        )

        if key in seen:
            continue

        seen.add(key)

        examples.append(
            (
                question,
                response
            )
        )


print(
    "Usable examples:",
    len(examples)
)


output = []

for question, response in examples:

    output.append(
        "User: "
        + question
        + "\n"
    )

    output.append(
        "Assistant: "
        + response
        + "\n\n"
    )


text = "".join(output)


output_path = Path(
    OUTPUT_PATH
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_path.write_text(
    text,
    encoding="utf-8"
)


print(
    "Characters:",
    len(text)
)

print(
    "Examples:",
    len(examples)
)

print(
    "Saved to:",
    OUTPUT_PATH
)