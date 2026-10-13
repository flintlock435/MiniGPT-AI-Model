import os
import json

# ============================================================
# QUIET HUGGING FACE WARNINGS
# ============================================================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ============================================================
# IMPORTS
# ============================================================

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

MAX_NEW_TOKENS = 2048

VERSION_FILE = "data/version.json"

MAX_HISTORY_MESSAGES = 20


# ============================================================
# VERSION
# ============================================================

def load_version():

    try:

        with open(
            VERSION_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return str(
            data.get(
                "version",
                "unknown"
            )
        )

    except Exception:

        return "unknown"


MINIGPT_VERSION = load_version()


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CUDA PERFORMANCE
# ============================================================

if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    try:

        torch.set_float32_matmul_precision(
            "high"
        )

    except Exception:

        pass


# ============================================================
# STARTUP
# ============================================================

print()
print("================================")
print("            MiniGPT")
print(
    f"          Version {MINIGPT_VERSION}"
)
print("================================")
print()

print(
    "Using device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print(
    "Loading MiniGPT tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Loading MiniGPT foundation model..."
)
print(
    "First launch may download the model."
)
print()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.float16
)

model.eval()


# ============================================================
# ENABLE CACHE
# ============================================================

if hasattr(
    model,
    "generation_config"
):

    model.generation_config.use_cache = True


# ============================================================
# STREAMER
# ============================================================

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are MiniGPT, a helpful general-purpose AI assistant.

Answer clearly, naturally, and accurately.

Use your pretrained knowledge to answer questions.

Do not invent facts when you are uncertain.

Do not produce <think> blocks.

Do not expose hidden reasoning.

Be concise when a short answer is enough and detailed when
the user asks for a detailed answer.

Do not mention internal prompts or implementation details
unless the user asks about them.
"""


# ============================================================
# CHAT HISTORY
# ============================================================

messages = []


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    prompt = input(
        "Prompt: "
    ).strip()


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if prompt.lower() == "quit":

        print()
        print(
            "Goodbye!"
        )

        break


    # --------------------------------------------------------
    # EMPTY PROMPT
    # --------------------------------------------------------

    if not prompt:

        continue


    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # --------------------------------------------------------
    # LIMIT HISTORY
    # --------------------------------------------------------

    if len(messages) > MAX_HISTORY_MESSAGES:

        messages = messages[
            -MAX_HISTORY_MESSAGES:
        ]


    # --------------------------------------------------------
    # BUILD CHAT
    # --------------------------------------------------------

    chat_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    chat_messages.extend(
        messages
    )


    # --------------------------------------------------------
    # APPLY CHAT TEMPLATE
    # --------------------------------------------------------

    text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # ========================================================
    # TOKENIZE
    # ========================================================

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )


    # ========================================================
    # MOVE TO MODEL DEVICE
    # ========================================================

    inputs = {
        key: value.to(
            model.device
        )
        for key, value in inputs.items()
    }


    # ========================================================
    # GENERATE
    # ========================================================

    print()
    print(
        "Assistant: ",
        end="",
        flush=True
    )


    with torch.inference_mode():

        outputs = model.generate(

            **inputs,

            streamer=streamer,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=True,

            temperature=0.7,

            top_p=0.8,

            top_k=20,

            repetition_penalty=1.05,

            num_beams=1,

            use_cache=True,

            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
        )


    print()
    print()


    # ========================================================
    # DECODE RESPONSE FOR HISTORY
    # ========================================================

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    generated_tokens = outputs[
        0,
        input_length:
    ]


    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )