import re


class SmallTokenizer:

    def __init__(self, text, vocab_size=4000):

        self.vocab_size_limit = vocab_size

        # Split text into words, punctuation, and whitespace.
        tokens = re.findall(
            r"\w+|[^\w\s]|\n",
            text,
            re.UNICODE
        )

        # Count token frequency.
        counts = {}

        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        # Special tokens.
        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

        # Most common tokens.
        most_common = sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True
        )

        available = (
            vocab_size
            - len(special_tokens)
        )

        vocabulary = (
            special_tokens
            + [
                token
                for token, _ in most_common[:available]
            ]
        )

        # Remove accidental duplicates.
        vocabulary = list(
            dict.fromkeys(vocabulary)
        )

        self.id_to_token = {
            i: token
            for i, token in enumerate(vocabulary)
        }

        self.token_to_id = {
            token: i
            for i, token in self.id_to_token.items()
        }

        self.pad_id = self.token_to_id["<PAD>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]

        self.vocab_size = len(
            self.id_to_token
        )


    def tokenize(self, text):

        return re.findall(
            r"\w+|[^\w\s]|\n",
            text,
            re.UNICODE
        )


    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):

        tokens = self.tokenize(text)

        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for token in tokens:

            ids.append(
                self.token_to_id.get(
                    token,
                    self.unk_id
                )
            )

        if add_eos:
            ids.append(self.eos_id)

        return ids


    def decode(self, ids):

        tokens = []

        for token_id in ids:

            token = self.id_to_token.get(
                int(token_id),
                "<UNK>"
            )

            if token in {
                "<PAD>",
                "<BOS>"
            }:
                continue

            if token == "<EOS>":
                break

            tokens.append(token)

        # Reconstruct readable text.
        result = ""

        for token in tokens:

            if token == "\n":

                result = result.rstrip() + "\n"

            elif not result:

                result = token

            elif (
                token in ".,!?;:)"
                or token == "'"
            ):

                result += token

            elif result.endswith(
                "(\n"
            ):

                result += token

            else:

                result += " " + token

        return result.strip()