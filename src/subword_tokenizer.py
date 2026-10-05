import tiktoken


class SubwordTokenizer:

    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")

        self.vocab_size = self.encoder.n_vocab

    def encode(self, text):
        return self.encoder.encode(
            text,
            allowed_special=set()
        )

    def decode(self, tokens):
        return self.encoder.decode(tokens)