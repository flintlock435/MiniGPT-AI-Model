class CharacterTokenizer:
    def __init__(self, text):
        # Get every unique character in the training data.
        self.chars = sorted(set(text))

        # Convert characters to integer IDs.
        self.char_to_id = {
            char: i for i, char in enumerate(self.chars)
        }

        # Convert integer IDs back to characters.
        self.id_to_char = {
            i: char for i, char in enumerate(self.chars)
        }

        self.vocab_size = len(self.chars)

    def encode(self, text):
        """Convert text into a list of integer token IDs."""
        return [self.char_to_id[char] for char in text]

    def decode(self, tokens):
        """Convert token IDs back into text."""
        return "".join(self.id_to_char[token] for token in tokens)