import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(
        self,
        text,
        tokenizer,
        block_size=128,
        num_samples=None
    ):

        self.tokenizer = tokenizer

        self.block_size = block_size

        self.tokens = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long
        )

        # Need block_size + 1 tokens
        # to create x and y pairs.

        self.available = (
            len(self.tokens)
            - block_size
        )

        if self.available < 1:
            raise ValueError(
                "Dataset is too small for "
                f"block_size={block_size}"
            )

        if num_samples is None:

            self.num_samples = self.available

        else:

            self.num_samples = min(
                num_samples,
                self.available
            )


    def __len__(self):

        return self.num_samples


    def __getitem__(self, index):

        # Wrap around so that we can request
        # a fixed number of samples.

        start = index % self.available

        x = self.tokens[
            start:
            start + self.block_size
        ]

        y = self.tokens[
            start + 1:
            start + self.block_size + 1
        ]

        return x, y