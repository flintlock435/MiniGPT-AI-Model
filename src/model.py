import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        assert embed_size % num_heads == 0

        self.num_heads = num_heads
        self.head_size = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.output = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, sequence_length, embed_size = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split embedding into multiple attention heads.
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        # Causal attention: don't allow the model to look into the future.
        attention = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)

        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device),
            diagonal=1
        ).bool()

        attention = attention.masked_fill(mask, float("-inf"))

        attention = F.softmax(attention, dim=-1)

        output = attention @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embed_size
        )

        return self.output(output)


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.attention = SelfAttention(
            embed_size,
            num_heads
        )

        self.feed_forward = FeedForward(embed_size)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=64,
        embed_size=128,
        num_heads=4,
        num_layers=4
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embed_size
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_size,
                    num_heads
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_size)

        self.output = nn.Linear(
            embed_size,
            vocab_size
        )

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape

        token_embeddings = self.token_embedding(idx)

        positions = torch.arange(
            sequence_length,
            device=idx.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.blocks(x)

        x = self.norm(x)

        logits = self.output(x)

        loss = None

        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape

            logits = logits.view(
                batch_size * sequence_length,
                vocab_size
            )

            targets = targets.view(
                batch_size * sequence_length
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss