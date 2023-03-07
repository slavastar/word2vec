import torch.nn as nn


class SkipGram(nn.Module):

    """
    Predicts the current word based on the context
    """

    def __init__(self, vocab_size: int, embed_size: int = 300, embed_max_norm: float = 1):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            max_norm=embed_max_norm
        )
        self.linear = nn.Linear(in_features=embed_size, out_features=vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)
        return x
