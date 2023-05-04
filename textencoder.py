import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, heads, dropout):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, texts):
        embeddings = self.embedding(texts)
        embeddings = embeddings.permute(1, 0, 2)
        features = self.transformer(embeddings)
        features = features.permute(1, 0, 2)
        features = self.norm(features)
        return features

enc = TextEncoder( 
    vocab_size=8, 
    embed_dim=16, 
    num_layers=4, 
    heads=8, 
    dropout=0.2
)

print(enc)