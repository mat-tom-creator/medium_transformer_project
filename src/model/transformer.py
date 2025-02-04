import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x, mask=None):
        # Attention with residual connection
        attended = self.attention(x, mask)
        x = self.norm1(x + attended)
        
        # Feed forward with residual connection
        fed_forward = self.feed_forward(x)
        x = self.norm2(x + fed_forward)
        
        return x

class MediumTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, 
                                          config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length,
                                             config.hidden_size)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        positions = torch.arange(0, seq_length, 
                               dtype=torch.long, 
                               device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits