import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = config.hidden_size // config.num_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        # Linear projections and reshape
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_size)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_size)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_size)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Calculate output
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.hidden_size)
        
        return self.out(out)