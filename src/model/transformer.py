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

    # src/model/transformer.py

    def generate(self, input_ids, max_length=100, num_return_sequences=1,
                temperature=0.7, do_sample=True, pad_token_id=None,
                bos_token_id=None, eos_token_id=None, top_k=50, top_p=0.9,
                repetition_penalty=1.2, no_repeat_ngram_size=3):
        """Generate text tokens."""
        try:
            device = input_ids.device
            batch_size = input_ids.shape[0]
            cur_len = input_ids.shape[1]
            vocab_size = self.token_embedding.weight.shape[0]
            
            # Store original input for return in case of error
            original_input = input_ids.clone()
            
            for _ in range(max_length - cur_len):
                with torch.no_grad():
                    # Forward pass
                    outputs = self(input_ids)
                    next_token_logits = outputs[:, -1, :] / temperature
                    
                    # Apply filtering and sampling
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        for batch_idx in range(next_token_logits.shape[0]):
                            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                            next_token_logits[batch_idx, indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    if do_sample:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1)
                    else:
                        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for EOS token
                    if eos_token_id is not None:
                        eos_in_next = (next_tokens == eos_token_id).any().item()
                        if eos_in_next:
                            break
                    
                    # Append new tokens
                    input_ids = torch.cat([input_ids, next_tokens], dim=1)
                    
                    # Check sequence length
                    if input_ids.shape[1] >= max_length:
                        break
            
            return input_ids
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return original_input