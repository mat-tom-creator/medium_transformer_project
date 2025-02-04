import torch
import numpy as np
from typing import Dict
from transformers import GPT2Tokenizer

class Metrics:
    def __init__(self, tokenizer: GPT2Tokenizer):
        self.tokenizer = tokenizer
    
    def calculate_perplexity(self, model, data_loader) -> float:
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids']
                outputs = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += input_ids.numel()
        
        return torch.exp(torch.tensor(total_loss / total_tokens))
    
    def generate_text(self, model, prompt: str, 
                     max_length: int = 100) -> str:
        model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, 
                                        return_tensors='pt')
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(output_ids[0])