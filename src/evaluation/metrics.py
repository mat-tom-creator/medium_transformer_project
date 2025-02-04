import torch
import numpy as np
from typing import Dict
from transformers import GPT2Tokenizer
import logging

class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
    
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
    
    def generate_text(self, model, prompt: str, max_length: int = 100) -> str:
        try:
            model = model.to(self.device)
            model.eval()
            
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            with torch.no_grad():
                # Generate
                outputs = model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return f"Error generating text: {str(e)}"