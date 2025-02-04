from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.encodings = tokenizer(texts,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=max_length,
                                 return_tensors='pt')
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() 
                for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Set padding token to EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def preprocess(self, texts):
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Create dataset
        dataset = TextDataset(cleaned_texts, 
                            self.tokenizer, 
                            self.config.max_seq_length)
        return dataset
        
    def _clean_text(self, text):
        # Basic cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text