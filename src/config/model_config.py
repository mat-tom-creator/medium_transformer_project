import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model Architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 16
    max_seq_length: int = 256
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = torch.cuda.is_available()
    cuda_device: int = 0

    # Data
    train_data_path: str = "data/train.txt"
    eval_data_path: str = "data/eval.txt"