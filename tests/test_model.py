import pytest
import torch
from src.config.model_config import ModelConfig
from src.model.transformer import MediumTransformer

def test_model_forward():
    config = ModelConfig()
    model = MediumTransformer(config)
    
    # Create dummy input
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(input_ids)
    
    # Check output shape
    expected_shape = (batch_size, seq_length, config.vocab_size)
    assert outputs.shape == expected_shape

def test_model_loss():
    config = ModelConfig()
    model = MediumTransformer(config)
    
    # Create dummy input and target
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass and loss calculation
    outputs = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)),
        targets.view(-1)
    )
    
    # Check loss
    assert not torch.isnan(loss)
    assert loss.item() > 0