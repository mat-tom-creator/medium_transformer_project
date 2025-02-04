import torch
from src.config.model_config import ModelConfig
from src.model.transformer import MediumTransformer
from src.evaluation.metrics import Metrics
from transformers import GPT2Tokenizer
import argparse

def main(args):
    # Load configuration
    config = ModelConfig()
    
    # Initialize model and load checkpoint
    model = MediumTransformer(config)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize tokenizer and metrics
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    metrics = Metrics(tokenizer)
    
    # Generate text from prompt
    generated_text = metrics.generate_text(
        model,
        args.prompt,
        max_length=args.max_length
    )
    print("Generated text:", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_length", type=int, default=100)
    args = parser.parse_args()
    main(args)