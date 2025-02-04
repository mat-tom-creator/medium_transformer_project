import torch
import os
import sys
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.model_config import ModelConfig
from src.model.transformer import MediumTransformer
from src.evaluation.metrics import Metrics
from transformers import GPT2Tokenizer
import argparse

# Allow ModelConfig serialization
torch.serialization.add_safe_globals([ModelConfig])

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/evaluation_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(checkpoint_path, config):
    """Load model from checkpoint"""
    try:
        model = MediumTransformer(config)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def main(args):
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting evaluation with prompt: {args.prompt}")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load configuration
        logger.info("Loading configuration...")
        config = ModelConfig()
        
        # Initialize model and load checkpoint
        logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
        model = load_model(args.checkpoint_path, config)
        model = model.to(device)
        model.eval()
        
        # Initialize tokenizer and metrics
        logger.info("Initializing tokenizer and metrics...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        metrics = Metrics(tokenizer)
        
        # Generate text from prompt
        logger.info("Generating text...")
        generated_text = metrics.generate_text(
            model,
            args.prompt,
            max_length=args.max_length
        )
        
        # Print results
        print("\nPrompt:", args.prompt)
        print("\nGenerated text:", generated_text)
        print("\nGeneration completed successfully!")
        
        # Save generated text
        output_dir = "generated_texts"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{output_dir}/generation_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {args.prompt}\n\n")
            f.write(f"Generated Text: {generated_text}\n")
        logger.info(f"Generated text saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using trained transformer model")
    parser.add_argument("--checkpoint_path", required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--prompt", default="Once upon a time",
                      help="Input prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                      help="Maximum length of generated text")
    args = parser.parse_args()
    main(args)