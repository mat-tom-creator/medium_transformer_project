import os
import sys
import logging
import argparse
import torch
from torch.utils.data import random_split

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.model_config import ModelConfig
from src.data.data_collector import DataCollector
from src.data.preprocessing import DataPreprocessor
from src.model.transformer import MediumTransformer
from src.training.trainer import Trainer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def main(args):
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 1. Load configuration
        logger.info("Initializing configuration...")
        config = ModelConfig()

        # 2. Collect and preprocess data
        logger.info("Collecting data...")
        collector = DataCollector(config)
        texts = collector.collect_wikitext(max_samples=args.max_samples)

        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(config)
        dataset = preprocessor.preprocess(texts)

        # 3. Split data into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # 4. Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        # 5. Initialize model
        logger.info("Initializing model...")
        model = MediumTransformer(config)

        # 6. Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 7. Initialize trainer
        logger.info("Setting up trainer...")
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            experiment_name=args.experiment_name
        )

        # 8. Start training
        logger.info("Starting training...")
        if args.resume_from:
            best_loss = trainer.train(resume_from=args.resume_from)
        else:
            best_loss = trainer.train()

        logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the transformer model")
    parser.add_argument("--max_samples", type=int, default=50000,
                      help="Maximum number of samples to use for training")
    parser.add_argument("--experiment_name", type=str, default="transformer-training",
                      help="Name for the wandb experiment")
    parser.add_argument("--resume_from", type=str,
                      help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    main(args)