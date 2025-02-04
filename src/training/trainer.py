# src/training/trainer.py

import os
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import wandb
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Any,
        experiment_name: str = "medium-transformer"
    ):
        """
        Initialize the trainer.
        Args:
            model: The model to train
            optimizer: The optimizer to use
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            config: Configuration object containing training parameters
            experiment_name: Name for the wandb experiment
        """
        # Setup logging first
        self.setup_logging()
        
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device and GPU optimizations
        self.setup_device()
        
        # Initialize wandb
        self.setup_wandb(experiment_name)
        
        # Create checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)
        
    def setup_logging(self):
        """Initialize logger with proper format."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def setup_device(self):
        """Setup device and GPU optimizations."""
        if not hasattr(self.config, 'cuda_device'):
            self.config.cuda_device = 0
            
        self.device = torch.device(f"cuda:{self.config.cuda_device}" 
                                if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            # Print GPU info
            gpu_name = torch.cuda.get_device_name(self.config.cuda_device)
            gpu_memory = torch.cuda.get_device_properties(self.config.cuda_device).total_memory / 1e9
            self.logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
            
            # Setup mixed precision training
            if self.config.fp16:
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                self.scaler = None
        else:
            self.logger.warning("CUDA is not available. Training on CPU.")
            self.scaler = None
            
    def setup_wandb(self, experiment_name: str):
        """Initialize wandb logging."""
        try:
            wandb.init(
                project=experiment_name,
                config={
                    "architecture": self.model.__class__.__name__,
                    "optimizer": self.optimizer.__class__.__name__,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "max_epochs": self.config.max_epochs,
                    "device": str(self.device),
                    "fp16": self.config.fp16,
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {str(e)}")
            self.logger.warning("Continuing without wandb logging")
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb and console."""
        try:
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to wandb: {str(e)}")
            
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save a model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'config': self.config
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                
            # Save regular checkpoint
            path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, path)
            self.logger.info(f"Saved checkpoint: {path}")
            
            # Save best model separately if it's the best so far
            if is_best:
                best_path = 'checkpoints/best_model.pt'
                torch.save(checkpoint, best_path)
                self.logger.info(f"Saved best model: {best_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            
    def load_checkpoint(self, path: str):
        """Load a model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            return checkpoint['epoch'], checkpoint['loss']
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            return 0, float('inf')
            
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        try:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Log memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                if gpu_memory > 7.0:  # Alert if using more than 7GB
                    self.logger.warning(f"High GPU memory usage: {gpu_memory:.2f} GB")
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.config.fp16):
                outputs = self.model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
                
            # Check for invalid loss
            if not torch.isfinite(loss):
                self.logger.error("Loss is not finite!")
                return float('inf')
                
            # Backward pass with gradient scaling if using mixed precision
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
            return loss.item()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                self.logger.error("GPU out of memory. Trying to recover...")
                return float('inf')
            self.logger.error(f"Error in train_step: {str(e)}")
            raise e

    def validate(self) -> float:
        """Run validation and return average loss."""
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast('cuda', enabled=self.config.fp16):
                    outputs = self.model(input_ids)
                    loss = torch.nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        labels.view(-1)
                    )
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else float('inf')
        
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        start_epoch = 0
        best_val_loss = float('inf')
        
        # Load checkpoint if resuming
        if resume_from:
            start_epoch, _ = self.load_checkpoint(resume_from)
            self.logger.info(f"Resuming from epoch {start_epoch}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        try:
            for epoch in range(start_epoch, self.config.max_epochs):
                self.logger.info(f"Starting epoch {epoch + 1}/{self.config.max_epochs}")
                
                # Training phase
                self.model.train()
                epoch_loss = 0
                num_batches = 0
                
                with tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.max_epochs}") as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        # Monitor memory periodically
                        if batch_idx % 100 == 0 and torch.cuda.is_available():
                            gpu_memory = torch.cuda.memory_allocated() / 1e9
                            self.logger.info(f"GPU Memory used: {gpu_memory:.2f} GB")
                        
                        batch_loss = self.train_step(batch)
                        
                        if batch_loss == float('inf'):
                            self.logger.warning(f"Skipping batch {batch_idx} due to infinite loss")
                            continue
                            
                        epoch_loss += batch_loss
                        num_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f'{batch_loss:.4f}',
                            'avg_loss': f'{epoch_loss/num_batches:.4f}',
                            'batch': f'{batch_idx}/{len(self.train_loader)}'
                        })
                        
                        # Log batch metrics
                        if batch_idx % 10 == 0:
                            self.log_metrics({
                                'batch_loss': batch_loss,
                                'batch': batch_idx + epoch * len(self.train_loader),
                                'learning_rate': self.optimizer.param_groups[0]['lr'],
                                'gpu_memory_gb': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                            })
                
                # Calculate epoch metrics
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
                self.logger.info(f"Running validation for epoch {epoch + 1}...")
                val_loss = self.validate()
                
                # Log epoch metrics
                self.log_metrics({
                    'epoch': epoch,
                    'train_loss': avg_epoch_loss,
                    'val_loss': val_loss
                })
                
                # Save checkpoint for every epoch
                self.save_checkpoint(epoch, val_loss, is_best=False)
                
                # Save best model separately if validation improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    self.logger.info(f"New best validation loss: {val_loss:.4f}")
                
                # Log to console
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.max_epochs} completed - "
                    f"Train Loss: {avg_epoch_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Best Val Loss: {best_val_loss:.4f}"
                )
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user.")
            self.save_checkpoint(epoch, avg_epoch_loss, is_best=False)
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
        finally:
            # Cleanup
            if wandb.run is not None:
                wandb.finish()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return best_val_loss