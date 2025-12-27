"""
Training utilities for chest X-ray classifier
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
import json
from pathlib import Path
from typing import Optional, Dict, Any


class Trainer:
    """
    Trainer class for chest X-ray classification models
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'models/checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.current_epoch = 0
    
    def train_epoch(self) -> tuple:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> tuple:
        """Validate on validation set"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        auc = roc_auc_score(all_labels, all_probs)
        
        return epoch_loss, epoch_acc, precision, recall, f1, auc
    
    def train(self, num_epochs: int, model_name: str = 'baseline'):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train
            model_name: Name for saving checkpoints
        """
        print(f"\n{'='*70}")
        print(f"TRAINING {model_name.upper()} MODEL")
        print(f"{'='*70}\n")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Checkpoint dir: {self.checkpoint_dir}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_prec)
            self.history['val_recall'].append(val_rec)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"\n📊 Metrics:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"   Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f}")
            print(f"   Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(model_name, epoch, is_best=True)
                print(f"\n   ✅ Best model saved (val_acc: {val_acc:.4f})")
        
        # Save final model
        self.save_checkpoint(model_name, num_epochs-1, is_best=False)
        
        # Save training history
        self.save_history(model_name)
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Final validation accuracy: {val_acc:.4f}\n")
    
    def save_checkpoint(self, model_name: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / f'{model_name}_best.pth'
        else:
            path = self.checkpoint_dir / f'{model_name}_final.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self, model_name: str):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / f'{model_name}_history.json'
        
        # Convert numpy types to Python types
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"💾 Training history saved to: {history_path}")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device to run on
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    print(f"\n{'='*70}")
    print("EVALUATING MODEL ON TEST SET")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist()
    }
    
    # Print results
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {cm}")
    print(f"   [[TN  FP]")
    print(f"    [FN  TP]]\n")
    
    return results
