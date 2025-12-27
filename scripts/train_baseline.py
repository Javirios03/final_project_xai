#!/usr/bin/env python3
"""
Train baseline model on clean dataset
"""

import sys
from pathlib import Path

# Add Project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import json

from src.data.dataset import get_dataloaders
from src.models.baseline import get_model
from src.models.trainer import Trainer, evaluate_model
from src.utils.reproducibility import set_seed


def main():
    # Reproducibility
    set_seed(42)
    
    # Configuration
    config = {
        'seed': 42,
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'model': 'ResNet18',
        'pretrained': True
    }
    
    print(f"\n{'='*70}")
    print("BASELINE MODEL TRAINING")
    print(f"{'='*70}\n")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Paths
    data_dir = Path('data/raw/chest_xray')
    split_file = Path('data/raw/train_val_split.json')
    
    # Data
    print("📊 Loading data...\n")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=str(data_dir),
        split_file=str(split_file),
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Model
    print(f"\n🏗️  Creating model...\n")
    model = get_model(
        num_classes=2,
        pretrained=config['pretrained'],
        device=config['device']
    )
    
    # Loss with class weights
    train_dataset = train_loader.dataset
    class_weights = train_dataset.get_class_weights().to(config['device'])
    print(f"\n⚖️  Class weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device'],
        checkpoint_dir='models/checkpoints'
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], model_name='baseline')
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*70}")
    
    # Load best model
    checkpoint = torch.load('models/checkpoints/baseline_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    results = evaluate_model(model, test_loader, device=config['device'])
    
    # Save results
    results_path = Path('models/checkpoints/baseline_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Test results saved to: {results_path}")
    
    # Save config
    config_path = Path('models/checkpoints/baseline_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Config saved to: {config_path}")
    
    print(f"\n{'='*70}")
    print("✅ BASELINE MODEL TRAINING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
