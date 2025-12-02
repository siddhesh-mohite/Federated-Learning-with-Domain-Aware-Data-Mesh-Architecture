"""
Run centralized training baseline experiment.
This establishes the accuracy upper bound for comparison.
"""

import sys
import os
from pathlib import Path

# FIX: Robustly add project root to path regardless of execution directory
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt

from src.models.cnn_mnist import SimpleCNN
from src.data.mnist_loader import MNISTDataLoader
from src.trainers.centralized import CentralizedTrainer


def main():
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("CENTRALIZED TRAINING BASELINE")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Load data
    print("\nLoading MNIST dataset...")
    data_loader = MNISTDataLoader()
    train_loader, test_loader = data_loader.load_centralized(batch_size=BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    
    # Create model
    print("\nCreating model...")
    model = SimpleCNN(num_classes=10)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Train
    trainer = CentralizedTrainer(model, device=DEVICE)
    history = trainer.train(
        train_loader, 
        test_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE
    )
    
    # Save results
    results_dir = project_root / "logs/centralized_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes.plot(history['train_loss'], label='Train Loss', marker='o')
    axes.plot(history['test_loss'], label='Test Loss', marker='s')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.set_title('Training and Test Loss')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    # Accuracy plot
    axes.plot(history['train_accuracy'], label='Train Accuracy', marker='o')
    axes.plot(history['test_accuracy'], label='Test Accuracy', marker='s')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Accuracy (%)')
    axes.set_title('Training and Test Accuracy')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150)
    print(f"\nPlots saved to {results_dir / 'training_curves.png'}")
    
    # Save model
    torch.save(model.state_dict(), results_dir / 'model.pt')
    print(f"Model saved to {results_dir / 'model.pt'}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Train Accuracy: {history['train_accuracy'][-1]:.2f}%")
    print(f"Final Test Accuracy:  {history['test_accuracy'][-1]:.2f}%")
    print(f"Best Test Accuracy:   {max(history['test_accuracy']):.2f}%")
    print(f"Total Training Time:  {sum(history['epoch_time']):.1f}s")
    print("=" * 60)
    print("✅ Centralized baseline complete!")
    
    return history


if __name__ == "__main__":
    main()
