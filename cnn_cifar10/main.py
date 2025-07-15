import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.model import VGG_CIFAR10
from src.train import train_one_epoch, validate, log_sample_predictions
from src.utils import get_cifar10_loaders, count_parameters
from config import *

def main():
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "momentum": MOMENTUM,
            "architecture": "VGG_CIFAR10",
            "device": str(device),
            "early_stopping_patience": 15
        }
    )
    
    # Create model
    model = VGG_CIFAR10().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log model info to wandb
    wandb.config.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    # Create data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader = get_cifar10_loaders(DATA_DIR, BATCH_SIZE)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM,
        weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    min_epochs = 10
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")  # Show current LR
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr
        })
        
        # Log sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_sample_predictions(model, val_loader, device)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/best_basic_cnn.pth")
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⏰ No improvement for {patience_counter} epochs")

            if epoch >= min_epochs and patience_counter >= patience:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch + 1}")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()

if __name__ == "__main__":
    main()
