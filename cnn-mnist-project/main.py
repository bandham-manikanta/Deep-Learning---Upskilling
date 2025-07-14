import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

from src.model import SimpleCNN
from src.train import train_one_epoch, validate, log_sample_predictions
from src.utils import get_data_loaders, count_parameters
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
            "architecture": "SimpleCNN",
            "device": str(device)
        }
    )
    
    # Create model
    model = SimpleCNN().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log model info to wandb
    wandb.config.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    # Create data loaders
    print("Loading MNIST dataset...")
    train_loader, val_loader = get_data_loaders(DATA_DIR, BATCH_SIZE)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
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
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Log sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_sample_predictions(model, val_loader, device)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/best_model.pth")
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()

if __name__ == "__main__":
    main()
    print("Training completed!")