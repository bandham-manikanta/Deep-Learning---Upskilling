import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

from src.efficientnet_transfer import EfficientNet_Transfer
from src.train import train_one_epoch, validate, count_parameters
from src.utils import get_cifar10_loaders
from config import *

def main():
    # Set device
    device = torch.device(DEVICE)
    print(f"🚀 Training EfficientNet-B3 on {device}")
    
    # Initialize wandb
    wandb.init(
        project="resnet-vs-efficientnet-cifar10",
        name="efficientnet-b5-frozen",
        config={
            "model": "EfficientNet-B3",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "SGD",
            "weight_decay": 1e-4,
            "architecture": "Transfer Learning",
        }
    )
    
    # Create EfficientNet-B3 model
    model = EfficientNet_Transfer(num_classes=10, freeze_backbone=True, variant='b5').to(device)
    total_params, trainable_params = count_parameters(model)
    
    print(f"📊 EfficientNet-B3 Model Stats:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Data loaders
    print("📁 Loading CIFAR-10 dataset...")
    train_loader, val_loader = get_cifar10_loaders(DATA_DIR, BATCH_SIZE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        # Higher LR for classifier (learning from scratch)
        {'params': model.backbone.classifier.parameters(), 'lr': 0.01},
        # Lower LR for last feature block (fine-tuning)
        {'params': model.backbone.features[-1].parameters(), 'lr': 0.001},
    ], momentum=0.9, weight_decay=1e-4)

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=1e-4)
    
    # Learning rate scheduler  
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # Training variables
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    min_epochs = 10
    
    print("🎯 Starting EfficientNet-B5 Training...")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} - EfficientNet-B5")
        print('='*60)
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"📈 Training   - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"📊 Validation - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"⚡ Speed      - Train: {train_metrics['avg_batch_time']*1000:.1f}ms/batch, Val: {val_metrics['avg_inference_time']*1000:.1f}ms/batch")
        print(f"📚 Learning Rate: {current_lr:.6f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics['loss'],
            "train_accuracy": train_metrics['accuracy'],
            "val_loss": val_metrics['loss'],
            "val_accuracy": val_metrics['accuracy'],
            "learning_rate": current_lr,
            "train_batch_time_ms": train_metrics['avg_batch_time'] * 1000,
            "val_inference_time_ms": val_metrics['avg_inference_time'] * 1000,
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/best_efficientnet_b3.pth")
            print(f"🎉 New best EfficientNet-B3 model saved! Val Acc: {val_metrics['accuracy']:.2f}%")
        else:
            patience_counter += 1
            print(f"⏰ No improvement for {patience_counter} epochs")

            if epoch >= min_epochs and patience_counter >= patience:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch + 1}")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
    
    print(f"\n🏁 EfficientNet-B3 Training Complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    # Log final metrics
    wandb.log({
        "final_best_accuracy": best_val_acc,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()

