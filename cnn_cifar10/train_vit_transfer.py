import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

from src.vit_transfer import VIT_Transfer
from src.train import train_one_epoch, validate, count_parameters
from src.utils import get_cifar10_loaders
from config import *

def main():
    # Set device
    device = torch.device(DEVICE)
    print(f"🚀 Training Vision Transformer on {device}")
    
    # Initialize wandb
    wandb.init(
        project="vit-vs-cnn-comparison",
        name="vit-b16-transfer-learning",
        config={
            "model": "ViT-B/16",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "AdamW",  # ViT works better with AdamW
            "weight_decay": 1e-4,
            "architecture": "Vision Transformer",
        }
    )
    
    # Create ViT model
    model = VIT_Transfer(num_classes=10, model_name='vit_b_16', freeze_backbone=False).to(device)
    total_params, trainable_params = count_parameters(model)
    
    print(f"📊 Vision Transformer Model Stats:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model: ViT-Base/16 with ImageNet pretraining")
    
    # Data loaders
    print("📁 Loading CIFAR-10 dataset...")
    train_loader, val_loader = get_cifar10_loaders(DATA_DIR, BATCH_SIZE)
    
    # Loss function and optimizer (AdamW works better for ViT)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW([
    #     {'params': model.backbone.heads.head.parameters(), 'lr': 0.01},
    #     {'params': model.backbone.encoder.layers[-1].parameters(), 'lr': 0.001},
    # ], weight_decay=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine annealing scheduler (popular for ViT)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training variables
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    min_epochs = 10
    
    
    print("🎯 Starting Vision Transformer Training...")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} - Vision Transformer")
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
        print(f"⚡ Speed      - Train: {train_metrics['avg_batch_time']*1000:.1f}ms/batch")
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
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/best_vit_b16.pth")
            print(f"🎉 New best ViT model saved! Val Acc: {val_metrics['accuracy']:.2f}%")
        else:
            patience_counter += 1
            print(f"⏰ No improvement for {patience_counter} epochs")

            if epoch >= min_epochs and patience_counter >= patience:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch + 1}")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
    
    print(f"\n🏁 Vision Transformer Training Complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    # Log final metrics
    wandb.log({
        "final_best_accuracy": best_val_acc,
        "total_parameters": total_params,
        "architecture_type": "vision_transformer"
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()
