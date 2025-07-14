import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def log_sample_predictions(model, val_loader, device, num_samples=8):
    """Log some sample predictions to wandb"""
    model.eval()
    images, labels = next(iter(val_loader))
    images, labels = images[:num_samples], labels[:num_samples]
    
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # MNIST normalization constants
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    
    # Create wandb images
    wandb_images = []
    for i in range(num_samples):
        # Convert to numpy and handle grayscale properly
        img = images[i].cpu().squeeze()
        img = img * MNIST_STD + MNIST_MEAN  # Reverse normalization
        img = torch.clamp(img, 0, 1)  # Ensure [0,1] range
        img = img.numpy()
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        
        wandb_images.append(wandb.Image(
            img, 
            caption=f"True: {true_label}, Pred: {pred_label}"
        ))
    
    wandb.log({"sample_predictions": wandb_images})
