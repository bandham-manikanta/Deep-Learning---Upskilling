import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
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
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output, 1)
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
    
    # CIFAR-10 classes
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Denormalize images for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    wandb_images = []
    for i in range(num_samples):
        # Denormalize image
        img = images[i].cpu()
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()  # CHW to HWC
        
        true_label = classes[labels[i].item()]
        pred_label = classes[predictions[i].item()]
        
        wandb_images.append(wandb.Image(
            img, 
            caption=f"True: {true_label}, Pred: {pred_label}"
        ))
    
    wandb.log({"sample_predictions": wandb_images})
