import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import time

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch with detailed metrics"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []
    
    epoch_start = time.time()
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        batch_start = time.time()
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'epoch_time': epoch_time,
        'avg_batch_time': avg_batch_time,
        'total_samples': total
    }

def validate(model, val_loader, criterion, device):
    """Validate with detailed metrics"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            start_time = time.time()
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'avg_inference_time': avg_inference_time,
        'total_samples': total
    }

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """Estimate FLOPs (simplified)"""
    try:
        from torchprofile import profile_macs
        inputs = torch.randn(input_size)
        macs = profile_macs(model, inputs)
        return macs / 1e9  # Convert to GFLOPs
    except ImportError:
        return "Install torchprofile for FLOP calculation"

