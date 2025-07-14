import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir, batch_size=64, num_workers = 2):
    """Create train and validation data loaders from MNIST dataset"""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (.3081,))])
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.MNIST(
        data_dir, 
        train=True,
        download=True,
        transform=transform
        )
    test_dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
    )

    return train_loader, test_loader

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params