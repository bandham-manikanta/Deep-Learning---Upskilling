import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def get_cifar10_loaders(data_dir, batch_size=64, num_workers=2):
    """Create CIFAR-10 loaders with properly ordered transforms"""

    # FIXED: Proper transform order
    # transform_train = transforms.Compose([
    #     transforms.Resize(224),  # Resize to ImageNet size
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
    # ])
    
    # transform_test = transforms.Compose([
    #     transforms.Resize(224),  # Resize to ImageNet size
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
    # ])

    transform_train = transforms.Compose([
        # PIL Image transforms (before ToTensor)
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.2
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        # Convert to tensor FIRST
        transforms.ToTensor(),

        # Tensor transforms (after ToTensor)
        # transforms.RandomErasing(p=0.2),  # ✅ Now gets tensor input
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download and load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]
