import torch

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
MOMENTUM = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
DATA_DIR = "./data"
MODEL_SAVE_PATH = "./models"

# Wandb
WANDB_PROJECT = "cnn-cifar10"

# CIFAR-10 normalization (better than 0.5, 0.5, 0.5)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
