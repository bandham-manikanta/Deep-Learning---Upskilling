"""Training Configurations"""

import torch 

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 20
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'

DATA_DIR = './data'
MODEL_SAVE_PATH = './models'

WANDB_PROJECT = 'cnn-mnist-learning'