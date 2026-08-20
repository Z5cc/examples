import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {DEVICE}')
DATA_POINTS = 80
BATCH_SIZE = 64
EPOCHS = 8000
SEED = 66
if SEED is not None:
    torch.manual_seed(SEED)
NOISE = 3 # losses shall get close to that noise
NEURONS = 16
print(f'neurons: {NEURONS}')
WEIGHT_DECAY = 0.001 # regularization
print(f'weight_decay: {WEIGHT_DECAY}')
P = 0.03    # dropout
