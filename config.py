import torch

# Training config
BATCH_SIZE = 32
MAX_ITERS = 1000
LEARNING_RATE = 1e-3
CHECKPOINT_INTERVAL = MAX_ITERS // 10

# Model config
BLOCK_SIZE = 15
N_EMBED = 384
N_HEAD = 8
N_LAYER = 6
MAX_SEQ_LEN = 14
DROPOUT_RATE = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data generation config
SUPPORTED_OPS = ['+', '-', '/', '*']
MAX_NUMBER = 100
PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'