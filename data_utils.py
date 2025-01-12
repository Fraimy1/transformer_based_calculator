import torch
from config import SUPPORTED_OPS, PAD_TOKEN, START_TOKEN, END_TOKEN

# Create vocabulary
stoi = {str(i): i for i in range(10)}  # 0-9 digits
for i, oper in enumerate(SUPPORTED_OPS, start=10):
    stoi[oper] = i
stoi[' '] = len(stoi)
stoi['='] = len(stoi)
stoi[PAD_TOKEN] = len(stoi)
stoi[END_TOKEN] = len(stoi)
stoi[START_TOKEN] = len(stoi)

itos = {v: k for k, v in stoi.items()}
vocab_size = len(stoi)

decode = lambda l: ''.join([itos[i] for i in l])