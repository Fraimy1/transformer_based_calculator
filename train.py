import torch
from torch.nn import functional as F
from gpt import GPT
from data_generation import generate_data_batch
from config import (BATCH_SIZE, MAX_ITERS, LEARNING_RATE, DEVICE,
                    CHECKPOINT_INTERVAL, MAX_SEQ_LEN, MAX_NUMBER)
from eval_utils import checkpoint_model, evaluate_model
from data_utils import vocab_size, decode, itos
import os


torch.manual_seed(1337)

model = GPT()
m = model.to(DEVICE)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model_folder = f'training_{max(int(n.split('_')[-1]) for n in os.listdir('model_weights')) + 1}'
os.makedirs(os.path.join('model_weights', model_folder), exist_ok=False)
# Training
for iter in range(MAX_ITERS):
    # sample a batch of data
    xb, yb = generate_data_batch(MAX_SEQ_LEN, BATCH_SIZE, ['+'], minimal=0, maximum=MAX_NUMBER)
    # Fix: properly move tensors to device and assign back to variables
    xb = xb.to(DEVICE)
    yb = yb.to(DEVICE)
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # print(decode(xb[0].tolist()), decode(yb[0].tolist()), itos[logits[0].argmax(dim=-1).item()])
    
    print(f'{iter}/{MAX_ITERS} - {loss.item()}')
    
    if (iter+1) % CHECKPOINT_INTERVAL == 0 or iter ==0:
        model_path = os.path.join('model_weights', model_folder, f'gpt_{iter+1}.pt')
        checkpoint_model(model, model_path)
        evaluate_model(model, 15, MAX_NUMBER, 30, ['+'])


#TODO:
# 0. Clean the code of unused parts - done
# 1. Implement proper evaluation function - done, up for testing
# 2. Implement example generation function for each mathematical operation - done
# 3. Think of a way to improve the model's "reasoning skills"