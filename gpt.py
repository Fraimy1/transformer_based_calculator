import torch
import torch.nn as nn
from torch.nn import functional as F
from config import (BLOCK_SIZE, N_EMBED, N_HEAD, N_LAYER, 
                   DROPOUT_RATE, DEVICE, END_TOKEN, PAD_TOKEN)
from data_utils import vocab_size, stoi, itos, decode
from data_generation import generate_data_batch

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.keys = nn.Linear(N_EMBED, head_size*num_heads, bias=False)
        self.queries = nn.Linear(N_EMBED, head_size*num_heads, bias=False)
        self.values = nn.Linear(N_EMBED, head_size*num_heads, bias=False)
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.num_heads = num_heads
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    
    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        
        k = self.keys(x).view(B, self.num_heads, T, C//self.num_heads)
        q = self.queries(x).view(B, self.num_heads, T, C//self.num_heads)
        v = self.values(x).view(B, self.num_heads, T, C//self.num_heads)

        # compute raw attention weights
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, nh, T, T)

        # causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # padding mask (optional, if provided)
        if pad_mask is not None:
            # broadcast pad_mask to (B, 1, 1, T) or (B,1,T,1) depending on your logic
            # then compare current dimension to match shape (B, nh, T, T)
            # Typically you'd shape it as (B, 1, 1, T) then broadcast
            wei = wei.masked_fill(pad_mask, float('-inf'))

        # softmax
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v  # (B, nh, T, head_size)
        out = out.view(B, T, C)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, N_EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT_RATE)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, N_EMBED, N_HEAD):
        super().__init__()
        head_size = N_EMBED // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)
        self.ffwd = FeedForward(N_EMBED)
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x, pad_mask=None):
        x = x + self.sa(self.ln1(x), pad_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEAD=N_HEAD) for i in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # create a pad_mask to block PAD tokens in attention
        pad_mask = (idx == stoi[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # pass pad_mask to each Block so that MHA can use it
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)

        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=stoi[PAD_TOKEN])

        return logits, loss

    def generate(self, idx, max_new_tokens):
        with torch.no_grad():
            self.eval()
            output = torch.tensor([])
            output = output.to(DEVICE)
            for _ in range(max_new_tokens):
                if idx.size(1) > BLOCK_SIZE:
                    idx_cond = idx[:, -BLOCK_SIZE:]
                else:
                    idx_cond = idx

                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.argmax(probs, dim=1, keepdim=True)
                if idx_next.item() == stoi[END_TOKEN]:
                    break
                idx = torch.cat((idx, idx_next), dim=1)
                output = torch.cat((output, idx_next), dim=1)
            self.train()
        return output

if __name__ == '__main__':
    model = GPT()
    model.to(DEVICE)
    x, y = generate_data_batch(
        max_seq_len=15,
        batch_size=1,
        operations=['+'],
        maximum=10**3
    )
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    logits, loss = model(x, y)
    print(logits.shape, loss)
    print(x, y)
    print('-'*100)
    preds = model.generate(x, 30)
    print(preds)
    print('-'*100)
    print(decode(preds[0].tolist()))
    print("Done!")