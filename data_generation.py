import torch
from typing import List, Tuple
from torch import Tensor
from config import SUPPORTED_OPS, PAD_TOKEN, START_TOKEN, END_TOKEN
from data_utils import stoi, itos
import math

torch.manual_seed(1337)

def calculate_result(a: int, b: int, op: str) -> float:
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b if b != 0 else float('inf')

def generate_data_batch(
    max_seq_len: int,
    batch_size: int,
    operations: List[str],
    maximum: int,
    minimal: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    For each example:
      - input = "n1 op n2 = "
      - target = "result<END>"
    Both are padded to (max_seq_len,).
    Returns: (input_tensors, target_tensors) of shape (batch_size, max_seq_len).
    """

    
    input_tensors = []
    target_tensors = []
    
    while len(input_tensors) < batch_size:
        n1 = torch.randint(minimal, maximum, (1,)).item()
        n2 = torch.randint(minimal, maximum, (1,)).item()
        op = operations[torch.randint(0, len(operations), (1,)).item()]
        
        result = calculate_result(n1, n2, op)
        if not torch.isfinite(torch.tensor(result)):
            continue
        
        # Build strings
        problem_str = f"{n1} {op} {n2} = "
        result_str = f"{result}"  # e.g. "123<END>"
        
        # Convert to token IDs
        problem_tokens = [stoi[ch] for ch in problem_str]
        result_tokens  = [stoi[ch] for ch in result_str] + [stoi[END_TOKEN]]

        # Skip if either side is longer than max_seq_len
        if len(problem_tokens) > max_seq_len:
            raise ValueError(f"Problem too long {len(problem_tokens)} > {max_seq_len}: '{problem_str}'")
        elif len(result_tokens) > max_seq_len:
            raise ValueError(f"Result too long {len(result_tokens)} > {max_seq_len}: '{result_str}'")
        # Pad each to max_seq_len
        problem_tokens += [stoi[PAD_TOKEN]] * (max_seq_len - len(problem_tokens))
        result_tokens  += [stoi[PAD_TOKEN]] * (max_seq_len - len(result_tokens))

        # Append to batch
        input_tensors.append(torch.tensor(problem_tokens, dtype=torch.long))
        target_tensors.append(torch.tensor(result_tokens, dtype=torch.long))
    
    return torch.stack(input_tensors), torch.stack(target_tensors)



if __name__ == "__main__":
    x, y = generate_data_batch(
        max_seq_len=15,
        batch_size=1,
        operations=['+'],
        maximum=100
    )
    
    for i in range(len(x)):
        input_str = ''.join(itos[t.item()] for t in x[i])
        target_str = ''.join(itos[t.item()] for t in y[i])
        print(f"Input:  {input_str}")
        print(f"Target: {target_str}")