import torch
from data_generation import generate_data_batch, itos, stoi, PAD_TOKEN
from config import BLOCK_SIZE, DEVICE, END_TOKEN
import re

def _parse_result(s: str):
    s = s.replace(END_TOKEN, '').strip()
    match = re.search(r'([\-\+]?\d+(\.\d+)?)', s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def evaluate_model(model, n_examples, maximum, max_new_tokens=30, operations=['+']):
    xs, ys = generate_data_batch(BLOCK_SIZE, n_examples, operations, maximum=maximum)
    xs = xs.to(DEVICE)
    ys = ys.to(DEVICE)

    print('-' * 100)
    print("Input                  | Target                | Prediction            | Error      | Correct?")
    print('-' * 100)

    total_error = 0.0
    correct_count = 0
    PAD_IDX = stoi[PAD_TOKEN]

    with torch.no_grad():
        model.eval()
        for i in range(n_examples):
            # Decode input
            x_tokens = xs[i].tolist()
            while x_tokens and x_tokens[-1] == PAD_IDX:
                x_tokens.pop()
            input_str = ''.join(itos.get(tok, '') for tok in x_tokens)

            # Decode target
            y_tokens = ys[i].tolist()
            while y_tokens and y_tokens[-1] == PAD_IDX:
                y_tokens.pop()
            target_str = ''.join(itos.get(tok, '') for tok in y_tokens)

            # Generate prediction
            x_input = xs[i:i+1, :]
            generated = model.generate(x_input, max_new_tokens)
            if len(generated) == 0:
                continue
            gen_tokens = generated[0].tolist()
            while gen_tokens and PAD_IDX in gen_tokens:
                gen_tokens.remove(PAD_IDX)
            prediction_str = ''.join(itos.get(tok, '') for tok in gen_tokens)

            # Parse numeric results
            target_val = _parse_result(target_str)
            pred_val = _parse_result(prediction_str)
            if target_val is not None and pred_val is not None:
                error_val = abs(target_val - pred_val)
            else:
                error_val = float('inf')

            is_correct = (error_val < 1e-6)
            if is_correct:
                correct_count += 1
            if error_val != float('inf'):
                total_error += error_val

            max_len_print = 1000
            inp_print  = (input_str[:max_len_print] + '…') if len(input_str) > max_len_print else input_str
            tgt_print  = (target_str[:max_len_print] + '…') if len(target_str) > max_len_print else target_str
            pred_print = (str(pred_val) + '…') if len(prediction_str) > max_len_print else prediction_str

            print(f"{inp_print} | {tgt_print} | {pred_print} | {error_val:<10.4g} | {'Correct' if is_correct else 'Wrong'}")

        model.train()

    accuracy = correct_count / n_examples
    mean_error = total_error / n_examples
    print('-' * 100)
    print(f"Accuracy: {correct_count}/{n_examples} = {accuracy:.3f}")
    print(f"Mean numerical error: {mean_error:.4g}")
    print('-' * 100)




def checkpoint_model(model, path):
    torch.save(model.state_dict(), path)
    print('Model was saved to', path)
