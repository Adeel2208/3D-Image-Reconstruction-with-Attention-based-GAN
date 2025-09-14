import os
import torch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_model_path = os.path.join(os.path.dirname(filename), 'best_model.pth.tar')
        torch.save(state, best_model_path)
    print(f"Checkpoint saved at {filename}")