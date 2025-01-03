"""Inspect model checkpoint structure."""
import torch

checkpoint = torch.load('models/best_model.pt', map_location='cpu')
print('Keys in checkpoint:', checkpoint.keys())
for key, value in checkpoint.items():
    if isinstance(value, dict):
        print(f'\n{key} contains:', value.keys())
    else:
        print(f'\n{key} type:', type(value))
