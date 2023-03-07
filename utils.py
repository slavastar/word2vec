import os
import yaml
from yaml.loader import SafeLoader

import torch
from torch.optim.lr_scheduler import LambdaLR


def load_yaml(path: str | dict) -> dict:
    if isinstance(path, str):
        with open(path) as file:
            return yaml.load(file, Loader=SafeLoader)
    return path


def save_yaml(path: str, data: dict):
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    return LambdaLR(optimizer, lr_lambda=lambda epoch: (total_epochs - epoch) / total_epochs, verbose=verbose)


def save_vocab(model_dir: str, vocab):
    vocab_path = os.path.join(model_dir, 'vocab.pt')
    torch.save(vocab, vocab_path)
