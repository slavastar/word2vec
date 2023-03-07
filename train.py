import argparse
import os

import torch.cuda
import torch.nn as nn
import torch.optim as optim

from dataloader import get_dataloader_and_vocab
from model import get_model_class
from training import Training
from utils import get_lr_scheduler, save_vocab, load_yaml


def train(config: dict):
    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])

    dataloader_train, vocab = get_dataloader_and_vocab(
        data_dir=config['data_dir'],
        split='train',
        model_name=config['model'],
        batch_size=config['batch_size'],
        shuffle=True,
        min_word_frequency=config['min_word_frequency'],
        n_words=config['n_words'],
        max_sequence_length=config['max_sequence_length']
    )

    dataloader_val, _ = get_dataloader_and_vocab(
        data_dir=config['data_dir'],
        split='valid',
        model_name=config['model'],
        batch_size=config['batch_size'],
        shuffle=True,
        min_word_frequency=config['min_word_frequency'],
        n_words=config['n_words'],
        max_sequence_length=config['max_sequence_length']
    )

    print(f"Vocabulary size: {len(vocab)}")

    model_class = get_model_class(config['model'])
    model = model_class(vocab_size=len(vocab), embed_size=config['embed_size'], embed_max_norm=config['embed_max_norm'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = get_lr_scheduler(optimizer, total_epochs=config['epochs'], verbose=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Training(
        model=model,
        epochs=config['epochs'],
        dataloader_train=dataloader_train,
        steps_train=config['steps_train'],
        dataloader_val=dataloader_val,
        steps_val=config['steps_val'],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config['checkpoint_frequency'],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config['model_dir']
    )

    trainer.train()
    trainer.save_loss()
    save_vocab(config['model_dir'], vocab)

    print(f"Model artifacts are saved to folder: {config['model_dir']}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file (*.yaml)')
    args = parser.parse_args()

    config = load_yaml(args.config)
    train(config)
