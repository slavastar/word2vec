import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from word2vec.src.model import SkipGram


class Training:

    def __init__(self, model, epochs: int, dataloader_train, steps_train, dataloader_val, steps_val,
                 checkpoint_frequency, criterion, optimizer, lr_scheduler, device, model_dir):
        self.model = model
        self.epochs = epochs
        self.dataloader_train = dataloader_train
        self.steps_train = steps_train
        self.dataloader_val = dataloader_val
        self.steps_val = steps_val
        self.checkpoint_frequency = checkpoint_frequency
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir

        self.loss = {'train': [], 'val': []}
        self.model.to(device)

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()

            print("Epoch {} / {}. Train loss: {:.4f}, Validation loss: {:.4f}".format(
                epoch + 1,
                self.epochs,
                self.loss['train'][-1],
                self.loss['val'][-1]
            ))

            self.lr_scheduler.step()

            if (epoch + 1) % self.checkpoint_frequency == 0:
                self._save_checkpoint(epoch + 1)

    def _train_epoch(self):
        self.model.train()
        losses = []

        for i, data in enumerate(self.dataloader_train, 1):
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if isinstance(self.model, SkipGram):
                labels = F.one_hot(labels, num_classes=outputs.shape[1]).sum(1).float()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if i == self.steps_train:
                break

        epoch_loss = np.mean(losses)
        self.loss['train'].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for i, data in enumerate(self.dataloader_val, 1):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.model(inputs)
                if isinstance(self.model, SkipGram):
                    labels = F.one_hot(labels, num_classes=outputs.shape[1]).sum(1).float()
                loss = self.criterion(outputs, labels)

                losses.append(loss.item())

                if i == self.steps_val:
                    break

        epoch_loss = np.mean(losses)
        self.loss['val'].append(epoch_loss)

    def _save_checkpoint(self, epoch: int):
        model_path = f'checkpoint_{str(epoch).zfill(3)}.pt'
        model_path = os.path.join(self.model_dir, model_path)
        torch.save(self.model, model_path)

    def save_model(self):
        model_path = os.path.join(self.model_dir, 'model.pt')
        torch.save(self.model, model_path)

    def save_loss(self):
        loss_path = os.path.join(self.model_dir, 'loss.json')
        with open(loss_path, 'w') as file:
            json.dump(self.loss, file)
