import argparse
import os
import sys

import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from data import mnist


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(f"Leaarning rate: {lr}")
    dir_name = os.path.dirname(os.path.realpath(__file__))
    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.train()
    train = torch.load(f"data/process/train.pt")
    train_set = TensorDataset(
        torch.tensor(train[0], dtype=torch.float32),
        torch.tensor(train[1], dtype=torch.int64),
    )
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 30
    training_loss = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        training_loss.append(running_loss / len(trainloader))
        if (e + 1) % 5 == 0:
            print(f"[{e+1}/{epochs}] Training loss: {running_loss/len(trainloader)}")
    torch.save(model.state_dict(), f"{dir_name}/checkpoints/last_epoch.pt")

    # show training curve
    plt.plot(training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.savefig(f"reports/figures/training_curve.png")


cli.add_command(train)


if __name__ == "__main__":
    cli()
