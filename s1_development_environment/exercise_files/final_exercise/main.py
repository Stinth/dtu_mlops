import argparse
import os
import sys

import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader

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
    train_set, _ = mnist()
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
    torch.save(model.state_dict(), f"{dir_name}/last_epoch.pt")

    # show training curve
    plt.plot(training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.savefig(f"{dir_name}/training_curve.png")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    dir_name = os.path.dirname(os.path.realpath(__file__))
    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    _, test_set = mnist()
    testloader = DataLoader(test_set, batch_size=64, shuffle=True)
    with torch.no_grad():
        accuracy = 0
        for images, labels in testloader:
            output = model(images)

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Accuracy: {accuracy*100/len(testloader)}%")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
