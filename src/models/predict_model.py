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
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    dir_name = os.path.dirname(os.path.realpath(__file__))
    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    test = torch.load(f"data/process/test.pt")
    test_set = TensorDataset(
        torch.tensor(test[0], dtype=torch.float32),
        torch.tensor(test[1], dtype=torch.int64),
    )
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


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
