import argparse
import os
import sys

import click
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from data import mnist
from models.model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_checkpoint")
def visualize(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    dir_name = os.path.dirname(os.path.realpath(__file__))
    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)


cli.add_command(visualize)


if __name__ == "__main__":
    cli()
