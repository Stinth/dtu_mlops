import glob
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset


def mnist():
    while not os.path.exists("data/corruptmnist"):
        print("Corrupted MNIST dataset not found. Moving up a directory...")
        os.chdir("..")

    # exchange with the corrupted mnist dataset
    test = np.load(r"data/corruptmnist/test.npz")
    train = {"images": np.empty((0, 28, 28)), "labels": np.empty((0,))}
    for train_file in glob.glob("data/corruptmnist/train*.npz"):
        x = np.load(train_file)
        train["images"] = np.concatenate([train["images"], x["images"]])
        train["labels"] = np.concatenate([train["labels"], x["labels"]])

    train_set = TensorDataset(
        torch.tensor(train["images"], dtype=torch.float32),
        torch.tensor(train["labels"], dtype=torch.int64),
    )
    test_set = TensorDataset(
        torch.tensor(test["images"], dtype=torch.float32),
        torch.tensor(test["labels"], dtype=torch.int64),
    )
    return train_set, test_set


# model = torch.load("last_epoch.pt")
# model.eval()
