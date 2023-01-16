# -*- coding: utf-8 -*-
import glob
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    test = np.load(input_filepath + r"/test.npz")
    test_images = torch.tensor(test["images"])
    test_labels = torch.tensor(test["labels"])

    train = {"images": np.empty((0, 28, 28)), "labels": np.empty((0,))}
    for train_file in glob.glob(input_filepath + "/train*.npz"):
        x = np.load(train_file)
        train["images"] = np.concatenate([train["images"], x["images"]])
        train["labels"] = np.concatenate([train["labels"], x["labels"]])

    train_images = torch.tensor(train["images"])
    train_labels = torch.tensor(train["labels"])

    # normalize images
    test_images = (test_images - test_images.mean()) / test_images.std()
    train_images = (train_images - train_images.mean()) / train_images.std()

    torch.save((test_images, test_labels), output_filepath + "/test.pt")
    torch.save((train_images, train_labels), output_filepath + "/train.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
