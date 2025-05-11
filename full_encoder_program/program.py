# Import necessary libraries

import pandas as pd
import torch
import os

import numpy as np

from DataLoader import *
from Node import *

from program_dataset import _create_dataset
from program_model import _run_model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset_size = 10_000_000
    test_size = 10_000

    layers = list(range(1, 11))
    atempts = 5

    setx2 = DataLoader(_create_dataset(Node.create_dataset(2), test_size), dataframe=False)
    setx3 = DataLoader(_create_dataset(Node.create_dataset(3), dataset_size), dataframe=False)
    setx4 = DataLoader(_create_dataset(Node.create_dataset(4), test_size), dataframe=False)
    train, test = setx3.split(0.9)

    all_data = []

    for layer in layers:
        best = None
        score = None

        for atempt in atempts:
            model, train_loss, validation_loss, validation_test, generalisation_x2, generalisation_x4 = _run_model(train, test, setx2, setx4, layer, device)
            all_data.append([layer, atempt, train_loss, validation_loss, validation_test, generalisation_x2, generalisation_x4])
            if best is None or \
                validation_test[2] > score[2] or \
                (validation_test[2] == score[2] and validation_test[1] > score[1]) or \
                (validation_test[2] == score[2] and validation_test[1] == score[1] and validation_test[0] > score[0]):
                best = model
                score = validation_test
        
        torch.save(best.state_dict(), f'returned/models/model_{layer}.pt')
    
    df = pd.DataFrame(all_data, columns=['layers', 'atempt', 'train_loss', 'validation_loss', 'validation_test', 'generalisation_x2', 'generalisation_x4'])
    df.to_csv('returned/layers.csv', index=False)