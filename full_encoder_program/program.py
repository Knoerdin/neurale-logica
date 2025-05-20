# Import necessary libraries

import pandas as pd
import torch
import os

import numpy as np
import tqdm

from DataLoader import *
from Node import *

from program_dataset import _create_dataset
from program_model import _run_model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset_size = 10_000_000
    test_size = 10_000

    layers = list(range(1, 7))
    attempts = 5
    runs = 3

    all_data = []
    contaminations = []

    for run in range(runs):
        if not os.path.exists(f'returned/models/run_{run}'):
            os.makedirs(f'returned/models/run_{run}')

        fullset = DataLoader(_create_dataset(Node.create_dataset(3), dataset_size), dataframe=False)

        setx2 = DataLoader(_create_dataset(Node.create_dataset(2), test_size), dataframe=False)
        setx4 = DataLoader(_create_dataset(Node.create_dataset(4), test_size), dataframe=False)

        train, test = fullset.split(0.9)

        setx3, _ = test.split(test_size / len(test))

        contamination_validation = 0
        testx2_validation = 0
        testx3_validation = 0
        testx4_validation = 0


        for i in tqdm.tqdm(range(test_size), desc='Calculating contamination', bar_format='{desc:<20}{percentage:3.0f}%|{bar:25}{r_bar}'):
            if test._database['encoder_input'][i] in train._database['encoder_input']: 
                contamination_validation += 1
            if setx2._database['encoder_input'][i] in train._database['encoder_input']:
                testx2_validation += 1
            if setx3._database['encoder_input'][i] in train._database['encoder_input']:
                testx3_validation += 1
            if setx4._database['encoder_input'][i] in train._database['encoder_input']:
                testx4_validation += 1
        contamination_validation /= test_size
        testx2_validation /= test_size
        testx3_validation /= test_size
        testx4_validation /= test_size

        contaminations.append([run, contamination_validation, testx2_validation, testx3_validation, testx4_validation])
        df = pd.DataFrame(contaminations, columns=['run', 'contamination', 'testx2', 'testx3', 'testx4'])
        df.to_csv('returned/contamination.csv', index=False)

        for layer in layers:
            best = None
            score = None

            for attempt in range(attempts):
                model, train_loss, validation_loss, generalisation_x3, generalisation_x2, generalisation_x4 = _run_model(train, test, setx2, setx4, layer, device,test_size = test_size)
                all_data.append([run, layer, attempt, train_loss, validation_loss, generalisation_x2, generalisation_x3, generalisation_x4])
                if best is None or \
                    generalisation_x3[2] > score[2] or \
                    (generalisation_x3[2] == score[2] and generalisation_x3[1] > score[1]) or \
                    (generalisation_x3[2] == score[2] and generalisation_x3[1] == score[1] and generalisation_x3[0] > score[0]):
                    best = model
                    score = generalisation_x3
            df = pd.DataFrame(all_data, columns=['run', 'layers', 'attempt', 'train_loss', 'validation_loss', 'generalisation_x2', 'generalisation_x3', 'generalisation_x4'])
            df.to_csv('returned/layers.csv', index=False)
            torch.save(best.state_dict(), f'returned/models/run_{run}/model_{layer}.pt')
    
    