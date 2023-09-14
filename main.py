import pandas as pd
import torch

train_data = pd.read_csv('Data/train.csv')
# Random shuffle and split data into train and validation sets
