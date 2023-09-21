import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

train_data = pd.read_csv('Data/train.csv')

## Data Preprocessing
# Turn into DataFrame
train_data = pd.DataFrame(train_data)
# TODO: Find ranges to better fill empty values
train_data = train_data.replace(r'', np.nan, regex=True)

# Random shuffle and split data into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
train_targets = train_data['Survived'].values
train_features = train_data.drop(['Survived'], axis=1).values
val_targets = val_data['Survived'].values
val_features = val_data.drop(['Survived'], axis=1).values
print(train_features)
print(train_targets)
