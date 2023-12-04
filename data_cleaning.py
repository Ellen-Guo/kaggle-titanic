import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('Data/train.csv')

## Data Preprocessing
train_data = pd.DataFrame(train_data)
train_data = train_data.replace(r'', np.nan, regex=True)

# Encoding textual data into numerical values
## Binarize sex column
lb = LabelBinarizer()
train_data['Sex'] = lb.fit_transform(train_data['Sex'])
## Fill empty age values with mean
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_data['Age'] = np.round(imp.fit_transform(train_data[['Age']]))
## Remove non-numerical values from Ticket
train_data['Ticket'] = [re.sub("[^0-9]", "", i) for i in train_data['Ticket']]
## Encode Embarked column
enc = OrdinalEncoder()
train_data['Embarked'] = enc.fit_transform(train_data[['Embarked']])
train_data['Embarked'] = enc.set_params(encoded_missing_value=-1).fit_transform(train_data[['Embarked']])
cleaned_data = train_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
# Random shuffle and split data into train and validation sets
cleaned_train_data, val_data = train_test_split(cleaned_data, test_size=0.2, random_state=42)
train_targets = cleaned_train_data['Survived'].values
train_features = cleaned_train_data.drop(['Survived'], axis=1).values
val_targets = val_data['Survived'].values
val_features = val_data.drop(['Survived'], axis=1).values

train = pd.concat(
    [
        pd.DataFrame(train_features, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']), 
        pd.DataFrame(train_targets, columns=['Survived'])
    ], axis=1)

val = pd.concat(
    [
        pd.DataFrame(val_features, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']),
        pd.DataFrame(val_targets, columns=['Survived'])
    ], axis=1)

train.to_csv('Clean_Data/train_cleaned.csv')
val.to_csv('Clean_Data/val_cleaned.csv')