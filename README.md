# kaggle-titanic
Goal: Predict survival on the Titanic based on features / parameters provided in the dataset

### Data Directory
Data files are pulled from Kaggle's Titanic competition, that contains the following:
- train.csv: model train and validation dataset
- test.csv: model test dataset
- generate_submission.py: dataset that contains passenger id and gender

### Clean Data Directory
CSV files within this directory are cleaned data files that are ready for model training and testing.

### Data Processing
The following has been done to the data provided in the train and test dataset:
- Convert **Sex** data into binary values
- Missing values in the **Age** column are filled with the mean age
- Clean **Ticket** data by removing chars and retaining only the ticket number
- Convert **Embarked** data into numerical categorical values
- Dropped columns: **PassengerId**, **Name**, **Cabin**

### Model
Logistic Regression:  
- Accuracy: 58.66%
