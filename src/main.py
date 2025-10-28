from prepocessing import Dataset
import pandas as pd

dataset = Dataset('../dataset/train.csv')

dataset.load_data()
dataset.visualizzazione() 

dataset.drop_columns(["Id", "Utilities"])

dataset.data.info()

none_keys = []
for i, cols in enumerate(dataset.data.columns):
    if dataset.data[cols].isnull().any():
        none_keys.append(cols)
print(len(none_keys))

dataset.fill_NaN()

none_keys = []
for i, cols in enumerate(dataset.data.columns):
    if dataset.data[cols].isnull().any():
        none_keys.append(cols)
print(len(none_keys))