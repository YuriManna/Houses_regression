from prepocessing import Dataset
import pandas as pd

dataset = Dataset('../dataset/train.csv')

dataset.load_data()
dataset.visualizzazione() 

dataset.drop_columns(["Id", "Utilities"])

dataset.data.info()

dataset.convert_ordinal("ExterQual", ['Po', 'Fa', 'TA', 'Gd', 'Ex'])

print(dataset.data["ExterQual"])