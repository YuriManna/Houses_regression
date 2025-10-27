from prepocessing import Dataset
import pandas as pd

dataset = Dataset('dataset/train.csv')

dataset.load_data()
dataset.visualizzazione() 

