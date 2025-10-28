from sklearn.linear_model import LinearRegression
import pandas as pd
from prepocessing import Dataset

class Modello:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = LinearRegression()

    def train(self):
        X = self.dataset.data.drop("SalePrice", axis=1)
        y = self.dataset.data["SalePrice"]
        self.model.fit(X, y)

    def prediction(self, X_new):
        return self.model.predict(X_new)
    
    def evaluate(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        return score
    



    

