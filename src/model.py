from sklearn.linear_model import LinearRegression , SGDRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from prepocessing import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

class LinReg:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = LinearRegression()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def split_data(self, test_size=0.2):
        y = self.dataset.data["SalePrice"]
        X = self.dataset.data.drop("SalePrice", axis=1)
        self.X_train, self.X_test,self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def prediction(self):
        return self.model.predict(self.X_test)
    
    def evaluate(self):
        score = self.model.score(self.X_test, self.y_test)
        y_pred = self.model.predict(self.X_test)
        self.plots(y_pred)
        
        return score
    
    def plots(self, y_pred):
        sns.set_theme(style="whitegrid")

        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.show()

        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()


class Ridge:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = SGDRegressor()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def split_data(self, test_size=0.2):
        y = self.dataset.data["SalePrice"]
        X = self.dataset.data.drop("SalePrice", axis=1)
        self.X_train, self.X_test,self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def prediction(self):
        return self.model.predict(self.X_test)
    
    def evaluate(self):
        score = self.model.score(self.X_test, self.y_test)
        y_pred = self.model.predict(self.X_test)
        self.plots(y_pred)
        
        return score
    
    def plots(self, y_pred):
        sns.set_theme(style="whitegrid")
        
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.show()

        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()



    

