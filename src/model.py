from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression , Ridge as SKRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Modello(ABC):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.set_model()

    @abstractmethod
    def set_model(self):
        pass

    def split_data(self, test_size=0.2):
        y = self.dataset.data["SalePrice"]
        X = self.dataset.data.drop("SalePrice", axis=1)
        self.X_train, self.X_test,self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def prediction(self):
        return self.model.predict(self.X_test)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        self.plots(y_pred)


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

class LinReg(Modello):
    def __init__(self, dataset):
        super().__init__(dataset)

    def set_model(self):
        self.model = LinearRegression()

class Ridge(Modello):
    def __init__(self, dataset):
        super().__init__(dataset)

    def set_model(self):
        self.model = SKRidge(alpha=1.0, random_state=42)