import pandas as pd

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load data from a CSV file."""
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def visualizzazione(self):
        """Visualizza le prime righe del dataset."""
        if self.data is not None:
            print(self.data.head())
        else:
            print("Data not loaded. Please load the data first.")

    def drop_columns(self, columns):
        """Drop specified columns from the dataset."""
        if self.data is not None:
            self.data = self.data.drop(columns=columns)
        else:
            print("Data not loaded. Please load the data first.")
