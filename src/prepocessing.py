import pandas as pd

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load data from a CSV file."""
        self.data = pd.read_csv(self.file_path, na_values=["", "NaN", "None"], keep_default_na=False)
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

    def fill_NaN (self):
        for cols in self.data.columns:
            if self.data[cols].isnull().any():
                if self.data[cols].dtype == 'int64':
                    self.fill_NaN_int(cols)
                elif self.data[cols].dtype == 'float64':
                    self.fill_NaN_double(cols)
                else:
                    self.fill_NaN_string(cols)

    def fill_NaN_int (self, col):
        median = self.data[col].median()
        self.data[col] = self.data[col].fillna(median)

    def fill_NaN_double (self, col):
        mean = self.data[col].mean()
        self.data[col] = self.data[col].fillna(mean)

    def fill_NaN_string (self, col):
        mode = self.data[col].mode()[0]
        self.data[col] = self.data[col].fillna(mode)

    def convert_ordinal (self, col, values):
        for i, v in enumerate(values):
            self.data[col].loc[self.data[col] == v] = i