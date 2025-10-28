from prepocessing import Dataset
import pandas as pd

dataset = Dataset('../dataset/train.csv')

dataset.load_data()
dataset.visualizzazione() 

dataset.drop_columns(["Id", "Utilities"])

dataset.data.info()

dataset.convert_ordinal("ExterQual", ['Po', 'Fa', 'TA', 'Gd', 'Ex'])

print(dataset.data["ExterQual"])

dataset.convert_nominal("MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "Electrical", "GarageType", "PavedDrive", "MiscFeature", "SaleType", "SaleCondition")

print(dataset.data['Neighborhood_Blmngtn'].head())
print(dataset.data[dataset.data['Neighborhood_CollgCr'] == True])
