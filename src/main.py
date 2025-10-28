from prepocessing import Dataset
import pandas as pd

dataset = Dataset('../dataset/train.csv')

dataset.load_data()
dataset.visualizzazione() 

dataset.drop_columns(["Id", "Utilities"])

dataset.data.info()

dataset.convert_ordinal("ExterQual", ['Po', 'Fa', 'TA', 'Gd', 'Ex'])

print(dataset.data["ExterQual"])

dataset.convert_nominal("MSZoning", "Street", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition")
