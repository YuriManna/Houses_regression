from prepocessing import Dataset
from model import Modello
import pandas as pd

dataset = Dataset('../dataset/test.csv')

dataset.load_data()
dataset.visualizzazione() 

dataset.drop_columns(["Id", "Utilities"])

dataset.data.info()

regularity_map = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
uti_map = {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}
slope_map = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
quality_map = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
rating_map = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
functionality_map = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
finished_map = {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
fence_map = {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
exposure_map = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

for col, mapping in [("LotShape", regularity_map),
                     #("Utilities", uti_map),
                     ("LandSlope", slope_map),
                     ("ExterQual", quality_map),                     
                     ("ExterCond", quality_map),                     
                     ("BsmtQual", quality_map),                     
                     ("BsmtCond", quality_map),
                     ("BsmtExposure", exposure_map),
                     ("BsmtFinType1", finished_map),
                     ("BsmtFinType2", finished_map),
                     ("HeatingQC", quality_map),
                     ("KitchenQual", quality_map),
                     ("Functional", functionality_map),
                     ("FireplaceQu", quality_map),
                     ("GarageFinish", finished_map),
                     ("GarageQual", quality_map),
                     ("GarageCond", quality_map),
                     ("PoolQC", quality_map),
                     ("Fence", fence_map)]:
    dataset.map_column(col, mapping)

dataset.convert_nominal(["MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "Electrical", "GarageType", "PavedDrive", "MiscFeature", "SaleType", "SaleCondition"])

dataset.convert_string_to_number("MasVnrArea")
dataset.convert_string_to_number("LotFrontage")
dataset.convert_string_to_number("GarageYrBlt")



dataset.fill_NaN()

dataset.export_data('../dataset/test_clean.csv')
'''
dataset = Dataset('../dataset/train_clean.csv')
dataset.load_data()
model = Modello(dataset)
model.train()

model.prediction
'''