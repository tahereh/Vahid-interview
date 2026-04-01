import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Loading Data & Overview
data_path = 'case-data.csv'
data = pd.read_csv(data_path)

print(data.head())
print(data.shape)

# Preprocessing
# Checking for missining values 
missing = pd.DataFrame({
    "Missing Count": data.isnull().sum(),
    "Missing %": (data.isnull().sum() / len(data)) * 100
})

print(missing)

total_missing = data.isnull().sum().sum()
total_values = data.size

percentage_missing = (total_missing / total_values) * 100
print("Total Missing %:", percentage_missing)

# Cleaning and seperating numerical and categorical data
data = data.drop(columns=['Unnamed: 0'])
data['clicks'] = data['clicks'].fillna(0)  # Alternatively, we can use the median or drop 'clicks,' as it does not play any role in the following modeling.