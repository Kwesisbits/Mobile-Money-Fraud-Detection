import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow

import kagglehub

# Download latest version
path = kagglehub.dataset_download("ealaxi/paysim1")

print("Path to dataset files:", path)

import os
file_list = os.listdir(path)
csv_file = [f for f in file_list if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))
display(df.head())

df.isnull.sum()
df.info()
df.describe()

df = df.drop(["nameOrig", "nameDest"], axis=1) #these columns are irrelevant for training the models 
df = pd.get_dummies(df, columns=['type'], drop_first=True) #One-hot encode categorical data
df.head()
