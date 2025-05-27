#data cleaning and preprocessing 

import pandas as pd
import numpy as np
df = pd.read_csv('C:/Users/SAI RAM/Downloads/archive (1).zip')  #change folder path its a titanic dataset
print("Missing Values Per Column:")

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])
df.drop(['Cabin'],axis = 1, inplace = True)
df.shape
df.describe().T

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex']) # male = 1 female = 0
df['Embarked'] = le.fit_transform(df['Embarked'])# c =0 q =1 s =2 
df.head()
print(df.columns)

from sklearn.preprocessing import StandardScaler
num_coloumn = ['Age', 'Fare','SibSp', 'Parch']
scaler = StandardScaler()
df[num_coloumn] = scaler.fit_transform(df[num_coloumn])
df.head()


import matplotlib.pyplot as plt
import seaborn as sns
num_coloumn = ['Age', 'Fare','SibSp', 'Parch']
for col in num_coloumn:
  plt.figure(figsize=(6,4))
  sns.boxplot(x= df[col])
  plt.title(f"boxplot of {col}")
  plt.show()

def remove(df,colu):
  Q1  = df[colu].quantile(0.25)
  Q3 = df[colu].quantile(0.75)
  IQR = Q3-Q1
  lower_bound = Q1 -1.5*IQR
  upper_bound = Q3 -1.5*IQR
  return df[(df[colu]>=lower_bound)&(df[colu]<=upper_bound)]
for col in num_coloumn:
  df = remove(df,col)

#updated visulaization

import matplotlib.pyplot as plt
import seaborn as sns
num_coloumn = ['Age', 'Fare','SibSp', 'Parch']
for col in num_coloumn:
  plt.figure(figsize=(6,4))
  sns.boxplot(x= df[col])
  plt.title(f"boxplot of {col}")
  plt.show()
