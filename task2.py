import pandas as pd
import numpy as np
df = pd.read_csv('/content/archive (1).zip')  #change folder path its a titanic dataset
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
df


from sklearn.preprocessing import StandardScaler
num_coloumn = ['Age', 'Fare','SibSp', 'Parch']
scaler = StandardScaler()
df[num_coloumn] = scaler.fit_transform(df[num_coloumn])
df.head()

import matplotlib.pyplot as plt
import seaborn as sns

graph = ['Age','Fare']

fig, axes = plt.subplots(1, 2, figsize=(18, 10))  # Adjust figsize as needed
fig.suptitle('Feature Distributions', fontsize=18, color='darkred')


for i,col in enumerate(graph):
    
    sns.boxplot(x=df[col],data = df, ax =axes[i], palette='pastel')
    axes[i].set_title(f'Boxplot of {col}')
    

plt.tight_layout(rect=[0, 0.3, 1, 0.95])  # leave space for suptitle
plt.show()


sns.pairplot(df, vars=['Age', 'Fare', 'Pclass'], hue='Survived')
plt.show() 

import matplotlib.pyplot as plt
import seaborn as sns

gra = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
discrete_cols = ['Survived', 'Pclass', 'SibSp', 'Parch']
continuous_cols = ['Age', 'Fare']

# Create a 2-row, 3-column subplot grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust figsize as needed
fig.suptitle('Feature Distributions', fontsize=18, color='darkred')

# Flatten axes array for easy indexing
axes = axes.flatten()

# Loop through features and plot on each subplot
for i, col in enumerate(gra):
    ax = axes[i]
    if col in discrete_cols:
        sns.countplot(x=col, data=df, ax=ax, palette='pastel')
    elif col in continuous_cols:
        sns.histplot(data=df, x=col, bins=20, kde=True, color='skyblue', ax=ax)

    ax.set_title(f'{col} Distribution', fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
plt.show()





