import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv('/content/Iris.csv')
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])
#x = df.drop(['Id', 'Species'],axis = 1, inplace = True)
df.shape
df.describe().T
df

x = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Precision, Recall, F1-Score
print("\nAccuracy Report:\n", accuracy_score(y_test, y_pred))
class_names = df['Species'].unique()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()

plt.bar(y_pred,y_test)
plt.show()
