from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree


df = pd.read_csv("/content/archive (4).zip")  # adjust path Heart disease dataset
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf.score(X_test, y_test):.2f}")

cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")


for depth in [2, 4, 6, 8, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    print(f"Depth={depth}, Train acc={model.score(X_train, y_train):.2f}, Test acc={model.score(X_test, y_test):.2f}")


# Visualize
plt.figure(figsize=(40,20))
tree.plot_tree(model, feature_names=X.columns,label = 'all', class_names=['No Disease', 'Disease'], filled=True)
plt.show()

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

