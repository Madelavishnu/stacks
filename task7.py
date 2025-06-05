import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


df = pd.read_csv('/content/archive (5).zip')# change path breast cancer dataset
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

df.shape
df.describe().T
df

x = df.drop(['diagnosis','id'], axis = 1 )
y = df['diagnosis']

le = LabelEncoder()
y = le.fit_transform(df['diagnosis'])


from sklearn.preprocessing import StandardScaler

x = df.drop(['id', 'diagnosis'], axis=1)

# Standardize numeric features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(x_train, y_train)
print("Linear Kernel:\n", classification_report(y_test, svm_linear.predict(x_test)))

# RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(x_train, y_train)
print("RBF Kernel:\n", classification_report(y_test, svm_rbf.predict(x_test)))



# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']  # Only tuning RBF kernel here
}

# Set up GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=1)
grid.fit(x_train, y_train)

# Print best hyperparameters
print(" Best Parameters:", grid.best_params_)
print(" Best CV Score:", grid.best_score_)


# Use the best model from GridSearch (RBF SVM)
best_model = grid.best_estimator_

# Evaluate using 5-fold cross-validation
cv_scores = cross_val_score(best_model, x_scaled, y, cv=5)

# Print accuracy from each fold and average
print("Fold-wise Accuracy Scores:", cv_scores)
print(" Average Cross-Validation Accuracy:", cv_scores.mean())



# Step 1: Reduce features to 2D using PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# Step 2: Train SVM on 2D data
svm_vis = SVC(kernel='linear')
svm_vis.fit(x_pca, y)

# Step 3: Function to plot decision boundary
def plot_decision_boundary(x, y, model):
    h = 0.02  # step size in mesh
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('SVM Decision Boundary in 2D PCA Space')
    plt.show()

# Step 4: Call the plot function
plot_decision_boundary(x_pca, y, svm_vis)

