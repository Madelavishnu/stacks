import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/content/archive (6).zip')# change path mall customer segmentation dataset

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])
df.shape
df.describe().T
df


x= df.drop(['CustomerID','Gender'], axis = 1)
y = df['Gender']

lb = LabelEncoder()
y = lb.fit_transform(y)

standard = StandardScaler()
x_scaled = standard.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size= 0.2, random_state = 42)

kmean = KMeans(n_clusters=5)
y_kmeans = kmean.fit_predict(x_scaled)

score = silhouette_score(x_scaled, y_kmeans)
print(f"Silhouette Score: {score}")

# Optional: Visualize clusters in 2D using PCA

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=y_kmeans, palette='viridis')
plt.title("Customer Segments (KMeans Clusters)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


