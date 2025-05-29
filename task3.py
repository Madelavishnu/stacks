import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Housing.csv')
df

le = LabelEncoder()

#yes = 1   no = 0
df['guestroom'] = le.fit_transform(df['guestroom']) 
df['basement'] = le.fit_transform(df['basement'])
df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
df['mainroad'] = le.fit_transform(df['mainroad'])
df['airconditioning'] = le.fit_transform(df['airconditioning'])
df['prefarea'] = le.fit_transform(df['prefarea'])# c =0 q =1 s =2 
df

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])
df.drop(['furnishingstatus'],axis = 1, inplace = True)
df.shape
df.describe().T

num_coloumn = ['area','stories','bedrooms','bathrooms','parking']
scaler = StandardScaler()
df[num_coloumn] = scaler.fit_transform(df[num_coloumn])
df.head()


x = df.drop(['price'],axis = 1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
y_pred = lr.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
