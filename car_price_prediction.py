import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veri seti okunur.
df = pd.read_csv("cclass.csv")

# Eksik veri kontrolü
print(df.isnull().sum())
#df.dropna(inplace=True)

# Aykırı değerler temizlenir.
q_low = df["price"].quantile(0.01)
q_high = df["price"].quantile(0.99)
df = df[(df["price"] >= q_low) & (df["price"] <= q_high)]

# Yeni özellikler oluşturulur.
df["car_age"] = 2025 - df["year"]
df["mileage_per_year"] = df["mileage"] / df["car_age"]
df["mileage_per_year"] = df["mileage_per_year"].replace([np.inf, -np.inf], np.nan)
df["mileage_per_year"] = df["mileage_per_year"].fillna(df["mileage_per_year"].median())


# Gereksiz sütunlar çıkartılır.
df.drop(["year", "model"], axis=1, inplace=True)

# 4. One-hot encoding/Kategorik verileri dönüştür
df = pd.get_dummies(df, columns=["transmission", "fuelType"], drop_first=False)

# 5. Fiyatı logaritmik olarak dönüştür
y = np.log1p(df["price"])  # log(price + 1)
X = df.drop("price", axis=1)

# 6. Eğitim ve test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Ölçekleme (SVM, KNN, Linear için)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 8. Model eğitimi ve optimizasyon

