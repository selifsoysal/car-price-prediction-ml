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

# Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params, n_iter=10, cv=3, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# SVM
svm_params = {
    'C': [10, 100],
    'gamma': ['scale', 0.01],
    'kernel': ['rbf']
}
svm_model = RandomizedSearchCV(SVR(), svm_params, n_iter=4, cv=3, n_jobs=-1)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# KNN
knn_params = {
    'n_neighbors': list(range(3, 15))
}
knn_model = RandomizedSearchCV(KNeighborsRegressor(), knn_params, n_iter=5, cv=3, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# 9. Değerlendirme fonksiyonu
def evaluate_model(name, y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} için R2 Skoru: {r2:.4f}")
    print(f"{name} için RMSE: {rmse:.2f}")
    print("-" * 40)

# 10. Modelleri değerlendir
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("Linear Regression", y_test, y_pred_lr)

# 11. Yeni araç örneğini tahmin et
new_car = pd.DataFrame([{
    "year": 2015,
    "mileage": 49850,
    "transmission": "Automatic",
    "fuelType": "Diesel",
    "engineSize": 2.1
}])

new_car["car_age"] = 2025 - new_car["year"]
new_car["mileage_per_year"] = new_car["mileage"] / new_car["car_age"]
new_car.drop("year", axis=1, inplace=True)

new_car = pd.get_dummies(new_car, columns=["transmission", "fuelType"], drop_first=False)

# Eğitimdeki sütunlarla eşleştirme
missing_cols = set(X.columns) - set(new_car.columns)
for col in missing_cols:
    new_car[col] = 0
new_car = new_car[X.columns]  # Aynı sıralama

# Ölçekleme
new_car_scaled = scaler.transform(new_car)

rf_pred = np.expm1(rf_model.predict(new_car)[0])
svm_pred = np.expm1(svm_model.predict(new_car_scaled)[0])
knn_pred = np.expm1(knn_model.predict(new_car_scaled)[0])
lr_pred = np.expm1(lr_model.predict(new_car_scaled)[0])

print(f"Tahmini Fiyat (Random Forest): {rf_pred:.2f}")
print(f"Tahmini Fiyat (SVM): {svm_pred:.2f}")
print(f"Tahmini Fiyat (KNN): {knn_pred:.2f}")
print(f"Tahmini Fiyat (Linear Regression): {lr_pred:.2f}")


y_test_real = np.expm1(y_test)
y_pred_rf_real = np.expm1(y_pred_rf)
y_pred_rf_real = np.expm1(y_pred_rf)
y_pred_svm_real = np.expm1(y_pred_svm)
y_pred_knn_real = np.expm1(y_pred_knn)
y_pred_lr_real = np.expm1(y_pred_lr)

# Grafik (Random Forest)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, y_pred_rf_real, alpha=0.5, color='teal')
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Random Forest: Gerçek vs Tahmin Edilen Fiyat")
plt.grid(True)
plt.tight_layout()
plt.show()
