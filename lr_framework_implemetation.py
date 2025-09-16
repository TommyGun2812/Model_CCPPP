import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

#-------DATA LOADING-------#
columns = ["AT", "V", "AP", "RH", "PE"]
df = pd.read_csv("./Folds5x2_pp.csv", names=columns)

# Features y target
X = df[["AT", "V", "AP", "RH"]].values
y = df["PE"].values

#-------DATA SPLITTING-------#
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#-------NORMALIZATION-------#
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val   = scaler_X.transform(X_val)
X_test  = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
y_val   = scaler_y.transform(y_val.reshape(-1,1)).ravel()
y_test  = scaler_y.transform(y_test.reshape(-1,1)).ravel()

#-------MODEL WITH SGD (gradient descent)-------#
sgd = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.01,
    max_iter=1,
    warm_start=True,
    random_state=42
)

n_epochs = 1000
train_errors, val_errors = [], []

for epoch in range(n_epochs):
    sgd.fit(X_train, y_train)  # actualiza parámetros
    
    # errores en train y val
    y_train_pred = sgd.predict(X_train)
    y_val_pred   = sgd.predict(X_val)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))

#-------METRICAS FINALES (normalizadas)-------#
y_train_pred_final = sgd.predict(X_train)
y_val_pred_final   = sgd.predict(X_val)
y_test_pred_final  = sgd.predict(X_test)

final_train_mse = mean_squared_error(y_train, y_train_pred_final)
final_val_mse   = mean_squared_error(y_val, y_val_pred_final)
final_test_mse  = mean_squared_error(y_test, y_test_pred_final)

final_train_r2  = r2_score(y_train, y_train_pred_final)
final_val_r2    = r2_score(y_val, y_val_pred_final)
final_test_r2   = r2_score(y_test, y_test_pred_final)

print("\n---- MÉTRICAS FINALES (ESCALA NORMALIZADA) ----")
print(f"Train MSE: {final_train_mse:.4f} | Train R²: {final_train_r2:.4f}")
print(f"Val   MSE: {final_val_mse:.4f} | Val   R²: {final_val_r2:.4f}")
print(f"Test  MSE: {final_test_mse:.4f} | Test  R²: {final_test_r2:.4f}")

#-------METRICAS FINALES (escala real)-------#
# Invertir el escalado para y
y_train_real      = scaler_y.inverse_transform(y_train.reshape(-1,1)).ravel()
y_val_real        = scaler_y.inverse_transform(y_val.reshape(-1,1)).ravel()
y_test_real       = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()

y_train_pred_real = scaler_y.inverse_transform(y_train_pred_final.reshape(-1,1)).ravel()
y_val_pred_real   = scaler_y.inverse_transform(y_val_pred_final.reshape(-1,1)).ravel()
y_test_pred_real  = scaler_y.inverse_transform(y_test_pred_final.reshape(-1,1)).ravel()

final_train_mse_real = mean_squared_error(y_train_real, y_train_pred_real)
final_val_mse_real   = mean_squared_error(y_val_real, y_val_pred_real)
final_test_mse_real  = mean_squared_error(y_test_real, y_test_pred_real)

final_train_r2_real  = r2_score(y_train_real, y_train_pred_real)
final_val_r2_real    = r2_score(y_val_real, y_val_pred_real)
final_test_r2_real   = r2_score(y_test_real, y_test_pred_real)


#-------PLOTTING-------#
plt.figure(figsize=(8,5))
plt.plot(train_errors, label="Training Loss", color="blue")
plt.plot(val_errors, label="Validation Loss", color="orange")
plt.title("Pérdidas vs Épocas")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(7,7))
plt.scatter(y_test, y_test_pred_final, alpha=0.5, color="purple")
plt.plot([-3,3], [-3,3], "k--", linewidth=2)
plt.title("Predicciones vs Valores reales (test)")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(y_test_real, label="Valores reales", marker="o", linestyle="", alpha=0.7, color="blue")
plt.plot(y_test_pred_real, label="Predicciones", marker="x", linestyle="", alpha=0.7, color="red")
plt.title("Comparación valores reales vs predichos (Test set)")
plt.xlabel("Instancias")
plt.ylabel("PE (MW)")
plt.legend()
plt.grid(True)
plt.show()
