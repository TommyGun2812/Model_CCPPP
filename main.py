import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### MODEL'S FUNCTIONS ###
# Hypothesis function
def hyp_function(params: list, samples: list, bias):
    # Converting function to numpy arrays if necesary
    params  = np.asarray(params)
    samples = np.asarray(samples)
    # Return point product between params and samples
    return params @ samples + bias

# This is the MSE function, since it is used for a linear regression model
def cost_function(params:list, samples:list, bias, Y:list):
    # Converting function params to numpy arrays if necesary
    params  = np.asarray(params)
    samples = np.asarray(samples)
    Y       = np.asarray(Y)
    # Cost variable initialized
    cost = 0.0
    # Obtaining mean square error
    for i in range(len(samples)): 
        cost += (hyp_function(params, samples[i], bias) - Y[i]) ** 2
    cost /= len(samples)
    return cost

# Gradient descend function
def update_function(params: list, samples: list, Y:list, bias, alfa): 
    # Converting function params to numpy arrays if necesary
    params  = np.asarray(params)
    samples = np.asarray(samples)
    Y       = np.asarray(Y)
    # Local variables
    params_new = params.copy()
    bias_new   = 0.0
    grad       = 0.0
    # Data matrix dimension
    m, n = samples.shape
    # Theta gradient
    for i in range(n): 
        grad = 0.0
        for j in range(m): 
            grad += (hyp_function(params, samples[j], bias) - Y[j]) * samples[j][i]
        
        params_new[i] = params[i] - alfa * (1/len(samples)) * grad
        
    grad = 0.0
    # Bias gradient
    for i in range(m):
        grad += (hyp_function(params, samples[i], bias) - Y[i])

    bias_new = bias - alfa * (1/len(samples)) * grad

    return params_new, bias_new


### PLOTTING INFORMATION ###

def training_plot(error: list):
    plt.figure(figsize=(8,5))
    plt.plot(error, color='blue', linewidth=2)
    plt.title("Convergencia del MSE durante el entrenamiento")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()

def plot_test_results(theta, bias, test_set, test_labels):
    # Convertir a numpy arrays
    samples_test = test_set.to_numpy()
    y_test = test_labels.to_numpy()
    
    # Obtener predicciones
    predictions = np.array([hyp_function(theta, x, bias) for x in samples_test])
    
    # Crear scatter plot
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, predictions, alpha=0.5, color="purple")
    plt.plot([-3,3], [-3,3], color="black", linestyle="--", linewidth=2)  # Línea ideal y=x
    plt.title("Predicciones vs Valores Reales")
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.grid(True)
    plt.show()


### COEFFICIENT OF DETERMINATION ###
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)       # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)


if __name__ == '__main__':

    #-------DATA LOADING-------#
    columns = ["AT", "V", "AP", "RH", "PE"]
    df = pd.read_csv("./CCPP/Folds5x2_pp_copy.csv", names=columns)

    #-------DATA TRANSFORMATION-------#
    # Splitting Features and Targets
    features = df[["AT", "V", "AP", "RH"]]
    target   = df["PE"]

    # Features Mean
    at_mean  = df["AT"].mean()
    v_mean   =  df["V"].mean()
    ap_mean  = df["AP"].mean()
    rh_mean  = df["RH"].mean()
    pe_mean  = df["PE"].mean() 

    # Features Standard Deviation
    at_std = df["AT"].std()
    v_std  =  df["V"].std()
    ap_std = df["AP"].std()
    rh_std = df["RH"].std()
    pe_std = df["PE"].std()

    # Target's Mean
    target_mean = target.mean()

    # Target's Standard Deviation
    target_std  = target.std()


    # Features Standarization
    features.loc[:, "AT"] = (features["AT"] - at_mean) / at_std
    features.loc[:,  "V"] = (features["V"] - v_mean) / v_std
    features.loc[:, "AP"] = (features["AP"] - ap_mean) / ap_std
    features.loc[:, "RH"] = (features["RH"] - rh_mean) / rh_std

    # Target Standarization
    target = (target - target_mean) / target_std

    #-------DATA SPLITTING-------#
    train_set = features.iloc[0:2872, :] # Data's 70%
    test_set  = features.iloc[2872:, : ] # Data's 30%

    #-------DATA LABELS-------#
    train_labels = target.iloc[0:2872]
    test_labels  = target.iloc[2872: ] 

    #-------HYPERPARAMETERS-------#
    epoch   = 1000
    bias    = 0.0
    alfa    = 0.01
    samples = train_set.to_numpy()
    y       = train_labels.to_numpy()
    theta   = np.zeros(features.shape[1])
    error   = [] # Error list

    #-------TRAINING LOOP-------#
    i = 0
    while (i < epoch): 
        error.append(cost_function(theta, samples, bias, y,))

        if error[-1] == 0: 
            break

        # Previous parameters
        theta_prev = theta.copy()
        bias_prev  = bias

        # Updating parameters
        theta, bias = update_function(theta, samples, y, bias, alfa)
        
        print(f"Epoch {i+1}")
        print(f"Previous theta: {theta_prev}, Previous bias: {bias_prev}")
        print(f"Updated  theta: {theta}, Updated  bias: {bias}\n")
        i += 1
    

    #-------TESTING MDOEL-------#
    # Obtaining predictions with learned parameters
    samples_test = test_set.to_numpy()
    y_test       = test_labels.to_numpy()

    predictions = np.array([hyp_function(theta, x, bias) for x in samples_test])

    # Calcular R²
    accuracy = r2_score(y_test, predictions)
    print(f"R² del modelo: {accuracy:.4f}")

    # PLotting error evolution
    training_plot(error)
    plot_test_results(theta, bias, test_set, test_labels)
