# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.model_selection import RandomizedSearchCV

import shap

import warnings

# 1) Load dataset
diabetes_data = load_diabetes(as_frame=True, scaled=False)

# Features and target
X = diabetes_data.data.copy()
y = diabetes_data.target.copy()

print("Dataset loaded successfully")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

# 2) Print range of values for one column
column_name = X.columns[0]  # choose the first feature
print(f"Range of column '{column_name}':")
print(f"Min: {X[column_name].min()}, Max: {X[column_name].max()}")

# 3) Handling NaN values
print("\nChecking for NaN values...")
print(X.isna().sum())
print(y.isna().sum())
data = pd.concat([X, y.rename('target')], axis=1)
data_new = data.dropna()

X = data_new.drop(columns=['target'])
y = data_new['target']

print(f"Samples after NaN removal: {X.shape[0]}")

# 4) Histogram of target variable
plt.figure()
plt.hist(y, bins=30)
plt.title("Histogram of Target Variable")
plt.xlabel("Target value")
plt.ylabel("Frequency")
plt.show()

# 5) Min-Max Scaling
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

print("Data normalization completed (Min-Max scaling)")

# 6) Random seed
seed = 42
np.random.seed(seed)

# Suppress warning (random seed)
warnings.filterwarnings("ignore", category=FutureWarning)

# 7) K-Fold configuration
kf = KFold(n_splits=6, shuffle=True, random_state=seed)

print("K-Fold cross-validation setup completed")
print(f"Number of folds: {kf.get_n_splits()}")

# Evaluation metric functions
def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def max_err(y_true, y_pred):
    """Maximum Error"""
    return max_error(y_true, y_pred)

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (in %)
    Handles zero values safely
    """
    eps = 1e-8  # small constant to avoid division by zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

# 8) Create a list to store the data
csv_data = []
# Append header row
csv_data.append([
    "Model", "Set", "Fold",
    "Max_n", "RMSE_n", "MAE_n", "MAPE_n",
    "Max", "RMSE", "MAE", "MAPE"
])

print("CSV data structure initialized")

# 9) Regression Models & Hyperparameter Spaces
# Model definitions
models = {
    "RandomForest": RandomForestRegressor(random_state=seed),
    "GPR": GaussianProcessRegressor(),
    "KNN": KNeighborsRegressor(),
    "GradientBoosting": GradientBoostingRegressor(random_state=seed)
}

# Hyperparameter distributions for RandomizedSearchCV
param_distributions = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "subsample": [0.8, 1.0]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },
    "GPR": {
        "kernel": [
            C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            C(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e2)),
            C(0.5, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        ],
        "alpha": [1e-2, 0.1, 0.2, 0.5]
    }
}

print("Models and hyperparameter spaces initialized")

# Main Training Loop (K-Fold & Models)
fold_idx = 0

for train_index, test_index in kf.split(X_scaled):
    print(f"Processing Fold {fold_idx}")

    # Split data using indices (no data replication)
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

    for model_name, model in models.items():
        print(f"Training model: {model_name}")

        # Randomized Search for hyperparameter tuning
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=3,
            random_state=seed,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Predictions (normalized scale)
        y_train_pred_n = best_model.predict(X_train)
        y_test_pred_n = best_model.predict(X_test)

        # Denormalize predictions and true values
        y_train_true = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

        y_train_pred = y_scaler.inverse_transform(y_train_pred_n.reshape(-1, 1)).ravel()
        y_test_pred = y_scaler.inverse_transform(y_test_pred_n.reshape(-1, 1)).ravel()

        # Metrics (normalized)
        train_metrics_n = (
            max_err(y_train, y_train_pred_n),
            rmse(y_train, y_train_pred_n),
            mae(y_train, y_train_pred_n),
            mape(y_train, y_train_pred_n)
        )

        test_metrics_n = (
            max_err(y_test, y_test_pred_n),
            rmse(y_test, y_test_pred_n),
            mae(y_test, y_test_pred_n),
            mape(y_test, y_test_pred_n)
        )

        # Metrics (original scale)
        train_metrics = (
            max_err(y_train_true, y_train_pred),
            rmse(y_train_true, y_train_pred),
            mae(y_train_true, y_train_pred),
            mape(y_train_true, y_train_pred)
        )

        test_metrics = (
            max_err(y_test_true, y_test_pred),
            rmse(y_test_true, y_test_pred),
            mae(y_test_true, y_test_pred),
            mape(y_test_true, y_test_pred)
        )

        # Actual vs Predicted plot (test set)
        plt.figure()
        plt.scatter(y_test_true, y_test_pred, alpha=0.7, label=f"{model_name} (Fold {fold_idx})")
        # Ideal line y = x
        min_val = min(y_test_true.min(), y_test_pred.min())
        max_val = max(y_test_true.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Ideal')

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted")
        plt.legend(loc='best')

        plt.show()

        # Append results to CSV data
        csv_data.append([
            model_name, "Train", fold_idx,
            *train_metrics_n,
            *train_metrics
        ])

        csv_data.append([
            model_name, "Test", fold_idx,
            *test_metrics_n,
            *test_metrics
        ])

        # SHAP Explainability
        if model_name in ["RandomForest", "GradientBoosting"]:
            print(f"Generating SHAP plots for {model_name}, Fold {fold_idx}...")
            
            try:
                # TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(best_model)
                shap_explanation = explainer(X_test, check_additivity=False)
                
                # 1) SHAP Summary Plot
                plt.figure()
                plt.title(f"SHAP Summary - {model_name} (Fold {fold_idx})")
                shap.summary_plot(shap_explanation, X_test, feature_names=X.columns, show=False)
                plt.show()

                # 2) Waterfall plots
                for i in range(2):
                    plt.figure()

                    shap.plots.waterfall(
                        shap.Explanation(
                            values=shap_explanation.values[i],
                            base_values=shap_explanation.base_values[i],
                            data=X_test[i],
                            feature_names=X.columns
                        ),
                        show=False
                    )

                    plt.title(f"Waterfall - {model_name} (Fold {fold_idx}, Class {i})")
                    plt.show()
                    
            except Exception as e:
                print(f"SHAP failed for {model_name} on fold {fold_idx}: {e}")

    # Increment fold index
    fold_idx += 1

print("Training loop completed")

# Save results to CSV
results_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
filename = "regression_results.csv"
results_df.to_csv(filename, index=False)

print(f"\nResults saved successfully to: {filename}")
print(results_df.head())
