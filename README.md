***Diabetes Progression Prediction - Regression Pipeline***

# **Overview**
-This project implements a comprehensive Machine Learning pipeline in Python to predict disease progression one year after baseline using the Scikit-Learn Diabetes dataset.
It applies robust data preprocessing, hyperparameter tuning, and cross-validation to evaluate various regression algorithms. Additionally, it integrates SHAP (SHapley --Additive exPlanations) to provide deep model interpretability.

# **Dataset**
--Source: The built-in load_diabetes dataset from the sklearn.datasets module.  
--Features: 10 baseline variables including age, sex, body mass index, average blood pressure, and six blood serum measurements.  
--Target Variable: A quantitative measure of disease progression one year after baseline.  

# **Features & Data Processing**
-The data pipeline applies standard data science practices for continuous variables:
--Missing Value Handling: The script automatically scans for NaN values in both features and the target variable, dropping any incomplete rows.  
--Normalization: Applies MinMaxScaler to scale both the feature set (X) and the target variable (y) to a standard range for optimal model training.  
--Cross-Validation: Utilizes K-Fold Cross-Validation configured with 6 splits to ensure robust, unbiased model evaluation.

# **Models Evaluated & Tuned**
-The script trains, tunes, and compares the following regression algorithms:
--Random Forest Regressor  
--Gaussian Process Regressor (GPR) (Utilizing RBF and Constant kernels)  
--K-Nearest Neighbors Regressor (KNN)  
--Gradient Boosting Regressor
-Note: Hyperparameter optimization is performed dynamically during training using RandomizedSearchCV (with 3-fold inner CV) to find the best estimator for each model type.

# **Evaluation Metrics**
-Model performance is evaluated on both the Train and Test sets using the following metrics, calculated for both the normalized data and the denormalized (original) scale:
--RMSE: Root Mean Squared Error  
--MAE: Mean Absolute Error  
--Max Error
--MAPE: Mean Absolute Percentage Error (custom implementation to safely handle zero values)

# **Model Interpretability (SHAP)**
-To avoid "black-box" predictions, the pipeline integrates the shap library for the tree-based algorithms (Random Forest and Gradient Boosting). 
During execution, it automatically generates:
--SHAP Summary Plots: To visualize global feature importance and the directional impact of features across the test set.  
--SHAP Waterfall Plots: To explain the specific contributions of each feature for individual sample predictions.

# **Visualizations**
-Throughout the execution, the script generates several plots:
--A histogram showing the distribution of the target variable.  
--Scatter plots mapping Actual vs. Predicted values for the test set, including an ideal trendline (y=x).

# **Outputs**
-After all folds are processed, the evaluation metrics for all models are aggregated and exported to a CSV file: regression_results.csv: 
Contains detailed performance metrics (Max Error, RMSE, MAE, MAPE) separated by Model, Fold, and Dataset type (Train/Test).

# **Requirements**
-To run this project, you need Python installed along with the following libraries:
--numpy  
--pandas  
--matplotlib  
--scikit-learn  
--shap  

# **How to Run**
1) Execute the Python script directly. No external data files are required as the dataset is loaded directly from sklearn.
2) The program will sequentially display the data visualizations, Actual vs. Predicted plots, and SHAP diagrams. Close each plot window to allow the execution to proceed to the next training phase.
3) Upon completion, check the root directory for the generated regression_results.csv report.
