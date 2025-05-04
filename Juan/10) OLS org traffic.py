# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:59:28 2025

@author: juanj
"""
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset
file_path = "competitorperf_cleaned.csv"
df = pd.read_csv(file_path)

# Drop non-numeric and irrelevant columns
df_numeric = df.drop(columns=["CompetitorID", "Day", "domain", "Total Traffic", "Organic traffic value", "Organic pages"], errors='ignore')

# Replace zeros with a small value to avoid log issues
df_numeric = df_numeric.replace(0, 1e-10)

# Apply log transformation to all variables
df_log = np.log(df_numeric)

# Define independent variables (X) and dependent variable (y)
X_log = df_log.drop(columns=["Organic traffic"], errors='ignore')
y_log = df_log["Organic traffic"]

# Add a constant for the intercept
X_log_const = sm.add_constant(X_log)

# Split the data into training and testing sets (80-20 split)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log_const, y_log, test_size=0.2, random_state=42)

# Fit the regression model using statsmodels
model_sm_log = sm.OLS(y_train_log, X_train_log).fit()

# Predictions
y_pred_log = model_sm_log.predict(X_test_log)

# Model evaluation
r2_log = r2_score(y_test_log, y_pred_log)
mae_log = mean_absolute_error(y_test_log, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))

# Extract coefficients, t-values, and p-values
stats_df_log = pd.DataFrame({
    "Feature": model_sm_log.params.index,
    "Coefficient": model_sm_log.params.values,
    "T-Value": model_sm_log.tvalues.values,
    "P-Value": model_sm_log.pvalues.values
})

# Display results
print("Model Summary:\n", model_sm_log.summary())
print("R² Score:", r2_log)
print("Mean Absolute Error (MAE):", mae_log)
print("Root Mean Squared Error (RMSE):", rmse_log)
print("\nRegression Results:")
print(stats_df_log)
