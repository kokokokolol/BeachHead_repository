# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:42:02 2025

@author: juanj
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the dataset
file_path = "events_and_news_data_updated.csv"
df = pd.read_csv(file_path)

# Compute correlation coefficients
correlation_organic = df["Total Events"].corr(df["Organic"])
correlation_paid = df["Total Events"].corr(df["Paid"])

# Generate scatter plots
plt.figure(figsize=(12, 5))

# Scatter plot for Total Events vs Organic
plt.subplot(1, 2, 1)
sns.scatterplot(x=df["Total Events"], y=df["Organic"])
plt.xlabel("Total Events")
plt.ylabel("Organic")
plt.title(f"Total Events vs Organic (Correlation: {correlation_organic:.2f})")

# Scatter plot for Total Events vs Paid
plt.subplot(1, 2, 2)
sns.scatterplot(x=df["Total Events"], y=df["Paid"])
plt.xlabel("Total Events")
plt.ylabel("Paid")
plt.title(f"Total Events vs Paid (Correlation: {correlation_paid:.2f})")

plt.tight_layout()
plt.show()

# Return correlation values
correlation_organic, correlation_paid



# Regression: Organic as dependent variable
X_org = df["Total Events"]  # Independent variable
y_org = df["Organic"]  # Dependent variable

X_org = sm.add_constant(X_org)  # Add intercept
model_org = sm.OLS(y_org, X_org).fit()  # Fit the model

# Regression: Paid as dependent variable
X_paid = df["Total Events"]  # Independent variable
y_paid = df["Paid"]  # Dependent variable

X_paid = sm.add_constant(X_paid)  # Add intercept
model_paid = sm.OLS(y_paid, X_paid).fit()  # Fit the model

# Display regression summaries
model_org.summary(), model_paid.summary()