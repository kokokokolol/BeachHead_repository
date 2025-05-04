# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:33:33 2025

@author: juanj
"""

import pandas as pd

# Load the CSV file
file_path = "competitorperf.csv"
df = pd.read_csv(file_path)

# Create a new column 'Total Traffic' as the sum of 'Organic traffic' and 'Paid traffic'
df["Total Traffic"] = df["Organic traffic"] + df["Paid traffic"]

# Define the output file path
output_file_path = "competitorperf_updated.csv"

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file_path, index=False)

# Provide the download link to the user
print(f"Updated file saved at: {output_file_path}")

# Load the newly uploaded dataset
file_path_new = "competitorperf_updated.csv"
df_new = pd.read_csv(file_path_new)

# Replace all missing values with 0
df_cleaned = df_new.fillna(0)

# Save the cleaned file
cleaned_file_path = "competitorperf_cleaned.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

# Provide the cleaned file path
print(cleaned_file_path)