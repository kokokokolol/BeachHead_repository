# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:36:35 2025

@author: juanj
"""


import pandas as pd

# Load the CSV file
file_path = "traffic_grouped_data.csv"
df = pd.read_csv(file_path)

# Group by year and month, summing up numerical columns
grouped_df = df.groupby(['year', 'month']).sum(numeric_only=True).reset_index()

# Define the export file path
export_path = "grouped_traffic_no_target.csv"

# Save the grouped dataframe to CSV
grouped_df.to_csv(export_path, index=False)

# Provide the download link
export_path

