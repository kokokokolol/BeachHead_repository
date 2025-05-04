# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:01:00 2025

@author: juanj
"""

import pandas as pd

# Load the original grouped dataset
file_path = "traffic_grouped_data.csv"
df = pd.read_csv(file_path)

# Group by year and month, summing up numerical columns
grouped_df = df.groupby(['year', 'month']).sum(numeric_only=True).reset_index()

# Load the newly uploaded dataset
new_file_path = "grouped_data_with_title_count3_dateseparated.csv"
df_new = pd.read_csv(new_file_path)

# Rename columns in the new dataset to match the original dataset
df_new.rename(columns={'Year': 'year', 'Month': 'month'}, inplace=True)

# Merge both dataframes on 'year' and 'month' columns
merged_df = pd.merge(grouped_df, df_new, on=['year', 'month'], how='inner')

# Drop 'items_count' and 'total_count' columns
merged_df.drop(columns=['items_count', 'total_count'], inplace=True)

# Rename 'Unique Title Count' to 'Total Events'
merged_df.rename(columns={'Unique Title Count': 'Total Events'}, inplace=True)

# Define the updated export file path
updated_export_path = "events_and_news_data_updated.csv"

# Save the updated dataframe to CSV
merged_df.to_csv(updated_export_path, index=False)

# Provide the download link
updated_export_path


