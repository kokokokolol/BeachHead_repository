# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:47:43 2025

@author: juanj
"""
import pandas as pd
import os

# Load the CSV file
file_path = "flattened_api_results1.csv"
output_file_path = "traffic_grouped_data.csv"

# Check if file exists before attempting to read it
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    # Create new columns 'Organic' and 'Paid' based on 'metric_type'
    df['Organic'] = df['count'].where(df['metric_type'] == 'organic', 0)
    df['Paid'] = df['count'].where(df['metric_type'] == 'paid', 0)

    # Drop unnecessary columns
    df.drop(columns=['count', 'metric_type', 'se_type', 'location_code', 'language_code'], inplace=True)

    # Group by 'target', 'year', and 'month', summing the numerical columns
    df_grouped = df.groupby(['target', 'year', 'month'], as_index=False).sum()

    # Save the grouped DataFrame to a CSV file
    df_grouped.to_csv(output_file_path, index=False)
    print(f"Grouped data has been saved to {output_file_path}")
    
    # Display a preview of the grouped DataFrame
    print(df_grouped.head())
else:
    print(f"Error: The file '{file_path}' was not found. Please check the file path and try again.")
