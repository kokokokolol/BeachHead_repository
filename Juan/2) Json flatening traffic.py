# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:30:27 2025

@author: juanj
"""


import pandas as pd
import json
import ast  

# Step 1: Load the CSV file
file_path = r"C:\Users\juanj\OneDrive\Escritorio\python_Client\api_results5.csv"
df = pd.read_csv(file_path)

# Step 2: Parse the 'items' column using ast.literal_eval
df['items'] = df['items'].apply(ast.literal_eval)  # Fix here

# Step 3: Flatten the nested JSON structure
flattened_data = []
for _, row in df.iterrows():
    se_type = row['se_type']
    location_code = row['location_code']
    language_code = row['language_code']
    total_count = row['total_count']
    items_count = row['items_count']
    items = row['items']
    
    for item in items:
        target = item['target']
        metrics = item['metrics']
        
        # Process organic metrics
        for organic_metric in metrics['organic']:
            flattened_data.append({
                'se_type': se_type,
                'location_code': location_code,
                'language_code': language_code,
                'total_count': total_count,
                'items_count': items_count,
                'target': target,
                'metric_type': 'organic',
                'year': organic_metric['year'],
                'month': organic_metric['month'],
                'etv': organic_metric['etv'],
                'count': organic_metric['count']
            })
        
        # Process paid metrics
        for paid_metric in metrics['paid']:
            flattened_data.append({
                'se_type': se_type,
                'location_code': location_code,
                'language_code': language_code,
                'total_count': total_count,
                'items_count': items_count,
                'target': target,
                'metric_type': 'paid',
                'year': paid_metric['year'],
                'month': paid_metric['month'],
                'etv': paid_metric['etv'],
                'count': paid_metric['count']
            })

# Step 4: Convert to DataFrame
flattened_df = pd.DataFrame(flattened_data)


# Step 5: Export to CSV
output_file_path = "flattened_api_results1.csv"  # Specify the output file path
flattened_df.to_csv(output_file_path, index=False)

print(f"Data successfully exported to {output_file_path}")