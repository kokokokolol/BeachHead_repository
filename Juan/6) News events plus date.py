# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:40:50 2025

@author: juanj
"""

import pandas as pd


# Load the CSV file
df = pd.read_csv('grouped_data_with_title_count3.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract 'Year' and 'Month' into new columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Group by 'Year' and 'Month' and aggregate the data
grouped_df = df.groupby(['Year', 'Month']).agg({
    'Article Count': 'sum',  # Sum of articles per month and year
    'Weight': 'sum',         # Sum of weights per month and year
    'Sentiment': 'mean',     # Average sentiment per month and year
    'Relevance': 'mean',     # Average relevance per month and year
    'Unique Title Count': 'sum'  # Sum of unique titles per month and year
}).reset_index()

# Display the grouped DataFrame
print(grouped_df)

# Save the modified DataFrame to a new CSV file (optional)
grouped_df.to_csv('grouped_data_with_title_count3_dateseparated.csv', index=False)




