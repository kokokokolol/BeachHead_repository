# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:22:36 2024

@author: greg
"""

import pandas as pd

# Load your data into a DataFrame
# Example: df = pd.read_csv('ranked_keywords_with_traffic.csv')
df = pd.read_csv('ranked_keywords_with_traffic.csv')  # Replace this with your actual data loading step

# Calculate the average KD and 1.5x the average
avg_kd = df['competition'].mean()
max_kd_threshold = avg_kd * 1.5

# Group by 'Keyword' and perform aggregations
grouped_df = df.groupby('keyword').agg(
    Max_Keyword=('keyword', 'max'),
    Max_Volume=('search_volume', 'max'),
    Max_KD=('competition', 'max'),
    Max_CPC=('cpc', 'max'),
    Sum_Organic_Traffic=('2024-10', 'sum'),
)

# Calculate additional columns
grouped_df['Industry_Relevant'] = round(
    (grouped_df['Sum_Organic_Traffic'] / grouped_df['Max_Volume']) * 100, 0
)
grouped_df['Easy_Organic_Traffic'] = round(
    grouped_df['Sum_Organic_Traffic'] / grouped_df['Max_KD'], 0
)

# Filter rows based on conditions
filtered_df = grouped_df[
    (grouped_df['Sum_Organic_Traffic'] > 0) &
    (grouped_df['Max_KD'] <= max_kd_threshold)
]

# Sort by Easy_Organic_Traffic in descending order
result_df = filtered_df.sort_values(
    by='Easy_Organic_Traffic', ascending=False
)

# Display or save the resulting DataFrame
print(result_df)

# If needed, save the result to a CSV
result_df.head(20).to_csv('rank_easy_organic_keyword_targets.csv', index=False)
