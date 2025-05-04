# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:24:40 2024

@author: greg
"""

import pandas as pd

# Load your data into a DataFrame
# Example: df = pd.read_csv('raw_organic_keywords.csv')
df = pd.read_csv('ranked_keywords_with_traffic.csv')  # Replace this with your actual data loading step

# Calculate the average CPC
avg_cpc = df['cpc'].mean()

# Group by 'Keyword' and perform aggregations
grouped_df = df.groupby('keyword').agg(
    Max_Keyword=('keyword', 'max'),
    Max_Volume=('search_volume', 'max'),
    Max_KD=('competition', 'max'),
    Max_CPC=('cpc', 'max'),
    #Sum_Organic_Traffic=('organic_etv', 'sum'),
    Sum_Organic_Traffic=('2024-10', 'sum'), # Using '2024-10' Because other data is at domain level
)

# Calculate additional columns
grouped_df['Industry_Relevant'] = round(
    (grouped_df['Sum_Organic_Traffic'] / grouped_df['Max_Volume']) * 100, 0
)
grouped_df['Easy_CPC_Traffic'] = round(
    grouped_df['Sum_Organic_Traffic'] / grouped_df['Max_CPC'], 0
)

# Filter rows based on conditions
filtered_df = grouped_df[
    (grouped_df['Sum_Organic_Traffic'] > 0) &
    (grouped_df['Max_CPC'] <= avg_cpc)
]

# Sort by Easy_CPC_Traffic in descending order
result_df = filtered_df.sort_values(
    by='Easy_CPC_Traffic', ascending=False
)

# Display or save the resulting DataFrame
print(result_df.head(100))

# If needed, save the result to a CSV
result_df.head(100).to_csv('rank_easy_cpc_keyword_targets.csv', index=False)
