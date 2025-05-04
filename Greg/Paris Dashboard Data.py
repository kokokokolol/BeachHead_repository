#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 18:10:01 2025

@author: midoriwest
"""

import pandas as pd 

# Load the CSV file
file_path_org = "rank_easy_organic_keyword_targets.csv"
df_org = pd.read_csv(file_path_org)

# Define the keywords to filter
keywords_org = ["replay attacks", "hids", "infrastructure in it"]

# Filter the DataFrame
filtered_df_org = df_org[df_org['Max_Keyword'].str.contains('|'.join(keywords_org), case=False, na=False)]

# Calculate total and 5%
total_traffic_org = filtered_df_org['Sum_Organic_Traffic'].sum()
result_org = total_traffic_org * 0.05



# Load the CSV file
budget = 500
file_path = "rank_easy_cpc_keyword_targets.csv"  # Make sure this file is in your working directory
df = pd.read_csv(file_path)

# Drop rows with missing CPC or Organic Traffic values
df = df.dropna(subset=["Max_CPC", "Sum_Organic_Traffic"])

# Calculate the weighted average CPC
weighted_avg_cpc = (df["Max_CPC"] * df["Max_Volume"]).sum() / df["Max_Volume"].sum()
total_search_volume = (df["Max_Volume"]).sum()
total_search_volume_budget = budget/weighted_avg_cpc


# Load the CSV file
ind_file_path = "ranked_keywords_with_traffic.csv"  # Make sure the file is in your working directory
ind_df = pd.read_csv(ind_file_path)

# Drop rows with missing CPC or search volume values
ind_df_filtered = ind_df.dropna(subset=["cpc", "search_volume"])

# Calculate the weighted average CPC using search volume as the weight
ind_weighted_avg_cpc = (ind_df_filtered["cpc"] * ind_df_filtered["search_volume"]).sum() / ind_df_filtered["search_volume"].sum()




# Load the uploaded file
file_path = "ranked_keywords_with_traffic.csv"
df = pd.read_csv(file_path)

# Identify all year-month columns (format: YYYY-MM)
year_month_columns = [col for col in df.columns if col[:4].isdigit() and col[4] == '-']

# Group by keyword and compute the max for each month
max_by_keyword = df.groupby('keyword')[year_month_columns].max().reset_index()

# Save the summarized data to a new CSV file
output_path = "max_monthly_traffic_by_keyword.csv"
max_by_keyword.to_csv(output_path, index=False)

# Optional: print a quick preview
print(max_by_keyword.head())



# Load the dataset
df = pd.read_csv("max_monthly_traffic_by_keyword.csv")

# Identify year-month columns
year_month_columns = [col for col in df.columns if col[:4].isdigit() and col[4] == '-']

# Calculate industry-wide monthly totals
monthly_totals = df[year_month_columns].sum()
industry_mean = monthly_totals.mean()
industry_std = monthly_totals.std()

percent_std = industry_std/industry_mean

# Define your 3 keywords (replace with actual keyword values)
keywords = keywords_org

# Filter for those keywords and calculate their combined monthly total
selected_keywords_df = df[df['keyword'].isin(keywords)]
combined_keyword_traffic = selected_keywords_df[year_month_columns].sum(axis=1).sum()/12

# Estimate expected range for these 3 keywords based on industry pattern
# Assuming each keyword behaves like an average keyword, scale the industry range
estimated_lower = (combined_keyword_traffic - (combined_keyword_traffic * percent_std)) * 0.05
estimated_upper = (combined_keyword_traffic + (combined_keyword_traffic * percent_std)) * 0.05



# Print the result
print("Industry Weighted Average CPC:", ind_weighted_avg_cpc)
print("Weighted Average CPC:", weighted_avg_cpc)
print("Total Paid Search Volume", total_search_volume)
print("Total Paid Volume Budget", total_search_volume_budget)
print("Total Organic Traffic:", total_traffic_org)
print("5% of Total Traffic:", result_org)

# Print results
print(percent_std)
print(f"Actual combined traffic for keywords: {combined_keyword_traffic}")
print(f"Estimated traffic range (based on industry typical range for 3 keywords): {estimated_lower:.0f} to {estimated_upper:.0f}")import pandas as pd 

# Load the CSV file
file_path_org = "rank_easy_organic_keyword_targets.csv"
df_org = pd.read_csv(file_path_org)

# Define the keywords to filter
keywords_org = ["replay attacks", "hids", "infrastructure in it"]

# Filter the DataFrame
filtered_df_org = df_org[df_org['Max_Keyword'].str.contains('|'.join(keywords_org), case=False, na=False)]

# Calculate total and 5%
total_traffic_org = filtered_df_org['Sum_Organic_Traffic'].sum()
result_org = total_traffic_org * 0.05



# Load the CSV file
budget = 500
file_path = "rank_easy_cpc_keyword_targets.csv"  # Make sure this file is in your working directory
df = pd.read_csv(file_path)

# Drop rows with missing CPC or Organic Traffic values
df = df.dropna(subset=["Max_CPC", "Sum_Organic_Traffic"])

# Calculate the weighted average CPC
weighted_avg_cpc = (df["Max_CPC"] * df["Max_Volume"]).sum() / df["Max_Volume"].sum()
total_search_volume = (df["Max_Volume"]).sum()
total_search_volume_budget = budget/weighted_avg_cpc


# Load the CSV file
ind_file_path = "ranked_keywords_with_traffic.csv"  # Make sure the file is in your working directory
ind_df = pd.read_csv(ind_file_path)

# Drop rows with missing CPC or search volume values
ind_df_filtered = ind_df.dropna(subset=["cpc", "search_volume"])

# Calculate the weighted average CPC using search volume as the weight
ind_weighted_avg_cpc = (ind_df_filtered["cpc"] * ind_df_filtered["search_volume"]).sum() / ind_df_filtered["search_volume"].sum()




# Load the uploaded file
file_path = "ranked_keywords_with_traffic.csv"
df = pd.read_csv(file_path)

# Identify all year-month columns (format: YYYY-MM)
year_month_columns = [col for col in df.columns if col[:4].isdigit() and col[4] == '-']

# Group by keyword and compute the max for each month
max_by_keyword = df.groupby('keyword')[year_month_columns].max().reset_index()

# Save the summarized data to a new CSV file
output_path = "max_monthly_traffic_by_keyword.csv"
max_by_keyword.to_csv(output_path, index=False)

# Optional: print a quick preview
print(max_by_keyword.head())



# Load the dataset
df = pd.read_csv("max_monthly_traffic_by_keyword.csv")

# Identify year-month columns
year_month_columns = [col for col in df.columns if col[:4].isdigit() and col[4] == '-']

# Calculate industry-wide monthly totals
monthly_totals = df[year_month_columns].sum()
industry_mean = monthly_totals.mean()
industry_std = monthly_totals.std()

percent_std = industry_std/industry_mean

# Define your 3 keywords (replace with actual keyword values)
keywords = keywords_org

# Filter for those keywords and calculate their combined monthly total
selected_keywords_df = df[df['keyword'].isin(keywords)]
combined_keyword_traffic = selected_keywords_df[year_month_columns].sum(axis=1).sum()/12

# Estimate expected range for these 3 keywords based on industry pattern
# Assuming each keyword behaves like an average keyword, scale the industry range
estimated_lower = (combined_keyword_traffic - (combined_keyword_traffic * percent_std)) * 0.05
estimated_upper = (combined_keyword_traffic + (combined_keyword_traffic * percent_std)) * 0.05



# Print the result
print("Industry Weighted Average CPC:", ind_weighted_avg_cpc)
print("Weighted Average CPC:", weighted_avg_cpc)
print("Total Paid Search Volume", total_search_volume)
print("Total Paid Volume Budget", total_search_volume_budget)
print("Total Organic Traffic:", total_traffic_org)
print("5% of Total Traffic:", result_org)

# Print results
print(percent_std)
print(f"Actual combined traffic for keywords: {combined_keyword_traffic}")
print(f"Estimated traffic range (based on industry typical range for 3 keywords): {estimated_lower:.0f} to {estimated_upper:.0f}")