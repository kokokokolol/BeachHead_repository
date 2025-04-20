import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install_package("pandas")
install_package("openai")

install_package("eventregistry")

print("All required packages have been installed successfully.")

# ==================== Nick - START OF BeachHead Master Code v1.py ====================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 02:35:09 2024

@author: midoriwest
"""
import json
import csv
from client import RestClient

# Initialize the RestClient with your credentials
client = RestClient("email used for DataSEO here", "api key/code here") #Your credentials should be found here after making an account: https://app.dataforseo.com/api-access

# ^ You can download the file needed for this client here https://cdn.dataforseo.com/v3/examples/python/python_Client.zip

# List of URLs to process
urls = [
    "covertswarm.com",
    "redscan.com",
    "tekkis.com",
    "ek.co",
    "approach-cyber.com",
    "coresecurity.com",
    "packetlabs.net",
    "purplesec.us",
    "zelvin.com",
    "breachlock.com",
    "hackerone.com",
    "offsec.com",
    "whiteknightlabs.com",
    "synack.com",
    "bishopfox.com",
    "mitnicksecurity.com",
    "tcm-sec.com",
    "coalfire.com",
    "dionach.com",
    "raxis.com"
]

# Configure task parameters
default_params = {
    "mode": "as_is",
    "limit": 1000,  # Limit set to 1000
    "filters": [["dofollow", "=", True]],
    "backlinks_status_type": "live",
    "include_subdomains": True,
    "include_indirect_links": True,
    "exclude_internal_backlinks": True
}

# Prepare final JSON structure
final_results = {
    "version": "0.1.20231206",
    "status_code": 20000,
    "status_message": "Ok.",
    "time": "0.0 sec.",
    "cost": 0.0,
    "tasks_count": 0,
    "tasks_error": 0,
    "tasks": []
}

# Prepare CSV data
csv_data = []

# Loop through each URL and make a request
for url in urls:
    post_data = {
        0: {
            "target": url,
            **default_params,
        }
    }
    response = client.post("/v3/backlinks/backlinks/live", post_data)

# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# Finalize metadata
final_results["time"] = f"{len(urls) * 0.2:.4f} sec."

# Export to JSON file
output_json_file = "backlinks_results.json"
with open(output_json_file, "w", encoding="utf-8") as json_file:
    json.dump(final_results, json_file, indent=4)

print(f"\nResults have been exported to {output_json_file}!")

# Export to CSV file
output_csv_file = "backlinks_results.csv"
with open(output_csv_file, "w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "Target", "Type", "Domain From", "Domain To", "URL From", "URL To",
        "TLD From", "Backlink Spam Score", "Rank", "Page From Rank", "Domain From Rank",
        "Domain From Platform Type", "Page From External Links", "Page From Internal Links",
        "Page From Language", "First Seen", "Prev Seen", "Last Seen", "Item Type",
        "Attributes", "Links Count", "Group Count", "URL To Spam Score",
        "Ranked Keywords Info", "Page From Keywords Count Top 3",
        "Page From Keywords Count Top 10", "Page From Keywords Count Top 100"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Results have also been exported to {output_csv_file}!")

# ==================== END OF BeachHead Master Code v1.py ====================



# ==================== Unified Organic and Paid Traffic Prediction ====================

# Load the dataset
traffic_data = pd.read_csv("ranked_keywords_with_traffic.csv")

# -------------------- Organic Traffic Prediction --------------------
organic_feature_cols = [
    'search_volume', 'competition', 'cpc', 'organic_etv', 'paid_etv',
    '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04',
    '2024-05', '2024-06', '2024-07', '2024-08', '2024-09'
]
organic_target_col = '2024-10'

df_organic = traffic_data.dropna(subset=organic_feature_cols + [organic_target_col])
X_org = df_organic[organic_feature_cols]
y_org = df_organic[organic_target_col]

X_org_train, X_org_test, y_org_train, y_org_test = train_test_split(X_org, y_org, test_size=0.2, random_state=42)

rf_org_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_org_model.fit(X_org_train, y_org_train)

y_org_pred = rf_org_model.predict(X_org_test)
r2_org = r2_score(y_org_test, y_org_pred)
mae_org = mean_absolute_error(y_org_test, y_org_pred)
rmse_org = np.sqrt(mean_squared_error(y_org_test, y_org_pred))

print("\n📈 ORGANIC TRAFFIC PREDICTION")
print("R² Score:", round(r2_org, 4))
print("MAE:", round(mae_org, 2))
print("RMSE:", round(rmse_org, 2))

# -------------------- Paid Traffic Prediction --------------------
paid_train_months = ['2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']
paid_test_months = ['2024-07', '2024-08', '2024-09', '2024-10']

df_paid = traffic_data.dropna(subset=paid_train_months + paid_test_months + ['cpc'])
df_paid['train_traffic'] = df_paid[paid_train_months].sum(axis=1)
df_paid['test_traffic'] = df_paid[paid_test_months].sum(axis=1)
df_paid['Easy_CPC_Traffic'] = df_paid['train_traffic'] / df_paid['cpc']

X_paid = df_paid[['Easy_CPC_Traffic']]
y_paid = df_paid['test_traffic']

rf_paid_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_paid_model.fit(X_paid, y_paid)

y_paid_pred = rf_paid_model.predict(X_paid)
r2_paid = r2_score(y_paid, y_paid_pred)
mae_paid = mean_absolute_error(y_paid, y_paid_pred)
rmse_paid = np.sqrt(mean_squared_error(y_paid, y_paid_pred))

print("\n💰 PAID TRAFFIC PREDICTION")
print("R² Score:", round(r2_paid, 4))
print("MAE:", round(mae_paid, 2))
print("RMSE:", round(rmse_paid, 2))

# ==================== END OF Unified Organic and Paid Traffic Prediction ====================


# ==================== Jared - START OF categories and traffic.py ====================
# -*- coding: utf-8 -*-
"""MSBA Capstone 2 (JR).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BrQofV-hd327Mgab2Fmx2d6aIrch6YA5

This script reads a CSV file containing URLs, extracts unique URLs, and queries the Klazify API to categorize each URL based on its content. It then stores the category and confidence score in a dictionary, maps the results back to the original dataset, and saves the updated data to a new CSV file. The script also includes error handling and API rate limiting to ensure smooth execution.
"""

import pandas as pd
import requests
import time

# Your Klazify API key
API_KEY = "api key/code here"  # Replace with your actual API key

# Endpoint URL
API_URL = "https://www.klazify.com/api/categorize"

# Load your CSV file
df = pd.read_csv("backlinks_results.csv")

# Extract unique 'URL To' entries

# Check column names in the DataFrame
print("Available columns:", df.columns.tolist())

# Safely access the column if it exists
if "URL To" in df.columns:
    unique_urls = df["URL To"].unique()
elif "URL To" in df.columns:
    unique_urls = df["URL To"].unique()  # Adjusted for possible naming variation
else:
    raise KeyError("Neither 'URL To' nor 'URL To' columns found. Please check the CSV structure.")


# Set up headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Dictionary to store API results
category_results = {}

def categorize_url(url):
    """Fetch category information for a given URL."""
    payload = {"url": url}
    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        categories = data.get("domain", {}).get("categories", [])

        if categories:
            top_category = categories[0]  # Take the first (most relevant) category
            return top_category.get("name"), top_category.get("confidence")
        else:
            return None, None
    else:
        print(f"❌ Request failed for {url} with status code {response.status_code}")
        return None, None

# Loop through unique URLs and get category data
for i, url in enumerate(unique_urls):
    category_name, confidence = categorize_url(url)
    category_results[url] = (category_name, confidence)

    print(f"✅ Processed {i+1}/{len(unique_urls)}: {url} -> {category_name} (Confidence: {confidence})")

    # Respect API rate limits (adjust if necessary)
    time.sleep(1)

# Map results back to the original DataFrame
df["category_name"] = df["URL To"].map(lambda x: category_results.get(x, (None, None))[0])
df["category_confidence"] = df["URL To"].map(lambda x: category_results.get(x, (None, None))[1])

# Save the updated file
df.to_csv("backlinks_results_with_categories.csv", index=False)

print("🎉 Done! Updated CSV file is saved as 'backlinks_results_with_categories.csv'.")

"""This script uploads a CSV file containing URLs, extracts unique URLs from the URL_To column, and queries the DataForSEO API to retrieve organic and paid traffic estimates for each URL. It stores the results in a dictionary, maps them back to the dataset, and saves the updated file as backlinks_results_with_categories_and_traffic.csv. The script also includes error handling and rate limiting to ensure efficient API usage."""

import pandas as pd

from client import RestClient
import time

# Step 1: Upload CSV file
print("Please upload the 'backlinks_results_with_categories.csv' file.")


import os

# Prompt user for the CSV file path
file_path = input("Enter the path to the CSV file: ").strip()

# List files in the current directory to assist user
print("Available files in the current directory:")
for f in os.listdir("."):
    print(f"- {f}")

# Attempt to load the CSV file with error handling
try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded: {file_path}")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}. Please check the path and try again.")
    exit(1)

print(f"Loaded data from {file_path}")

# ✅ ENHANCED DEBUGGING: Check data after loading
print("\n📊 DataFrame Columns:", df.columns.tolist())
print("🔎 Null values per column:\n", df.isnull().sum())
print("📝 Sample data after loading:\n", df.head())



# Step 2: Read the uploaded CSV file
file_name = file_path  # Get uploaded file name
data = pd.read_csv(file_name)

# Step 3: Debugging: Print column names
print("Columns in the uploaded CSV file:")
print(data.columns)

# Step 4: Ensure the correct column name for 'URL To'
column_name = "URL To"
if column_name not in data.columns:
    raise KeyError(f"Column '{column_name}' not found in the CSV file. Available columns: {list(data.columns)}")

# Extract unique URLs from 'URL To' column
unique_urls = data[column_name].unique()

# Initialize RestClient with your credentials
client = RestClient("email used for DataSEO here", "api key/code here")

# Dictionary to store traffic results for unique URLs
traffic_results = {}

# Step 5: Loop through unique URLs and make API calls
for index, url in enumerate(unique_urls):
    print(f"Processing {index + 1}/{len(unique_urls)}: {url}")

    # Prepare the API request payload
    post_data = {
        "0": {
            "targets": [url],
            "date_from": "2022-12-05",
            "date_to": "2024-12-04",
            "item_types": ["organic", "paid"]
        }
    }

    # Make the API request
    response = client.post("/v3/dataforseo_labs/google/bulk_traffic_estimation/live", post_data)

# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# Step 6: Map traffic data back to the full dataset
data["Organic"] = data["URL To"].map(lambda x: traffic_results.get(x, (0, 0))[0])
data["Paid"] = data["URL To"].map(lambda x: traffic_results.get(x, (0, 0))[1])

# Step 7: Save the updated DataFrame
output_file = "backlinks_results_with_categories_and_traffic.csv"
data.to_csv(output_file, index=False)

print(f"🎉 Updated traffic data saved to {output_file}")

"""This script trains an optimized Random Forest model to predict log-transformed organic traffic based on URL features and category groupings. It starts by uploading and preprocessing the dataset, including categorizing industries, extracting SEO-related features (domain length, HTTPS status, presence of numbers), and hyperparameter tuning using GridSearchCV to find the best model. Finally, it trains the best model, evaluates performance (R², MAE), and prints feature importance, helping identify the key drivers of organic traffic."""

# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor  # Importing Random Forest
from sklearn.model_selection import train_test_split, GridSearchCV  # Splitting data & hyperparameter tuning
from sklearn.metrics import r2_score, mean_absolute_error  # Model evaluation metrics
  # For uploading files in Google Colab

# Step 1: Upload CSV file in Google Colab
print("Please upload 'backlinks_results_with_categories_and_traffic.csv'.")


import os

# Prompt user for the CSV file path
file_path = input("Enter the path to the CSV file: ").strip()

# List files in the current directory to assist user
print("Available files in the current directory:")
for f in os.listdir("."):
    print(f"- {f}")

# Attempt to load the CSV file with error handling
try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded: {file_path}")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}. Please check the path and try again.")
    exit(1)

print(f"Loaded data from {file_path}")


# Step 2: Read the uploaded CSV file into a Pandas DataFrame
file_name = file_path  # Get the uploaded file name
df = pd.read_csv(file_name)  # Read CSV into DataFrame

# Step 3: Ensure numeric data & drop missing values
df['Organic'] = pd.to_numeric(df['Organic'], errors='coerce')  # Convert 'Organic' column to numeric
df['Paid'] = pd.to_numeric(df['Paid'], errors='coerce')  # Convert 'Paid' column to numeric
df = df[['category_name', 'Organic', 'Paid', 'URL To']].dropna()  # Keep only relevant columns and drop NaN values

# Step 4: Apply log transformation to 'Organic' traffic
df['log_organic'] = np.log1p(df['Organic'])  # log1p(x) = log(x + 1) to handle zeros

# Step 5: Function to simplify 'category_name' into broader topics
def simplify_category(category):
    """Groups categories into broader topics to reduce the number of unique labels."""
    if "Computers & Electronics" in category:
        return "Technology"
    elif "Finance" in category:
        return "Finance"
    elif "Health" in category:
        return "Health"
    elif "Business & Industrial" in category:
        return "Business"
    elif "Internet & Telecom" in category:
        return "Internet"
    elif "Science" in category:
        return "Science"
    elif "Education" in category or "Jobs & Education" in category:
        return "Education"
    else:
        return "Other"

# Apply the function to create a new column
df['category_group'] = df['category_name'].apply(simplify_category)

# Step 6: Convert categorical variable 'category_group' into numerical format (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['category_group'], drop_first=True, dtype=float)

# Step 7: Extract additional features from the URL
df_encoded['domain_length'] = df['URL To'].apply(lambda x: len(str(x)))  # Measure length of domain
df_encoded['has_numbers'] = df['URL To'].apply(lambda x: int(bool(re.search(r'\d', str(x)))))  # Check if URL contains numbers
df_encoded['is_https'] = df['URL To'].apply(lambda x: 1 if str(x).startswith("https") else 0)  # Check if the URL uses HTTPS

# Step 8: Prepare training data
X = df_encoded.drop(columns=['Organic', 'log_organic', 'URL To', 'category_name'])  # Drop unnecessary columns
y = df_encoded['log_organic']  # Define target variable (log-transformed organic traffic)

# Step 9: Split data into training (80%) and testing (20%) sets

# ✅ DEBUGGING: Check shapes before train_test_split
print("🔍 Features (X) shape:", X.shape)
print("🎯 Target (y) shape:", y.shape)
print("📝 Sample of X:\n", X.head())
print("📝 Sample of y:\n", y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Hyperparameter tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [5, 10, 20, None],  # Depth of trees (None means unlimited)
    'min_samples_split': [2, 5, 10]  # Minimum samples required to split a node
}

rf = RandomForestRegressor(random_state=42)  # Initialize the Random Forest model
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)  # Perform cross-validation to find best parameters
grid_search.fit(X_train, y_train)  # Fit the model to training data

best_rf = grid_search.best_estimator_  # Get the best model from Grid Search

# Step 11: Train the optimized Random Forest model on full training data
best_rf.fit(X_train, y_train)

# Step 12: Make predictions on the test set
y_pred_rf = best_rf.predict(X_test)

# Step 13: Evaluate Model Performance
r2_rf = r2_score(y_test, y_pred_rf)  # R² Score (How well the model fits)
mae_rf = mean_absolute_error(y_test, y_pred_rf)  # Mean Absolute Error (Average prediction error)

# Print the evaluation results
print(f"✅ Optimized Random Forest R² Score: {r2_rf:.4f}")  # Higher is better (closer to 1)
print(f"✅ Optimized Random Forest MAE: {mae_rf:.4f}")  # Lower is better

# Step 14: Analyze Feature Importance
feature_importance_rf = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Print feature importance values
print("\n📊 Random Forest Feature Importance:")
print(feature_importance_rf)
# ==================== END OF categories and traffic.py ====================


# ==================== Julie - START OF (a).py ====================
# -*- coding: utf-8 -*-
"""(a).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zJU-o8MyNydY-nZ2zZwq5o4F1V-BJXib
"""

import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

# API Credentials (replace with your credentials)
API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"

# API Endpoint
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/keywords_for_site/live"

# Target domains
target_domains = ["covertswarm.com", "apple.com", "google.com"]

# Headers for the API request
HEADERS = {
    "Content-Type": "application/json"
}

# Function to fetch keyword info for a domain
def fetch_keyword_info(domain):
    # Wrap the task in an array
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English"
        }
    ]
    response = requests.post(API_ENDPOINT, headers=HEADERS, json=payload, auth=HTTPBasicAuth(API_USER, API_PASSWORD))

    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # Debugging: Print full response
        if data.get("status_code") == 20000:
            keyword_info_list = []
            for task in data.get("tasks", []):
                for result in task.get("result", []):
                    for item in result.get("items", []):
                        keyword_info = {
                            "domain": domain,
                            "keyword": item.get("keyword", ""),
                            "se_type": result.get("se_type", ""),
                            "last_updated_time": result.get("last_updated_time", ""),
                            "competition": item.get("competition", "N/A"),
                            "competition_level": item.get("competition_level", "UNKNOWN"),
                            "cpc": item.get("cpc", 0.0),
                            "search_volume": item.get("search_volume", 0),
                            "low_top_of_page_bid": item.get("low_top_of_page_bid", 0.0),
                            "high_top_of_page_bid": item.get("high_top_of_page_bid", 0.0),
                            "categories": item.get("categories", []),
                            "monthly_searches": item.get("monthly_searches", [])
                        }
                        keyword_info_list.append(keyword_info)
            # Sort by search volume in ascending order and pick top 3
            keyword_info_list = sorted(keyword_info_list, key=lambda x: x["search_volume"])[:3]
            return keyword_info_list
        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")
    return []


# Fetch and compile keyword data for all domains
all_keywords_info = []
for domain in target_domains:
    print(f"Fetching keywords for {domain}...")
    keyword_info = fetch_keyword_info(domain)
    all_keywords_info.extend(keyword_info)

# Convert results to a DataFrame
df = pd.DataFrame(all_keywords_info)

# Save to CSV
output_file = "top_3_keywords_per_domain.csv"
df.to_csv(output_file, index=False)
print(f"Keyword information saved to {output_file}")

# Display the DataFrame
print(df)

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import time

# API credentials
API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/keywords_for_site/live"

# Domains to query
domains = ["covertswarm.com", "apple.com", "google.com"]

# Initialize a list to store keyword info
all_keywords_data = []

# Fetch data for each domain
for domain in domains:
    print(f"Fetching data for {domain}...")

    # Payload for the API request
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English"
        }
    ]

    # Make the API request
    response = requests.post(API_ENDPOINT, json=payload, auth=HTTPBasicAuth(API_USER, API_PASSWORD))

    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # Debugging: Print the full response

        if data.get("status_code") == 20000:
            for task in data.get("tasks", []):
                for result in task.get("result", []):
                    for item in result.get("items", []):
                        # Extract the required fields
                        keyword_info = {
                            "domain": domain,
                            "keyword": item.get("keyword", "N/A"),
                            "se_type": result.get("se_type", "N/A"),
                            "last_updated_time": result.get("last_updated_time", "N/A"),
                            "competition": item.get("competition", "N/A"),
                            "competition_level": item.get("competition_level", "UNKNOWN"),
                            "cpc": item.get("cpc", 0.0),
                            "search_volume": item.get("search_volume", 0),
                            "low_top_of_page_bid": item.get("low_top_of_page_bid", 0.0),
                            "high_top_of_page_bid": item.get("high_top_of_page_bid", 0.0),
                            "categories": item.get("categories", []),
                            "monthly_searches": item.get("monthly_searches", [])
                        }
                        all_keywords_data.append(keyword_info)
        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")

    # Pause between requests to avoid hitting rate limits
    time.sleep(2)

# Save data to a CSV file
if all_keywords_data:
    df = pd.DataFrame(all_keywords_data)
    csv_filename = "keyword_search_volume.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Keyword information saved to {csv_filename}")
else:
    print("No data retrieved.")

import requests
from requests.auth import HTTPBasicAuth
import time
import pandas as pd

API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/keywords_for_site/live"

# 도메인 리스트
domains = ["covertswarm.com", "apple.com", "google.com"]

# 결과 저장 리스트
all_results = []

for domain in domains:
    print(f"Fetching data for {domain}...")

    # 요청 payload 생성
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English"
        }
    ]

    # API 호출
    response = requests.post(API_ENDPOINT, json=payload, auth=HTTPBasicAuth(API_USER, API_PASSWORD))

    # HTTP 상태 코드 확인
    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # 응답 디버깅

        if data.get("status_code") == 20000:  # 성공 여부 확인
            tasks = data.get("tasks", [])
            for task in tasks:
                for result in task.get("result", []):
                    for item in result.get("items", []):
                        # 필요한 데이터 추출
                        keyword_info = {
                            "domain": domain,
                            "keyword": item.get("keyword", ""),
                            "search_volume": item.get("keyword_info", {}).get("search_volume", 0),
                            "competition_level": item.get("keyword_info", {}).get("competition_level", "UNKNOWN"),
                            "cpc": item.get("keyword_info", {}).get("cpc", 0),
                            "low_top_of_page_bid": item.get("keyword_info", {}).get("low_top_of_page_bid", 0.0),
                            "high_top_of_page_bid": item.get("keyword_info", {}).get("high_top_of_page_bid", 0.0),
                            "categories": item.get("keyword_info", {}).get("categories", []),
                            "monthly_searches": item.get("keyword_info", {}).get("monthly_searches", []),
                        }
                        all_results.append(keyword_info)
        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")

    # **API 호출 사이에 1초 지연 추가**
    time.sleep(1)

# 결과를 데이터프레임으로 변환
df = pd.DataFrame(all_results)

# CSV 파일로 저장
csv_filename = "keyword_search_volume.csv"
df.to_csv(csv_filename, index=False)
print(f"Keyword information saved to {csv_filename}")

import requests
from requests.auth import HTTPBasicAuth
import time
import pandas as pd

API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/keywords_for_site/live"

# Domain list
domains = ["redscan.com", "tekkis.com", "ek.co"]

# Date range filter
start_year = 2022
start_month = 12
start_day = 5
end_year = 2024
end_month = 12
end_day = 4

# Results storage
all_results = []

for domain in domains:
    print(f"Fetching data for {domain}...")

    # Create payload
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English"
        }
    ]

    # API request
    response = requests.post(API_ENDPOINT, json=payload, auth=HTTPBasicAuth(API_USER, API_PASSWORD))

    # Check HTTP status code
    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # Debugging

        if data.get("status_code") == 20000:  # Check success
            tasks = data.get("tasks", [])
            for task in tasks:
                for result in task.get("result", []):
                    for item in result.get("items", []):
                        # Extract base keyword info
                        base_info = {
                            "domain": domain,
                            "keyword": item.get("keyword", ""),
                            "last_updated_time": result.get("last_updated_time", "N/A"),
                            "previous_updated_time": result.get("previous_updated_time", "N/A"),  # Add previous_updated_time
                            "competition": item.get("keyword_info", {}).get("competition", "N/A"),
                            "competition_level": item.get("keyword_info", {}).get("competition_level", "UNKNOWN"),
                            "cpc": item.get("keyword_info", {}).get("cpc", 0),
                            "low_top_of_page_bid": item.get("keyword_info", {}).get("low_top_of_page_bid", 0.0),
                            "high_top_of_page_bid": item.get("keyword_info", {}).get("high_top_of_page_bid", 0.0),
                            "categories": item.get("keyword_info", {}).get("categories", [])
                        }

                        # Filter monthly searches within the specified date range
                        for monthly_data in item.get("keyword_info", {}).get("monthly_searches", []):
                            year = monthly_data.get("year")
                            month = monthly_data.get("month")
                            search_volume = monthly_data.get("search_volume", 0)

                            # Check if the date falls within the desired range
                            if (year > start_year or (year == start_year and month >= start_month)) and \
                               (year < end_year or (year == end_year and month <= end_month)):
                                expanded_info = base_info.copy()
                                expanded_info["year"] = year
                                expanded_info["month"] = month
                                expanded_info["search_volume"] = search_volume
                                all_results.append(expanded_info)
        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")

    # Add delay to avoid hitting API rate limits
    time.sleep(1)

# Convert results to a DataFrame
df = pd.DataFrame(all_results)

# Save to CSV file
csv_filename = "keyword_search_volume_with_update_times.csv"
df.to_csv(csv_filename, index=False)
print(f"Keyword information saved to {csv_filename}")

#Final Ver : last update, date range cant not be setted.
import requests
from requests.auth import HTTPBasicAuth
import time
import pandas as pd

API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/keywords_for_site/live"

# Domain list
domains = ["redscan.com", "tekkis.com", "ek.co"]

# Date range filter
start_year = 2022
start_month = 12
start_day = 5
end_year = 2024
end_month = 12
end_day = 4

# Results storage
all_results = []

for domain in domains:
    print(f"Fetching data for {domain}...")

    # Create payload
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English",
            "include_serp_info": True,  # Ensure SERP information is included
            "include_subdomains": True  # Include subdomains if applicable
        }
    ]

    # API request
    response = requests.post(API_ENDPOINT, json=payload, auth=HTTPBasicAuth(API_USER, API_PASSWORD))

    # Check HTTP status code
    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # Debugging

        if data.get("status_code") == 20000:  # Check success
            tasks = data.get("tasks", [])
            for task in tasks:
                for result in task.get("result", []):
                    for item in result.get("items", []):
                        # Extract base keyword info
                        base_info = {
                            "domain": domain,
                            "keyword": item.get("keyword", ""),
                            "last_updated_time": result.get("last_updated_time", "N/A"),
                            "previous_updated_time": result.get("previous_updated_time", "N/A"),  # Add previous_updated_time
                            "competition": item.get("keyword_info", {}).get("competition", "N/A"),
                            "competition_level": item.get("keyword_info", {}).get("competition_level", "UNKNOWN"),
                            "cpc": item.get("keyword_info", {}).get("cpc", 0),
                            "low_top_of_page_bid": item.get("keyword_info", {}).get("low_top_of_page_bid", 0.0),
                            "high_top_of_page_bid": item.get("keyword_info", {}).get("high_top_of_page_bid", 0.0),
                            "categories": item.get("keyword_info", {}).get("categories", [])
                        }

                        # Filter monthly searches within the specified date range
                        for monthly_data in item.get("keyword_info", {}).get("monthly_searches", []):
                            year = monthly_data.get("year")
                            month = monthly_data.get("month")
                            search_volume = monthly_data.get("search_volume", 0)

                            # Check if the date falls within the desired range
                            if (year > start_year or (year == start_year and month >= start_month)) and \
                               (year < end_year or (year == end_year and month <= end_month)):
                                expanded_info = base_info.copy()
                                expanded_info["year"] = year
                                expanded_info["month"] = month
                                expanded_info["search_volume"] = search_volume
                                all_results.append(expanded_info)
        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")

    # Add delay to avoid hitting API rate limits
    time.sleep(1)

# Convert results to a DataFrame
df = pd.DataFrame(all_results)

# Save to CSV file
csv_filename = "keyword_search_volume_with_update_times.csv"
df.to_csv(csv_filename, index=False)
print(f"Keyword information saved to {csv_filename}")

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import time

# API Credentials
API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/keywords_for_site/live"

# Target domains
domains = ["redscan.com", "tekkis.com", "covertswarm.com"]

# List to store results
all_results = []

# Date range filter
start_year, start_month = 2021, 12
end_year, end_month = 2022, 12

# Loop through each domain to fetch data
for domain in domains:
    print(f"Fetching data for {domain}...")

    # Create payload with necessary parameters
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English",
            "include_serp_info": True,  # Ensure SERP information is included
            "include_subdomains": True  # Include subdomains if applicable
        }
    ]

    # Send API request
    response = requests.post(
        API_ENDPOINT,
        json=payload,
        auth=HTTPBasicAuth(API_USER, API_PASSWORD)
    )

    # Check HTTP response status
    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # Debugging

        if data.get("status_code") == 20000:  # Check if API request was successful
            tasks = data.get("tasks", [])
            for task in tasks:
                for result in task.get("result", []):
                    for item in result.get("items", []):
                        # Extract the required fields
                        keyword_info = {
                            "domain": domain,
                            "keyword": item.get("keyword", "N/A"),
                            "last_updated_time": result.get("last_updated_time", "N/A"),
                            "previous_updated_time": result.get("previous_updated_time", "N/A"),
                            "competition": item.get("keyword_info", {}).get("competition", 0.0),
                            "search_volume": item.get("keyword_info", {}).get("search_volume", 0),
                            "cpc": item.get("keyword_info", {}).get("cpc", 0.0),
                            "low_top_of_page_bid": item.get("keyword_info", {}).get("low_top_of_page_bid", 0.0),
                            "high_top_of_page_bid": item.get("keyword_info", {}).get("high_top_of_page_bid", 0.0),
                            "categories": item.get("keyword_info", {}).get("categories", []),
                            "monthly_searches": item.get("keyword_info", {}).get("monthly_searches", [])
                        }
                        all_results.append(keyword_info)
        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")

    # Pause between requests to avoid hitting API rate limits
    time.sleep(1)

# Convert results to a pandas DataFrame
df = pd.DataFrame(all_results)

# Expand monthly_searches and filter by date range
if not df.empty:
    monthly_data = []
    for _, row in df.iterrows():
        for month_data in row["monthly_searches"]:
            year, month = month_data["year"], month_data["month"]
            # Check if the date falls within the specified range
            if (start_year < year < end_year) or (
                (year == start_year and month >= start_month) or (year == end_year and month <= end_month)
            ):
                monthly_row = {
                    "domain": row["domain"],
                    "keyword": row["keyword"],
                    "last_updated_time": row["last_updated_time"],
                    "previous_updated_time": row["previous_updated_time"],
                    "competition": row["competition"],
                    "search_volume": row["search_volume"],
                    "cpc": row["cpc"],
                    "low_top_of_page_bid": row["low_top_of_page_bid"],
                    "high_top_of_page_bid": row["high_top_of_page_bid"],
                    "categories": row["categories"],
                    "year": year,
                    "month": month,
                    "monthly_search_volume": month_data["search_volume"]
                }
                monthly_data.append(monthly_row)

    # Create expanded DataFrame with monthly search data
    df = pd.DataFrame(monthly_data)

# Save results to CSV
csv_filename = "keyword_data_with_date_filter.csv"
df.to_csv(csv_filename, index=False)
print(f"Keyword data saved to {csv_filename}")


import requests

url = "https://cdn.dataforseo.com/v3/examples/python/python_Client.zip"
response = requests.get(url)

with open("python_Client.zip", "wb") as file:
    file.write(response.content)

print("Download completed.")


import zipfile

with zipfile.ZipFile("python_Client.zip", 'r') as zip_ref:
    zip_ref.extractall("python_Client_extracted")

print("Extraction completed.")


import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import time

# API Credentials
API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/ranked_keywords/live"

# Target domains
domains = ["covertswarm.com", "redscan.com", "tekkis.com", "ek.co"]

# List to store results
all_results = []

# Loop through each domain to fetch data
for domain in domains:
    print(f"Fetching data for {domain}...")

    # Create payload with necessary parameters
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English",
            "date_from": "2022-12-05",
            "date_to": "2024-12-04",
            "include_organic": True,
            "include_paid": True,
        }
    ]

    # Send API request
    response = requests.post(
        API_ENDPOINT,
        json=payload,
        auth=HTTPBasicAuth(API_USER, API_PASSWORD)
    )

    # Check HTTP response status
    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # Debugging

        if data.get("status_code") == 20000:  # Check if API request was successful
            tasks = data.get("tasks", [])
            for task in tasks:
                for result in task.get("result", []):
                    metrics = result.get("metrics", {})
                    for item in result.get("items", []):
                        keyword_info = item.get("keyword_data", {})
                        keyword_details = {
                            "domain": domain,
                            "keyword": keyword_info.get("keyword", "N/A"),
                            "search_volume": keyword_info.get("keyword_info", {}).get("search_volume", 0),
                            "competition": keyword_info.get("keyword_info", {}).get("competition", "N/A"),
                            "competition_level": keyword_info.get("keyword_info", {}).get("competition_level", "N/A"),
                            "cpc": keyword_info.get("keyword_info", {}).get("cpc", 0.0),
                            "last_updated_time": keyword_info.get("keyword_info", {}).get("last_updated_time", "N/A"),
                            "organic_pos_1": metrics.get("organic", {}).get("pos_1", 0),
                            "paid_pos_1": metrics.get("paid", {}).get("pos_1", 0),
                            "organic_etv": metrics.get("organic", {}).get("etv", 0),
                            "paid_etv": metrics.get("paid", {}).get("etv", 0),
                        }

                        # Add monthly searches
                        monthly_searches = keyword_info.get("keyword_info", {}).get("monthly_searches", [])
                        for monthly_data in monthly_searches:
                            year_month = f"{monthly_data['year']}-{monthly_data['month']:02}"
                            keyword_details[year_month] = monthly_data["search_volume"]

                        all_results.append(keyword_details)
        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")

    # Pause between requests to avoid hitting API rate limits
    time.sleep(1)

# Convert results to a pandas DataFrame
df = pd.DataFrame(all_results)

# Save results to CSV
csv_filename = "ranked_keywords_with_traffic.csv"
df.to_csv(csv_filename, index=False)
print(f"Keyword data saved to {csv_filename}")

#Final Ver_Organic Traffic by KWs based on domains_But only the recent 12 months.
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import time

# API Credentials
API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/ranked_keywords/live"

# Target domains
domains = [ "covertswarm.com",
        "redscan.com",
        "tekkis.com",
        "ek.co",
        "approach-cyber.com",
        "coresecurity.com",
        "packetlabs.net",
        "purplesec.us",
        "zelvin.com",
        "breachlock.com",
        "hackerone.com",
        "offsec.com",
        "whiteknightlabs.com",
        "synack.com",
        "bishopfox.com",
        "mitnicksecurity.com",
        "tcm-sec.com",
        "coalfire.com",
        "dionach.com",
        "raxis.com"]

# List to store results
all_results = []

# Date range
date_from = "2022-12-05"
date_to = "2024-12-04"

# Loop through each domain to fetch data
for domain in domains:
    print(f"Fetching data for {domain}...")

    # Create payload with necessary parameters
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English",
            "date_from": date_from,
            "date_to": date_to,
            "include_organic": True,
            "include_paid": True,
        }
    ]

    # Send API request
    response = requests.post(
        API_ENDPOINT,
        json=payload,
        auth=HTTPBasicAuth(API_USER, API_PASSWORD)
    )

    # Check HTTP response status
    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # Debugging

        if data.get("status_code") == 20000:  # Check if API request was successful
            tasks = data.get("tasks", [])
            for task in tasks:
                for result in task.get("result", []):
                    metrics = result.get("metrics", {})
                    for item in result.get("items", []):
                        keyword_info = item.get("keyword_data", {})
                        keyword_details = {
                            "domain": domain,
                            "keyword": keyword_info.get("keyword", "N/A"),
                            "search_volume": keyword_info.get("keyword_info", {}).get("search_volume", 0),
                            "competition": keyword_info.get("keyword_info", {}).get("competition", "N/A"),
                            "competition_level": keyword_info.get("keyword_info", {}).get("competition_level", "N/A"),
                            "cpc": keyword_info.get("keyword_info", {}).get("cpc", 0.0),
                            "last_updated_time": keyword_info.get("keyword_info", {}).get("last_updated_time", "N/A"),
                            # Organic Positions
                            "organic_pos_1": metrics.get("organic", {}).get("pos_1", 0),
                            "organic_pos_2_3": metrics.get("organic", {}).get("pos_2_3", 0),
                            "organic_pos_4_10": metrics.get("organic", {}).get("pos_4_10", 0),
                            "organic_pos_11_20": metrics.get("organic", {}).get("pos_11_20", 0),
                            "organic_pos_21_30": metrics.get("organic", {}).get("pos_21_30", 0),
                            "organic_pos_31_40": metrics.get("organic", {}).get("pos_31_40", 0),
                            "organic_pos_41_50": metrics.get("organic", {}).get("pos_41_50", 0),
                            "organic_pos_51_60": metrics.get("organic", {}).get("pos_51_60", 0),
                            "organic_pos_61_70": metrics.get("organic", {}).get("pos_61_70", 0),
                            "organic_pos_71_80": metrics.get("organic", {}).get("pos_71_80", 0),
                            "organic_pos_81_90": metrics.get("organic", {}).get("pos_81_90", 0),
                            "organic_pos_91_100": metrics.get("organic", {}).get("pos_91_100", 0),
                            # Paid Positions
                            "paid_pos_1": metrics.get("paid", {}).get("pos_1", 0),
                            "paid_pos_2_3": metrics.get("paid", {}).get("pos_2_3", 0),
                            "paid_pos_4_10": metrics.get("paid", {}).get("pos_4_10", 0),
                            "paid_pos_11_20": metrics.get("paid", {}).get("pos_11_20", 0),
                            "paid_pos_21_30": metrics.get("paid", {}).get("pos_21_30", 0),
                            "paid_pos_31_40": metrics.get("paid", {}).get("pos_31_40", 0),
                            "paid_pos_41_50": metrics.get("paid", {}).get("pos_41_50", 0),
                            "paid_pos_51_60": metrics.get("paid", {}).get("pos_51_60", 0),
                            "paid_pos_61_70": metrics.get("paid", {}).get("pos_61_70", 0),
                            "paid_pos_71_80": metrics.get("paid", {}).get("pos_71_80", 0),
                            "paid_pos_81_90": metrics.get("paid", {}).get("pos_81_90", 0),
                            "paid_pos_91_100": metrics.get("paid", {}).get("pos_91_100", 0),
                            "organic_etv": metrics.get("organic", {}).get("etv", 0),
                            "paid_etv": metrics.get("paid", {}).get("etv", 0),
                        }

                        # Add monthly searches
                        monthly_searches = keyword_info.get("keyword_info", {}).get("monthly_searches", [])
                        for monthly_data in monthly_searches:
                            year_month = f"{monthly_data['year']}-{monthly_data['month']:02}"
                            keyword_details[year_month] = monthly_data["search_volume"]

                        all_results.append(keyword_details)
        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")

    # Pause between requests to avoid hitting API rate limits
    time.sleep(1)

# Convert results to a pandas DataFrame
df = pd.DataFrame(all_results)

# Save results to CSV
csv_filename = "ranked_keywords_with_traffic.csv"
df.to_csv(csv_filename, index=False)
print(f"Keyword data saved to {csv_filename}")

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import time

# API Credentials
API_USER = "email used for DataSEO here"
API_PASSWORD = "api key/code here"
API_ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/historical_bulk_traffic_estimation/live"

# Target domains
domains = ["covertswarm.com", "redscan.com", "tekkis.com", "ek.co"]

# Initialize result storage
all_results = []

# Loop through each domain
for domain in domains:
    print(f"Fetching data for {domain}...")

    # Create POST payload
    payload = [
        {
            "target": domain,
            "location_name": "United States",
            "language_name": "English",
            "date_from": "2022-12-01",  # Start date
            "date_to": "2024-12-31",    # End date
            "filters": [
                ["keyword_info.search_volume", ">", 0]
            ],
            "limit": 1000
        }
    ]

    # Send POST request
    response = requests.post(API_ENDPOINT, json=payload, auth=HTTPBasicAuth(API_USER, API_PASSWORD))

    # Check for HTTP response status
    if response.status_code == 200:
        data = response.json()
        print(f"Raw API Response for {domain}:\n{data}")  # Debugging

        # Process valid response
        if data.get("status_code") == 20000:
            tasks = data.get("tasks", [])
            for task in tasks:
                # Check if 'result' key exists and is iterable
                result = task.get("result")
                if result is not None and isinstance(result, list): # add a check here before iterating
                    for res in result:
                        for item in res.get("items", []):
                            # Extract data fields
                            keyword_info = {
                                "domain": domain,
                                "keyword": item.get("keyword", "N/A"),
                                "search_volume": item.get("keyword_info", {}).get("search_volume", 0),
                                "competition": item.get("keyword_info", {}).get("competition", 0.0),
                                "competition_level": item.get("keyword_info", {}).get("competition_level", "UNKNOWN"),
                                "cpc": item.get("keyword_info", {}).get("cpc", 0.0),
                                "last_updated_time": item.get("keyword_info", {}).get("last_updated_time", "N/A"),
                                "organic_pos_1": item.get("metrics", {}).get("organic", {}).get("pos_1", 0),
                                "organic_etv": item.get("metrics", {}).get("organic", {}).get("etv", 0.0),
                                "month": item.get("month", "N/A"),
                                "year": item.get("year", "N/A"),
                            }
                            all_results.append(keyword_info)
                else:
                    print(f"Warning: 'result' key is missing or not a list for domain: {domain}")

        else:
            print(f"Error fetching data for {domain}: {data.get('status_message')}")
    else:
        print(f"HTTP Error {response.status_code} for domain: {domain}")

    # Pause between requests to avoid hitting rate limits
    time.sleep(1)

# Convert results to a pandas DataFrame
df = pd.DataFrame(all_results)

# Save results to CSV
csv_filename = "historical_traffic_estimation.csv"
df.to_csv(csv_filename, index=False)
print(f"Keyword data saved to {csv_filename}")
# ==================== END OF (a).py ====================


# ==================== Julie - START OF (b)_1,2combined.py ====================
# -*- coding: utf-8 -*-
"""(b) 1,2combined

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16lOaqLaXIYLfNcP61aMQYl7ind_Eb-ai
"""

#1. Historical Search Volume - CSV file
import os
import sys
import zipfile
import pandas as pd


# Step 1: Upload the `python_Client` zip file
print("Please upload the `python_Client` zip file.")


import os

# Prompt user for the CSV file path
file_path = input("Enter the path to the CSV file: ").strip()

# List files in the current directory to assist user
print("Available files in the current directory:")
for f in os.listdir("."):
    print(f"- {f}")

# Attempt to load the CSV file with error handling
try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded: {file_path}")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}. Please check the path and try again.")
    exit(1)

print(f"Loaded data from {file_path}")


# Step 2: Automatically detect the uploaded file
if not uploaded:
    print("Error: No file uploaded.")
    sys.exit(1)

# Get the uploaded file name dynamically
zip_file_name = file_path
print(f"Uploaded file: {zip_file_name}")

# Step 3: Define the extraction folder
extracted_folder = "python_Client_extracted"

# Step 4: Unzip the uploaded file
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Verify the extracted content
if not os.path.exists(os.path.join(extracted_folder, "client.py")):
    print(f"Error: client.py not found in {extracted_folder}")
    sys.exit(1)

# Step 5: Append the extracted folder to Python's module search path
sys.path.append(extracted_folder)
try:
    from client import RestClient
    print("Client module imported successfully!")
except ModuleNotFoundError as e:
    print(f"Error importing client module: {e}")
    sys.exit(1)

# Step 6: Initialize the DataForSeo client
client = RestClient("email used for DataSEO here", "api key/cdoe here")  # Replace with your actual credentials

# Step 7: Extract relevant keywords using Google Trends API
initial_keywords = [
    "ethical hacking",
    "Joohyang"
]

google_trends_post_data = dict()
google_trends_post_data[len(google_trends_post_data)] = dict(
    location_name="United States",
    date_from="2022-12-05",  # Broader date range
    date_to="2024-12-04",
    keywords=initial_keywords
)

# Step 8: Send the POST request to Google Trends API
google_trends_response = client.post("/v3/keywords_data/google_trends/explore/live", google_trends_post_data)

# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# Debug the API response
print("Google Trends API Response:")
print(google_trends_response)

# Initialize a variable to store the relevant keywords
relevant_keywords_from_google_trend = []

if google_trends_response["status_code"] == 20000:
    # Extract relevant keywords from the Google Trends response
    tasks = google_trends_response.get("tasks", [])
    for task in tasks:
        result = task.get("result", [])
        for res in result:
            for item in res.get("items", []):
                if "keywords" in item:
                    relevant_keywords_from_google_trend.extend(item["keywords"])

    # Deduplicate keywords
    relevant_keywords_from_google_trend = list(set(relevant_keywords_from_google_trend))
    print(f"Extracted {len(relevant_keywords_from_google_trend)} relevant keywords from Google Trends:")
    print(relevant_keywords_from_google_trend)

    # Check if we have relevant keywords before proceeding
    if not relevant_keywords_from_google_trend:
        print("No relevant keywords found. Exiting.")
        sys.exit(0)

    # Step 9: Fetch detailed keyword metrics using Historical Search Volume API
    historical_post_data = dict()
    historical_post_data[len(historical_post_data)] = dict(
        keywords=relevant_keywords_from_google_trend,
        location_name="United States",
        language_name="English"
    )
    historical_response = client.post("/v3/dataforseo_labs/google/historical_search_volume/live", historical_post_data)

# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
#Final Ver
# Install necessary libraries

import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install_package("pandas")
install_package("openai")


print("Packages installed successfully.")




import os
import sys
import zipfile
import pandas as pd


# Step 1: Upload the `python_Client` zip file
print("Please upload the `python_Client` zip file.")


import os

# Prompt user for the CSV file path
file_path = input("Enter the path to the CSV file: ").strip()

# List files in the current directory to assist user
print("Available files in the current directory:")
for f in os.listdir("."):
    print(f"- {f}")

# Attempt to load the CSV file with error handling
try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded: {file_path}")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}. Please check the path and try again.")
    exit(1)

print(f"Loaded data from {file_path}")


# Step 2: Automatically detect the uploaded file
if not uploaded:
    print("Error: No file uploaded.")
    sys.exit(1)

# Get the uploaded file name dynamically
zip_file_name = file_path
print(f"Uploaded file: {zip_file_name}")

# Step 3: Define the extraction folder
extracted_folder = "python_Client_extracted"

# Step 4: Unzip the uploaded file
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Verify the extracted content
if not os.path.exists(os.path.join(extracted_folder, "client.py")):
    print(f"Error: client.py not found in {extracted_folder}")
    sys.exit(1)

# Step 5: Append the extracted folder to Python's module search path
sys.path.append(extracted_folder)
try:
    from client import RestClient
    print("Client module imported successfully!")
except ModuleNotFoundError as e:
    print(f"Error importing client module: {e}")
    sys.exit(1)

# Step 6: Initialize the DataForSeo client
client = RestClient("email used for DataSEO here", "api key/code here")  # Replace with your actual credentials

# Step 7: Use the provided 1000 keywords
keywords_list = [
    # Paste the full 1000 keywords here
    "Access control", "Advanced persistent threat", "Application security", "Authentication",
    "Authorization", "Behavioral analytics", "Black hat hacking", "Cloud computing security",
    "Cloud security", "Cryptography", "Cyber attack", "Cyber attacks", "Cyber defense",
    "Cyber hygiene", "Cyber incident", "Cyber incident response", "Cyber insurance", "Cyber law",
    "Cyber resilience", "Cyber risk", "Cyber risk management", "Cyber threat",
    "Cyber threat intelligence", "Cyber threats", "Cyber warfare", "Cybercrime",
    "Cybercrime prevention", "Cybersecurity", "Cybersecurity acquisitions",
    "Cybersecurity administrators", "Cybersecurity advancements", "Cybersecurity alliances",
    "Cybersecurity analysis", "Cybersecurity analysts", "Cybersecurity analytics",
    "Cybersecurity applications", "Cybersecurity architects", "Cybersecurity architecture",
    "Cybersecurity architecture assessment", "Cybersecurity architecture design",
    "Cybersecurity architecture design services", "Cybersecurity articles",
    "Cybersecurity assessment", "Cybersecurity assessment and evaluation",
    "Cybersecurity assessment services", "Cybersecurity assessment services provider",
    "Cybersecurity assessment services services", "Cybersecurity assessment tools",
    "Cybersecurity assessments", "Cybersecurity assessments services",
    "Cybersecurity attacks", "Cybersecurity audit", "Cybersecurity audit services",
    "Cybersecurity audit services provider", "Cybersecurity audit services services",
    "Cybersecurity audits", "Cybersecurity audits services", "Cybersecurity awareness",
    "Cybersecurity awareness programs", "Cybersecurity awareness programs provider",
    "Cybersecurity awareness training", "Cybersecurity awareness training programs",
    "Cybersecurity awareness training programs services", "Cybersecurity best practices",
    "Cybersecurity best practices implementation", "Cybersecurity best practices implementation services",
    "Cybersecurity best practices services", "Cybersecurity blogs", "Cybersecurity breach",
    "Cybersecurity breach prevention", "Cybersecurity breach response", "Cybersecurity breaches",
    "Cybersecurity careers", "Cybersecurity certification", "Cybersecurity certification programs",
    "Cybersecurity certifications", "Cybersecurity certifications programs",
    "Cybersecurity certifications programs services", "Cybersecurity certifications services",
    "Cybersecurity challenges", "Cybersecurity collaboration", "Cybersecurity collaborations",
    "Cybersecurity communities", "Cybersecurity community", "Cybersecurity companies",
    "Cybersecurity compliance", "Cybersecurity compliance and regulations",
    "Cybersecurity compliance assessment", "Cybersecurity compliance audits",
    "Cybersecurity compliance management", "Cybersecurity compliance management solutions",
    "Cybersecurity compliance services", "Cybersecurity compliance services services",
    "Cybersecurity conferences", "Cybersecurity consultants", "Cybersecurity consulting",
    "Cybersecurity consulting and advisory", "Cybersecurity consulting and advisory services",
    "Cybersecurity consulting companies", "Cybersecurity consulting firm",
    "Cybersecurity consulting firms", "Cybersecurity consulting services",
    "Cybersecurity consulting services provider", "Cybersecurity consulting services services",
    "Cybersecurity controls", "Cybersecurity controls and safeguards",
    "Cybersecurity controls and safeguards implementation", "Cybersecurity controls assessment",
    "Cybersecurity controls implementation", "Cybersecurity courses", "Cybersecurity defense",
    "Cybersecurity defenses", "Cybersecurity detection", "Cybersecurity detection and response",
    "Cybersecurity developers", "Cybersecurity development", "Cybersecurity devices",
    "Cybersecurity directors", "Cybersecurity distributors", "Cybersecurity education",
    "Cybersecurity educators", "Cybersecurity engineers", "Cybersecurity evaluation",
    "Cybersecurity events", "Cybersecurity experts", "Cybersecurity firms",
    "Cybersecurity forensics", "Cybersecurity forums", "Cybersecurity framework",
    "Cybersecurity frameworks", "Cybersecurity frameworks implementation",
    "Cybersecurity funding", "Cybersecurity governance", "Cybersecurity governance framework",
    "Cybersecurity governance framework development", "Cybersecurity governance framework development services",
    "Cybersecurity governance frameworks", "Cybersecurity grants", "Cybersecurity guidelines",
    "Cybersecurity hardware", "Cybersecurity implementation", "Cybersecurity incident",
    "Cybersecurity incident analysis", "Cybersecurity incident communication",
    "Cybersecurity incident detection", "Cybersecurity incident documentation",
    "Cybersecurity incident escalation", "Cybersecurity incident handling",
    "Cybersecurity incident investigation", "Cybersecurity incident management",
    "Cybersecurity incident management plan", "Cybersecurity incident management services",
    "Cybersecurity incident mitigation", "Cybersecurity incident prevention",
    "Cybersecurity incident recovery", "Cybersecurity incident reporting",
    "Cybersecurity incident response", "Cybersecurity incident response best practices",
    "Cybersecurity incident response communication plan", "Cybersecurity incident response exercise",
    "Cybersecurity incident response framework", "Cybersecurity incident response guidelines",
    "Cybersecurity incident response management", "Cybersecurity incident response plan",
    "Cybersecurity incident response plan example", "Cybersecurity incident response plan example pdf",
    "Cybersecurity incident response plan template", "Cybersecurity incident response planning",
    "Cybersecurity incident response planning services", "Cybersecurity incident response playbook",
    "Cybersecurity incident response policy", "Cybersecurity incident response procedures",
    "Cybersecurity incident response process", "Cybersecurity incident response services",
    "Cybersecurity incident response services services", "Cybersecurity incident response tabletop exercise",
    "Cybersecurity incident response team", "Cybersecurity incident response team roles and responsibilities",
    "Cybersecurity incident response training", "Cybersecurity incident response training program",
    "Cybersecurity incidents", "Cybersecurity industry", "Cybersecurity infrastructure",
    "Cybersecurity innovations", "Cybersecurity integrators", "Cybersecurity intelligence",
    "Cybersecurity internships", "Cybersecurity investments", "Cybersecurity jobs",
    "Cybersecurity joint ventures", "Cybersecurity laws", "Cybersecurity leaders",
    "Cybersecurity leadership", "Cybersecurity management", "Cybersecurity management and governance",
    "Cybersecurity management and governance framework", "Cybersecurity management services",
    "Cybersecurity management solutions", "Cybersecurity management solutions provider",
    "Cybersecurity managers", "Cybersecurity manufacturers", "Cybersecurity market",
    "Cybersecurity measures", "Cybersecurity measures evaluation", "Cybersecurity measures implementation",
    "Cybersecurity measures implementation provider", "Cybersecurity measures services",
    "Cybersecurity mergers", "Cybersecurity mitigation", "Cybersecurity mitigation strategies",
    "Cybersecurity monitoring", "Cybersecurity monitoring services", "Cybersecurity monitoring services provider",
    "Cybersecurity networks", "Cybersecurity news", "Cybersecurity offense",
    "Cybersecurity officers", "Cybersecurity operations", "Cybersecurity operations center",
    "Cybersecurity operations management", "Cybersecurity operations management services",
    "Cybersecurity operations management solutions", "Cybersecurity opportunities",
    "Cybersecurity organizations", "Cybersecurity partners", "Cybersecurity partnerships",
    "Cybersecurity penetration testing", "Cybersecurity planning", "Cybersecurity platforms",
    "Cybersecurity policies", "Cybersecurity policies and procedures",
    "Cybersecurity policies and procedures implementation", "Cybersecurity policies development",
    "Cybersecurity policy", "Cybersecurity policy development", "Cybersecurity policy development services",
    "Cybersecurity policy development services services", "Cybersecurity policy implementation",
    "Cybersecurity practices", "Cybersecurity prevention", "Cybersecurity procedures",
    "Cybersecurity products", "Cybersecurity professionals", "Cybersecurity program",
    "Cybersecurity protection", "Cybersecurity protection services", "Cybersecurity protection services provider",
    "Cybersecurity protocols", "Cybersecurity protocols development", "Cybersecurity protocols services",
    "Cybersecurity providers", "Cybersecurity publications", "Cybersecurity recovery",
    "Cybersecurity recovery planning", "Cybersecurity regulations", "Cybersecurity requirements",
    "Cybersecurity research", "Cybersecurity researchers", "Cybersecurity resellers",
    "Cybersecurity resilience", "Cybersecurity resilience building", "Cybersecurity resources",
    "Cybersecurity risk", "Cybersecurity risk analysis", "Cybersecurity risk assessment",
    "Cybersecurity risk assessment and management", "Cybersecurity risk assessment and mitigation",
    "Cybersecurity risk assessment audits", "Cybersecurity risk assessment best practices",
    "Cybersecurity risk assessment breaches", "Cybersecurity risk assessment certifications",
    "Cybersecurity risk assessment checklist", "Cybersecurity risk assessment compliance",
    "Cybersecurity risk assessment consulting", "Cybersecurity risk assessment controls",
    "Cybersecurity risk assessment defense", "Cybersecurity risk assessment framework",
    "Cybersecurity risk assessment framework nist", "Cybersecurity risk assessment frameworks",
    "Cybersecurity risk assessment guidelines", "Cybersecurity risk assessment matrix",
    "Cybersecurity risk assessment measures", "Cybersecurity risk assessment methodologies",
    "Cybersecurity risk assessment methodology", "Cybersecurity risk assessment methodology framework",
    "Cybersecurity risk assessment plan", "Cybersecurity risk assessment process",
    "Cybersecurity risk assessment processes", "Cybersecurity risk assessment programs",
    "Cybersecurity risk assessment protection", "Cybersecurity risk assessment providers",
    "Cybersecurity risk assessment questionnaire", "Cybersecurity risk assessment regulations",
    "Cybersecurity risk assessment report", "Cybersecurity risk assessment services",
    "Cybersecurity risk assessment software", "Cybersecurity risk assessment solutions",
    "Cybersecurity risk assessment standards", "Cybersecurity risk assessment techniques",
    "Cybersecurity risk assessment technology", "Cybersecurity risk assessment template",
    "Cybersecurity risk assessment threats", "Cybersecurity risk assessment tool",
    "Cybersecurity risk assessment tools", "Cybersecurity risk assessment training",
    "Cybersecurity risk assessment trends", "Cybersecurity risk controls",
    "Cybersecurity risk framework", "Cybersecurity risk governance", "Cybersecurity risk management",
    "Cybersecurity risk management framework", "Cybersecurity risk management services",
    "Cybersecurity risk management services services", "Cybersecurity risk management solutions",
    "Cybersecurity risk management strategies", "Cybersecurity risk management strategies services",
    "Cybersecurity risk mitigation", "Cybersecurity risk monitoring"
]

# Step 8: Extract data from Google Trends API
google_trends_post_data = dict()
google_trends_post_data[len(google_trends_post_data)] = dict(
    location_name="United States",
    date_from="2022-12-05",
    date_to="2024-12-04",
    keywords=keywords_list
)

google_trends_response = client.post("/v3/keywords_data/google_trends/explore/live", google_trends_post_data)

# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# Debug the API response
print("Google Trends API Response:")
print(google_trends_response)

# Initialize a variable to store the relevant keywords
relevant_keywords_from_google_trend = []

if google_trends_response["status_code"] == 20000:
    tasks = google_trends_response.get("tasks", [])
    for task in tasks:
        result = task.get("result", [])
        # Check if result is a list before iterating
        if isinstance(result, list): # this checks if result is a list
            for res in result:
                for item in res.get("items", []):
                    if "keywords" in item:
                        relevant_keywords_from_google_trend.extend(item["keywords"])
        else:
            print(f"Warning: 'result' is not a list for task: {task}. Skipping this task.")


    # Deduplicate keywords
    relevant_keywords_from_google_trend = list(set(relevant_keywords_from_google_trend))
    print(f"Extracted {len(relevant_keywords_from_google_trend)} relevant keywords from Google Trends:")
    print(relevant_keywords_from_google_trend)

    if not relevant_keywords_from_google_trend:
        print("No relevant keywords found. Continuing with original keywords.")
        # Instead of exiting, continue with the original keywords_list
        relevant_keywords_from_google_trend = keywords_list


    # Step 9: Fetch detailed keyword metrics using Historical Search Volume API
    historical_post_data = dict()
    historical_post_data[len(historical_post_data)] = dict(
        keywords=relevant_keywords_from_google_trend,
        location_name="United States",
        language_name="English"
    )
    historical_response = client.post("/v3/dataforseo_labs/google/historical_search_volume/live", historical_post_data)

# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
#2 Trying to extract the relevant keywords
import os
import sys
import zipfile
import pandas as pd


# Step 1: Upload the python_Client zip file
print("Please upload the python_Client zip file.")


import os

# Prompt user for the CSV file path
file_path = input("Enter the path to the CSV file: ").strip()

# List files in the current directory to assist user
print("Available files in the current directory:")
for f in os.listdir("."):
    print(f"- {f}")

# Attempt to load the CSV file with error handling
try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded: {file_path}")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}. Please check the path and try again.")
    exit(1)

print(f"Loaded data from {file_path}")


# Automatically detect the uploaded file
if not uploaded:
    print("Error: No file uploaded.")
    sys.exit(1)

# Extract the uploaded zip file
zip_file_name = file_path
print(f"Uploaded file: {zip_file_name}")

extracted_folder = "python_Client_extracted"
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Verify the extracted content
if not os.path.exists(os.path.join(extracted_folder, "client.py")):
    print(f"Error: client.py not found in {extracted_folder}")
    sys.exit(1)

# Add the extracted folder to Python's module search path
sys.path.append(extracted_folder)
try:
    from client import RestClient
    print("Client module imported successfully!")
except ModuleNotFoundError as e:
    print(f"Error importing client module: {e}")
    sys.exit(1)

# Initialize the DataForSeo client
client = RestClient("email used for DataSEO here", "api key/code here")  # Replace with your actual credentials

# Define the initial keywords
initial_keywords = [
    "samsung",
    "apple"
]

# Prepare the Google Trends API request payload
google_trends_post_data = dict()
google_trends_post_data[len(google_trends_post_data)] = dict(
    location_name="United States",
    date_from="2023-01-01",
    date_to="2024-11-24",
    keywords=initial_keywords
)

# Fetch data from Google Trends API
google_trends_response = client.post("/v3/keywords_data/google_trends/explore/live", google_trends_post_data)

# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# Debug: Print the API response
print("Google Trends API Response:")
print(google_trends_response)

if google_trends_response["status_code"] == 20000:
    tasks = google_trends_response.get("tasks", [])
    related_topics = []  # List to store related topics

    # Enhanced inspection of the API response
    print("Inspecting API response for related topics...")
    for task in tasks:
        result = task.get("result", [])
        for res in result:
            for item in res.get("items", []):
                if item.get("type") == "google_trends_topics_list":  # Check for related topics list
                    print(f"Found related topics: {item}")
                    for topic in item.get("data", {}).get("top", []):  # Extract "top" related topics
                        related_topics.append({
                            "topic_title": topic.get("topic_title"),
                            "topic_type": topic.get("topic_type"),
                            "value": topic.get("value")
                        })

    if not related_topics:
        print("No data found in `google_trends_topics_list`. Attempting fallback using `google_trends_graph`.")
        # Fallback to other response fields if no related topics are found
        for task in tasks:
            result = task.get("result", [])
            for res in result:
                for item in res.get("items", []):
                    if item.get("type") == "google_trends_graph":
                        related_topics = [{"topic_title": kw, "topic_type": "Keyword", "value": 0} for kw in item.get("keywords", [])]
                        print(f"Using fallback keywords: {related_topics}")
                        break

    if not related_topics:
        print("No relevant topics or keywords found. Exiting.")
        sys.exit(0)

    # Sort topics by value in descending order
    related_topics = sorted(related_topics, key=lambda x: x["value"], reverse=True)
    print("\nSorted Related Topics by Value:")
    for topic in related_topics:
        print(topic)

    # Extract topic titles for fetching keyword data
    relevant_keywords = [topic["topic_title"] for topic in related_topics if topic["topic_title"]]

    # Ensure relevant_keywords is not empty
    if not relevant_keywords:
        print("No relevant keywords extracted. Exiting.")
        sys.exit(0)

    print(f"Relevant keywords extracted: {relevant_keywords}")
else:
    print("Error with Google Trends API. Code: %d Message: %s" %
          (google_trends_response["status_code"], google_trends_response["status_message"]))

#3. Nick ver_was able to take out the strength(value)of each keywords(https://docs.dataforseo.com/v3/keywords_data/google_trends/explore/live/?python)
# Value here means, increase in the search term popularity. indicates the relative increase in the search term popularity within the given timeframe the value is provided in percentage (without the % sign)

# Debug: Print the entire API response for inspection
import json
print("Google Trends API Raw Response:")
print(json.dumps(google_trends_response, indent=4))

if google_trends_response["status_code"] == 20000:
    tasks = google_trends_response.get("tasks", [])
    related_topics = []  # List to store related topics

    # Enhanced inspection of the API response
    print("Inspecting API response for related topics...")
    for task in tasks:
        result = task.get("result", [])
        for res in result:
            for item in res.get("items", []):
                if item.get("type") == "google_trends_topics_list":  # Check for related topics list
                    print(f"Found related topics: {item}")
                    for topic in item.get("data", {}).get("top", []):  # Extract "top" related topics
                        value = topic.get("value")  # Safely extract 'value'
                        if value is not None:
                            related_topics.append({
                                "topic_title": topic.get("topic_title"),
                                "topic_type": topic.get("topic_type"),
                                "value": value
                            })
                        else:
                            print(f"Warning: Missing 'value' for topic: {topic}")

    if not related_topics:
        print("No data found in `google_trends_topics_list`. Attempting fallback using `google_trends_graph`.")
        # Fallback to other response fields if no related topics are found
        for task in tasks:
            result = task.get("result", [])
            for res in result:
                for item in res.get("items", []):
                    if item.get("type") == "google_trends_graph":
                        for kw in item.get("keywords", []):
                            related_topics.append({
                                "topic_title": kw,
                                "topic_type": "Keyword",
                                "value": 0  # Default fallback for graph keywords
                            })
                        print(f"Using fallback keywords: {related_topics}")
                        break

    # Exit if no topics are found
    if not related_topics:
        print("No relevant topics or keywords found. Exiting.")
        sys.exit(0)

    # Sort topics by value in descending order
    related_topics = sorted(related_topics, key=lambda x: x["value"], reverse=True)
    print("\nSorted Related Topics by Value:")
    for topic in related_topics:
        print(topic)

    # Extract topic titles for fetching keyword data
    relevant_keywords = [topic["topic_title"] for topic in related_topics if topic["topic_title"]]

    # Ensure relevant_keywords is not empty
    if not relevant_keywords:
        print("No relevant keywords extracted. Exiting.")
        sys.exit(0)

    print(f"Relevant keywords extracted: {relevant_keywords}")
else:
    print("Error with Google Trends API. Code: %d Message: %s" %
          (google_trends_response["status_code"], google_trends_response["status_message"]))
# ==================== END OF (b)_1,2combined.py ====================


# ==================== Julie - START OF 1) Traffic API 250214.py ====================
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:45:50 2024

@author: juanj
"""

import sys
import csv
from client import RestClient

# Add the directory containing client.py to the Python path
sys.path.append(r'/Users/midoriwest/Desktop/Code/BeachHead/Final_Codes/Master_Code/client.py')

# Initialize RestClient with your credentials
client = RestClient("email used for DataSEO here", "api key/code here")

# Prepare the post data
post_data = dict()
post_data[len(post_data)] = dict(
   targets=[
       "covertswarm.com",
       "redscan.com",
       "tekkis.com",
       "ek.co",
       "coresecurity.com",
       "packetlabs.net",
       "purplesec.us",
       "zelvin.com",
       "breachlock.com",
       "hackerone.com",
       "offsec.com",
       "whiteknightlabs.com",
       "synack.com",
       "bishopfox.com",
       "mitnicksecurity.com",
       "tcm-sec.com",
       "coalfire.com",
       "dionach.com",
       "raxis.com"
   ],
   location_name="United States",
   language_name="English",
   date_from="2022-12-05",
   date_to="2024-12-04",
   item_types=[
       "organic",
       "paid"
   ]
)

# Make the API request
response = client.post("/v3/dataforseo_labs/google/historical_bulk_traffic_estimation/live", post_data)

# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# ✅ API RESPONSE CHECK
if 'response' in locals() and isinstance(response, dict):
    print("🔎 API Response Status Code:", response.get("status_code"))
    print("📄 API Response Status Message:", response.get("status_message"))
    if response.get("status_code") != 20000:
        print("🚨 Warning: API response indicates an issue. Check API keys, credits, or parameters.")
    else:
        print("✅ API response received successfully.")
else:
    print("⚠️ No valid API response found.")
# Check if the response is successful
if response["status_code"] == 20000:
   # Extract the result data
   results = response["tasks"][0]["result"]

   # Specify the CSV file name
   csv_file = "api_results5.csv"

   # Open the CSV file for writing
   with open(csv_file, mode='w', newline='') as file:
       # Create a CSV writer object
       writer = csv.DictWriter(file, fieldnames=results[0].keys())

       # Write the header
       writer.writeheader()

       # Write the data rows
       for result in results:
           writer.writerow(result)

   print(f"Data exported to {csv_file} successfully.")
else:
   print("Error. Code: %d Message: %s" % (response["status_code"], response["status_message"]))

# ==================== END OF 1) Traffic API 250214.py ====================


# ==================== Juan - START OF 2) Json flatening traffic.py ====================
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:30:27 2025

@author: juanj
"""


import pandas as pd
import json
import ast  

# Step 1: Load the CSV file
file_path = r'/Users/midoriwest/Desktop/Code/BeachHead/Final_Codes/Master_Code/api_results5.csv'
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
# ==================== END OF 2) Json flatening traffic.py ====================


# ==================== Juan - START OF 3) Traffic grouped data.py ====================
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

# ==================== END OF 3) Traffic grouped data.py ====================


# ==================== Juan - START OF 4) gROUPED tRAFFIC NO TARGETS.py ====================
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


# ==================== END OF 4) gROUPED tRAFFIC NO TARGETS.py ====================


# ==================== Juan - START OF 5) News events Api pull and json flattening 250214.py ====================
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:17:12 2025

@author: juanj
"""

import eventregistry
import pandas as pd
import os
import ast
import numpy as np

from eventregistry import *
er = EventRegistry(apiKey = "event registry api key/code here")

query = {
    "$query": {
        "$and": [
            {
                "$or": [
                    {
                        "$and": [
                            {"keyword": "data", "keywordLoc": "title"},
                            {"keyword": "breach", "keywordLoc": "title"}
                        ]
                    },
                    {
                        "$and": [
                            {"keyword": "data", "keywordLoc": "title"},
                            {"keyword": "compromised", "keywordLoc": "title"}
                        ]
                    },
                    {
                        "$and": [
                            {"keyword": "cyber", "keywordLoc": "title"},
                            {"keyword": "attack", "keywordLoc": "title"}
                        ]
                    }
                ]
            },
            {
                "dateStart": "2022-12-05",
                "dateEnd": "2024-12-04",
                "lang": "eng",
                "minArticlesInEvent": 10
            }
        ]
    }
}

# Initialize QueryEventsIter with the complex query
q = QueryEventsIter.initWithComplexQuery(query)

# Prepare a list to hold all event data
all_events_data = []


# Execute the query and iterate over the results
for event in q.execQuery(er, maxItems=2000):
    all_events_data.append({
        "Title": event.get("title"),
        "Date": event.get("eventDate"),
        "Source": event.get("source", {}).get("uri"),
        "URL": event.get("url"),
        "Weight": event.get("wgt"),
        "Sentiment": event.get("sentiment"),
        "Social Score": event.get("socialScore"),
        "Relevance": event.get("relevance"),
        "Article Count": event.get("articleCounts")
    })
    
df = pd.DataFrame(all_events_data)
print (df)   

# Save to CSV
csv_path = "cyber_attack_events_with_additional_info.csv"
df.to_csv(csv_path, index=False)

##############################################################################
import pandas as pd
import ast

# Define the filtering structure for titles
filter_conditions = [
    {"$and": [{"keyword": "data", "keywordLoc": "title"}, {"keyword": "breach", "keywordLoc": "title"}]},
    {"$and": [{"keyword": "data", "keywordLoc": "title"}, {"keyword": "compromised", "keywordLoc": "title"}]},
    {"$and": [{"keyword": "cyber", "keywordLoc": "title"}, {"keyword": "attack", "keywordLoc": "title"}]},
]

# Helper function to check if a title matches the conditions
def matches_conditions(title, conditions):
    title_lower = title.lower()
    for condition in conditions:
        if "$and" in condition:
            if all(keyword["keyword"] in title_lower for keyword in condition["$and"]):
                return True
    return False

# Load dataset (replace 'csv_path' with your dataset file)
data = pd.read_csv(csv_path)

# Extract and clean titles (assuming the 'Title' column contains JSON-like strings)
data["Cleaned Title"] = data["Title"].apply(lambda x: ast.literal_eval(x).get("eng", "") if isinstance(x, str) else x)

# Filter rows where titles match the conditions
filtered_data = data[data["Cleaned Title"].apply(lambda title: matches_conditions(title, filter_conditions))]

# Ensure the 'Date' column in filtered_data is in datetime format
filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], errors='coerce')

# Filter out articles that are not in ENG
def extract_eng_count(article_count):
    try:
        # Parse the JSON-like string and extract the "eng" count
        return int(ast.literal_eval(article_count).get("eng", 0))
    except (ValueError, SyntaxError, AttributeError):
        return 0  # Return 0 if parsing fails

# Apply the function to the "Article Count" column
data["Article Count (ENG)"] = data["Article Count"].apply(extract_eng_count)

# Convert the 'Date' column to a datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Add a new column for the week of the year
data['Week'] = data['Date'].dt.isocalendar().week

# Calculate the absolute week starting from week 1
min_date = data['Date'].min()
data['Absolute Week'] = ((data['Date'] - min_date).dt.days // 7) + 1

# Apply the same logic to filtered_data
filtered_data["Week"] = filtered_data["Date"].dt.isocalendar().week
filtered_data["Absolute Week"] = ((filtered_data["Date"] - min_date).dt.days // 7) + 1

# Group the filtered data by "Absolute Week" and calculate relevant aggregations
grouped_data = filtered_data.groupby("Absolute Week").agg({
    "Cleaned Title": "count",  # Count the number of titles in each week
    "Weight": "sum",           # Sum the weight for each week
    "Sentiment": "mean",       # Average sentiment score for each week
    "Relevance": "mean",       # Average relevance score for each week
    "Date": "first"            # Include the first occurrence of the Date column for each week
}).rename(columns={"Cleaned Title": "Article Count"})

# Add the unique title count for each Absolute Week
grouped_data["Unique Title Count"] = filtered_data.groupby("Absolute Week")["Cleaned Title"].nunique()

# Reset the index for better readability
grouped_data.reset_index(inplace=True)

# Save the grouped data to a CSV file
grouped_data.to_csv("grouped_data_with_title_count3.csv", index=False)

# Display the grouped data
print("Grouped Data by Absolute Week with Title Count:")
print(grouped_data)


# ==================== END OF 5) News events Api pull and json flattening 250214.py ====================


# ==================== Juan - START OF 6) News events plus date.py ====================
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





# ==================== END OF 6) News events plus date.py ====================


# ==================== Juan - START OF 7) Traffic and events analysis file.py ====================
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



# ==================== END OF 7) Traffic and events analysis file.py ====================


# ==================== Juan - START OF 8) events vs traffic analysis.py ====================
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
# ==================== END OF 8) events vs traffic analysis.py ====================


# ================================================================================
# Greg - Section: Easy Organic Keywords.py
# ================================================================================
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



# ================================================================================
# Greg Section: Easy Paid Keywords.py
# ================================================================================
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

