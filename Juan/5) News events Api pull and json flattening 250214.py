# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:17:12 2025

@author: juanj
"""

pip install eventregistry

import eventregistry
import pandas as pd
import os
import ast
import numpy as np

from eventregistry import *
er = EventRegistry(apiKey = "5d8c154c-79c1-4673-9bac-6323d418e120")

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

