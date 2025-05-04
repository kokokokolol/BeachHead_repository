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
client = RestClient("nl2804@stern.nyu.edu", "da47243e760b6c2b") #Your credentials should be found here after making an account: https://app.dataforseo.com/api-access

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

    if response.get("status_code") == 20000:
        tasks = response.get("tasks", [])
        for task in tasks:
            final_results["tasks"].append(task)
            final_results["tasks_count"] += 1
            final_results["cost"] += task.get("cost", 0.0)

            # Ensure "result" is not None before iterating
            results = task.get("result")
            if results:  # Check if results is not None
                for result in results:
                    for item in result.get("items", []):
                        ranked_keywords_info = item.get("ranked_keywords_info", {})
                        csv_data.append({
                            "Target": result.get("target", ""),
                            "Type": item.get("type", ""),
                            "Domain From": item.get("domain_from", ""),
                            "Domain To": item.get("domain_to", ""),
                            "URL From": item.get("url_from", ""),
                            "URL To": item.get("url_to", ""),
                            "TLD From": item.get("tld_from", ""),
                            "Backlink Spam Score": item.get("backlink_spam_score", 0),
                            "Rank": item.get("rank", 0),
                            "Page From Rank": item.get("page_from_rank", 0),
                            "Domain From Rank": item.get("domain_from_rank", 0),
                            "Domain From Platform Type": ", ".join(item.get("domain_from_platform_type", [])) if isinstance(item.get("domain_from_platform_type"), list) else "",
                            "Page From External Links": item.get("page_from_external_links", 0),
                            "Page From Internal Links": item.get("page_from_internal_links", 0),
                            "Page From Language": item.get("page_from_language", ""),
                            "First Seen": item.get("first_seen", ""),
                            "Prev Seen": item.get("prev_seen", ""),
                            "Last Seen": item.get("last_seen", ""),
                            "Item Type": item.get("item_type", ""),
                            "Attributes": ", ".join(item.get("attributes", [])) if isinstance(item.get("attributes"), list) else "",
                            "Links Count": item.get("links_count", 0),
                            "Group Count": item.get("group_count", 0),
                            "URL To Spam Score": item.get("url_to_spam_score", 0),
                            "Ranked Keywords Info": json.dumps(ranked_keywords_info),
                            "Page From Keywords Count Top 3": ranked_keywords_info.get("page_from_keywords_count_top_3", 0),
                            "Page From Keywords Count Top 10": ranked_keywords_info.get("page_from_keywords_count_top_10", 0),
                            "Page From Keywords Count Top 100": ranked_keywords_info.get("page_from_keywords_count_top_100", 0)
                        })
    else:
        final_results["tasks_error"] += 1
        final_results["tasks"].append({
            "id": None,
            "status_code": response.get("status_code"),
            "status_message": response.get("status_message"),
            "result": None
        })

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
