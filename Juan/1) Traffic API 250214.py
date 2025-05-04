# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:45:50 2024

@author: juanj
"""

import sys
import csv
from client import RestClient

# Add the directory containing client.py to the Python path
sys.path.append(r'C:\Users\juanj\OneDrive\Escritorio\python_Client')

# Initialize RestClient with your credentials
client = RestClient("jdiazinfante@sicrea.com.mx", "c484537de45160d5")

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
