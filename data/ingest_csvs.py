import os
import sqlite3

import pandas as pd

db_path = 'incident_data.db'

# Connect to SQLite database (this will create it if it doesn't exist)
conn = sqlite3.connect(db_path)

files = [
    'cluster_app_stats.csv',
    'cluster_incident_logs.csv',
    'incident_traces.csv',
    'redis02_metrics.csv'
]

for file in files:
    if os.path.exists(file):
        table_name = file.replace('.csv', '')
        print(f"Ingesting {file} into table '{table_name}'...")
        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(file)
        # Write the data to a sqlite table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    else:
        print(f"Warning: File {file} not found.")

print("\n--- Query Result ---")
print("Distinct log_name values in cluster_incident_logs:")
cursor = conn.cursor()
try:
    cursor.execute("SELECT DISTINCT log_name FROM cluster_incident_logs;")
    for row in cursor.fetchall():
        print(f"- {row[0]}")
except Exception as e:
    print(f"Error executing query: {e}")

conn.close()
print(f"\nSuccess! Data ingested into {db_path}.")
