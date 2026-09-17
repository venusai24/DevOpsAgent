import pandas as pd


def main():
    csv_file = 'metric_container.csv'
    output_file = 'unique_metric_container_values.txt'
    
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file, usecols=['cmdb_id', 'kpi_name'])
    
    # Extract unique values
    unique_cmdb_ids = sorted(df['cmdb_id'].dropna().unique().astype(str))
    unique_kpi_names = sorted(df['kpi_name'].dropna().unique().astype(str))
    
    print(f"Found {len(unique_cmdb_ids)} unique CMDB IDs and {len(unique_kpi_names)} unique KPI names.")
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write(f"=== UNIQUE CMDB IDs ({len(unique_cmdb_ids)}) ===\n")
        for cmdb_id in unique_cmdb_ids:
            f.write(f"{cmdb_id}\n")
            
        f.write(f"\n=== UNIQUE KPI NAMES ({len(unique_kpi_names)}) ===\n")
        for kpi_name in unique_kpi_names:
            f.write(f"{kpi_name}\n")
            
    print(f"Results written to {output_file}")

if __name__ == '__main__':
    main()
