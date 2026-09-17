import pandas as pd


def main():
    csv_file = 'log_service.csv'
    print(f"Reading {csv_file} to find unique log names and sample values...")
    
    try:
        df = pd.read_csv(csv_file, usecols=['log_name', 'value'])
        unique_log_names = df['log_name'].dropna().unique()
        
        print("\nDifferent types of entries in the 'log_name' column with up to 2 sample values:")
        for name in sorted(unique_log_names):
            print(f"\n- {name}")
            sample_values = df[df['log_name'] == name]['value'].dropna().unique()[:2]
            for val in sample_values:
                val_str = str(val)
                if len(val_str) > 200:
                    val_str = val_str[:197] + "..."
                print(f"    * {val_str}")
            
    except Exception as e:
        print(f"Error reading the CSV file: {e}")

if __name__ == "__main__":
    main()
