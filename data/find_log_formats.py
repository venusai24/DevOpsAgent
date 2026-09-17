import random

import pandas as pd


def main():
    csv_file = 'log_service.csv'
    print(f"Scanning {csv_file} to find unique log names and random samples...")
    
    try:
        # Load necessary columns
        df = pd.read_csv(csv_file, usecols=['log_name', 'value'])
        
        # Remove any rows with missing log_name or value
        df = df.dropna(subset=['log_name', 'value'])
        
        # Find unique log names and sort them
        unique_log_names = sorted(df['log_name'].unique())
        
        print("\n" + "="*50)
        print("UNIQUE LOG NAMES AND RANDOM SAMPLES")
        print("="*50)
        print(f"Unique log names found: {unique_log_names}\n")
        
        # Print 3 random sample values for each type of log_name
        for log_name in unique_log_names:
            print(f"[{log_name}] (3 random samples):")
            print("-" * 50)
            
            # Filter values for the current log_name
            values = df[df['log_name'] == log_name]['value'].tolist()
            
            # Select 3 random samples (or fewer if not enough are present)
            num_samples = min(3, len(values))
            samples = random.sample(values, num_samples)
            
            for idx, val in enumerate(samples, 1):
                print(f"Sample {idx}:")
                print(val)
                print()
            print("-" * 50 + "\n")
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")

if __name__ == "__main__":
    main()

