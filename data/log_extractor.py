import re

import pandas as pd

# Define regex patterns with named capture groups to extract specific numeric/important fields
PATTERNS = {
    # Extract HTTP status, response size in bytes, and latencies (ms and seconds)
    'apache_access_log': re.compile(r'HTTP/1\.[01]"\s+(?P<status>\d+)\s+(?P<response_size>\d+|\-).*?api_url\s+(?P<latency_ms>\d+)\s+(?P<latency_s>[\d\.]+)'),
    
    # Similar to apache_access_log but slightly different string format
    'localhost_access_log': re.compile(r'HTTP/1\.[01]\s+(?P<status>\d+)\s+(?P<response_size>\d+|\-).*?api_url\s+(?P<latency_ms>\d+)\s+(?P<latency_s>[\d\.]+)'),
    
    # Extract query execution times and row metrics
    'slow': re.compile(r'Query_time:\s+(?P<query_time>[\d\.]+)\s+Lock_time:\s+(?P<lock_time>[\d\.]+)\s+Rows_sent:\s+(?P<rows_sent>\d+)\s+Rows_examined:\s+(?P<rows_examined>\d+)'),
    
    # Extract memory before/after GC, and CPU times
    'gc': re.compile(r'(?P<heap_before>\d+)K->(?P<heap_after>\d+)K.*?user=(?P<user_time>[\d\.]+)\s+sys=(?P<sys_time>[\d\.]+)\s+real=(?P<real_time>[\d\.]+)'),
    
    # Extract InnoDB loop metrics if present
    'error': re.compile(r'took\s+(?P<loop_took_ms>\d+)ms.*?flushed=(?P<pages_flushed>\d+).*?evicted=(?P<pages_evicted>\d+)'),
    
    # For catalina logs, primarily extract log level and message (less numeric data, more categorical)
    'catalina': re.compile(r'(?P<level>[A-Z]+)\s+\[(?P<thread>.*?)\]\s+(?P<component>\S+)\s+(?P<message>.*)')
}

def extract_features(log_name, log_string):
    """
    Apply the relevant regex pattern based on log_name to extract fields into a dictionary.
    """
    pattern = PATTERNS.get(log_name)
    if pattern:
        match = pattern.search(log_string)
        if match:
            return match.groupdict()
    return {} # Return empty if no match or no pattern

def main():
    csv_file = 'log_service.csv'
    print(f"Parsing logs from {csv_file} in chunks to save memory...\n")
    
    # Use chunksize to avoid loading the entire 200MB+ file into memory
    chunksize = 10000
    
    # A dictionary to store samples of parsed data for demonstration
    parsed_samples = {key: [] for key in PATTERNS.keys()}
    counts = {key: 0 for key in PATTERNS.keys()}
    
    # Read first chunk just to demonstrate the parsing capabilities
    for chunk in pd.read_csv(csv_file, usecols=['log_name', 'value'], chunksize=chunksize):
        for _, row in chunk.iterrows():
            log_name = str(row['log_name'])
            log_string = str(row['value'])
            
            # Stop collecting if we have enough samples for all types
            if counts.get(log_name, 0) < 2:
                extracted = extract_features(log_name, log_string)
                if extracted:
                    parsed_samples[log_name].append(extracted)
                    counts[log_name] += 1
        
        # Break early for demonstration purposes once we have samples
        if all(count >= 2 for count in counts.values()):
            break

    # Print results
    for log_name, samples in parsed_samples.items():
        if samples:
            print(f"--- Extracted Metrics for {log_name} ---")
            df_samples = pd.DataFrame(samples)
            print(df_samples.to_string(index=False))
            print()

if __name__ == "__main__":
    main()
