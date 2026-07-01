import os
import random

import numpy as np
import pandas as pd


def inject_causal_anomalies():
    data_dir = "incident_data"
    app_csv_path = os.path.join(data_dir, "cluster_app_stats.csv")
    redis_csv_path = os.path.join(data_dir, "redis02_metrics.csv")
    logs_csv_path = os.path.join(data_dir, "cluster_incident_logs.csv")
    
    # 1. Process App Stats (Network Latency and OOM Rejection impact)
    df_app = pd.read_csv(app_csv_path)
    # Phase 1: Network Latency (1614852000 to 1614852900)
    phase1_mask = (df_app['timestamp'] >= 1614852000) & (df_app['timestamp'] <= 1614852900)
    
    def apply_phase1_mrt(row):
        t = row['timestamp'] - 1614852000
        jitter = random.uniform(0, 50)
        base_mrt = row['mrt']
        if t < 300:
            return base_mrt + (t / 300) * 500 + jitter
        elif t < 600:
            return base_mrt + 500 + ((t - 300) / 300) * 1500 + jitter
        else:
            return base_mrt + 2000 + jitter

    def apply_phase1_sr(row):
        t = row['timestamp'] - 1614852000
        if t >= 600:
            return max(0, row['sr'] - random.uniform(2, 8))
        return row['sr']
        
    df_app.loc[phase1_mask, 'mrt'] = df_app[phase1_mask].apply(apply_phase1_mrt, axis=1)
    df_app.loc[phase1_mask, 'sr'] = df_app[phase1_mask].apply(apply_phase1_sr, axis=1)

    # Phase 2: Redis OOM Rejection impact on App Stats (1614853600 to 1614853800) Stage 4
    phase2_stage4_mask = (df_app['timestamp'] >= 1614853600) & (df_app['timestamp'] <= 1614853800)
    df_app.loc[phase2_stage4_mask, 'sr'] = df_app.loc[phase2_stage4_mask, 'sr'] - np.random.uniform(10, 20, size=phase2_stage4_mask.sum())
    df_app.loc[phase2_stage4_mask, 'mrt'] = df_app.loc[phase2_stage4_mask, 'mrt'] + np.random.uniform(500, 1000, size=phase2_stage4_mask.sum())
    
    df_app.to_csv(os.path.join(data_dir, "injected_cluster_app_stats.csv"), index=False)
    print("Injected cluster_app_stats.csv")

    # 2. Process Redis Metrics (OOM Leak)
    df_redis = pd.read_csv(redis_csv_path)
    
    def apply_phase2_redis(row):
        if row['timestamp'] < 1614852900 or row['timestamp'] > 1614853800:
            return row['value']
            
        t = row['timestamp'] - 1614852900
        kpi = row['kpi_name']
        val = row['value']
        
        # Stage 1 (Memory Leak): 0 to 400s
        # Stage 2 (Maxmemory Reached): 400 to 500s
        # Stage 3 (Evictions & Latency): 500 to 700s
        # Stage 4 (Rejection & Errors): 700 to 900s
        
        if 'used_memory' in kpi or 'MEMUsedMemPerc' in kpi:
            if t < 400:
                # ramp up to 99%
                factor = (t / 400)
                if 'MEMUsedMemPerc' in kpi:
                    return val + factor * (99.0 - val)
                else:
                    return val + factor * (4e9 - val) # Assume 4GB maxmemory
            else:
                if 'MEMUsedMemPerc' in kpi:
                    return 99.0 + random.uniform(-0.5, 0.5)
                else:
                    return 4e9 + random.uniform(-1e6, 1e6)
                    
        if kpi == 'redis-Redis_6379_Redis  (evicted_keys)':
            if t >= 500:
                return val + ((t - 500) / 400) * 100000 + random.uniform(0, 100)
                
        if kpi == 'OSLinux-CPU_CPU_CPUSysTime':
            if t >= 500:
                return 0.90 + random.uniform(0.0, 0.09)
                
        if kpi == 'redis-Redis_6379_Redis  (rejected_connections)':
            if t >= 700:
                return val + ((t - 700) / 200) * 5000 + random.uniform(0, 50)
                
        return val

    df_redis['value'] = df_redis.apply(apply_phase2_redis, axis=1)
    df_redis.to_csv(os.path.join(data_dir, "injected_redis02_metrics.csv"), index=False)
    print("Injected redis02_metrics.csv")
    
    # 3. Process Logs (OOM Logs)
    df_logs = pd.read_csv(logs_csv_path)
    # Inject OOM logs between 1614853600 and 1614853800
    new_logs = []
    for ts in range(1614853600, 1614853800, 30):
        new_logs.append({
            'timestamp': ts,
            'cmdb_id': 'Redis02',
            'log_level': 'ERROR',
            'message': "OOM command not allowed when used memory > 'maxmemory'."
        })
    df_new_logs = pd.DataFrame(new_logs)
    df_logs = pd.concat([df_logs, df_new_logs]).sort_values('timestamp')
    df_logs.to_csv(os.path.join(data_dir, "injected_cluster_incident_logs.csv"), index=False)
    print("Injected cluster_incident_logs.csv")

if __name__ == "__main__":
    inject_causal_anomalies()
