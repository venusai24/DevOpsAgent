.mode csv
.import cluster_app_stats.csv cluster_app_stats
.import cluster_incident_logs.csv cluster_incident_logs
.import incident_traces.csv incident_traces
.import redis02_metrics.csv redis02_metrics

SELECT DISTINCT log_name FROM cluster_incident_logs;
