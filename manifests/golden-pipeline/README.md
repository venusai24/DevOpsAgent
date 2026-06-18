# Golden Blueprint — Deployment Guide

## Autonomous SRE Log Collection Pipeline for Kubernetes

---

## Prerequisites

| Component | Minimum Version | Notes |
|:----------|:---------------|:------|
| Kubernetes | 1.28+ | Required for `scheduling.k8s.io/v1` PriorityClass |
| ClickHouse | 26.2+ | Required for GA `full_text` inverted indexes |
| Helm/Kustomize | 5.0+ / 5.3+ | For deployment orchestration |
| Prometheus Operator | 0.70+ | For ServiceMonitor/PrometheusRule CRDs |
| kubectl | 1.28+ | For `kubectl apply -k` |

## Architecture Overview

```
Edge (DaemonSet)      →  Aggregator (StatefulSet)  →  ClickHouse (Tiered)
┌─────────────────┐      ┌──────────────────────┐     ┌──────────────┐
│ Vector Edge     │─gRPC─│ Vector Aggregator    │────▶│ logs_hot     │ 7d
│ • Parse JSON    │      │ • Semantic Rescue    │────▶│ logs_warm    │ 30d
│ • Extract level │      │ • Infra Noise Gate   │────▶│ logs_cold    │ 90d
│ • Tag provenance│      │ • Probe Noise Gate   │     └──────────────┘
│ • ZSTD compress │      │ • Multiline Assembly │
└─────────────────┘      │ • Tiered Routing     │     ┌──────────────┐
                         └──────────────────────┘     │ k8s_events   │ 30d
OTel Events (HA)  ──────────────────────────────────▶│              │
└─ LeaderElection                                     └──────────────┘
```

## Quick Start

### 1. Deploy the Pipeline

```bash
# Preview what will be created
kubectl kustomize manifests/golden-pipeline/

# Deploy everything
kubectl apply -k manifests/golden-pipeline/
```

### 2. Initialize ClickHouse Schema

```bash
# Port-forward to ClickHouse (adjust service name for your deployment)
kubectl port-forward svc/clickhouse -n observability 8123:8123 &

# Apply the schema
cat manifests/golden-pipeline/07-clickhouse-schema.sql | \
  curl -s 'http://localhost:8123/' --data-binary @-

# Or via clickhouse-client
cat manifests/golden-pipeline/07-clickhouse-schema.sql | \
  kubectl exec -i -n observability deploy/clickhouse -- \
  clickhouse-client --multiquery
```

> ⚠️ **Important**: Change the `airs_agent` user password in
> `07-clickhouse-schema.sql` before deploying to production!

### 3. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n observability -l app.kubernetes.io/part-of=golden-pipeline

# Verify Vector Edge DaemonSet is on all nodes
kubectl get ds vector-edge -n observability

# Verify Vector Aggregator StatefulSet
kubectl get sts vector-aggregator -n observability

# Verify OTel Events Collector (should see 2 pods)
kubectl get deploy otel-k8s-events -n observability

# Check Vector Edge health
kubectl logs -n observability -l app.kubernetes.io/name=vector-edge --tail=20

# Check Vector Aggregator health
kubectl logs -n observability -l app.kubernetes.io/name=vector-aggregator --tail=20

# Check OTel Events health
kubectl logs -n observability -l app.kubernetes.io/name=otel-k8s-events --tail=20
```

### 4. Verify Data Flow

```bash
# Query ClickHouse for recent hot-tier logs
kubectl exec -i -n observability deploy/clickhouse -- \
  clickhouse-client --query "
    SELECT count(), Namespace, Severity
    FROM telemetry.logs_hot
    WHERE Timestamp > now() - INTERVAL 5 MINUTE
    GROUP BY Namespace, Severity
    ORDER BY count() DESC
    LIMIT 20
  "

# Query K8s events
kubectl exec -i -n observability deploy/clickhouse -- \
  clickhouse-client --query "
    SELECT Timestamp, Namespace, ObjectName, Reason, Message
    FROM telemetry.k8s_events
    ORDER BY Timestamp DESC
    LIMIT 10
  "
```

## Configuration

### Adjusting for Cluster Size

| Cluster Size | Edge Memory | Aggregator Replicas | Aggregator Memory | Disk Buffer |
|:-------------|:-----------|:-------------------|:-----------------|:------------|
| Small (<20 nodes) | 128Mi | 2 | 1Gi | 5Gi |
| Medium (20-100 nodes) | 256Mi | 3 | 4Gi | 20Gi |
| Large (100-500 nodes) | 512Mi | 5-10 | 8Gi | 50Gi |
| XL (500+ nodes) | 1Gi | 10-20 | 16Gi | 100Gi |

### Dynamic Verbosity Override (AI Agent)

The AIRS AI agent can temporarily enable full verbosity for a specific pod:

```bash
# Enable DEBUG passthrough for a specific pod (5-minute TTL)
EXPIRY=$(date -u -d "+5 minutes" +%FT%TZ)
kubectl annotate configmap vector-aggregator-config -n observability \
  "airs.io/bypass-namespace=production" \
  "airs.io/bypass-pod=payment-service-abc123" \
  "airs.io/bypass-expires=${EXPIRY}" \
  "airs.io/bypass-reason=investigating-timeout-storm" \
  --overwrite
```

The TTL Reconciler CronJob will automatically revert after expiry.

### Storage Classes

For production, set the `storageClassName` in the Vector Aggregator
StatefulSet's `volumeClaimTemplates` to match your cluster's storage:

```yaml
storageClassName: managed-premium    # Azure AKS
storageClassName: gp3                # AWS EKS
storageClassName: standard-rwo       # GKE
```

## Manifest Index

| File | Purpose |
|:-----|:--------|
| `00-namespace-priorities.yaml` | Namespace + PriorityClasses |
| `01-rbac.yaml` | ServiceAccounts, ClusterRoles, RoleBindings |
| `02-vector-edge-config.toml` | Edge DaemonSet config (documented reference) |
| `03-vector-edge-daemonset.yaml` | Edge DaemonSet + inlined ConfigMap |
| `04-vector-aggregator-config.toml` | Aggregator config (documented reference) |
| `05-vector-aggregator-deployment.yaml` | Aggregator StatefulSet + HPA + ConfigMap |
| `06-otel-k8s-events.yaml` | OTel K8s Events Collector (HA LeaderElection) |
| `07-clickhouse-schema.sql` | ClickHouse tiered schema + query quotas |
| `08-pipeline-monitoring.yaml` | ServiceMonitors + PrometheusRules |
| `09-configmap-ttl-reconciler.yaml` | CronJob for verbosity override TTL |
| `kustomization.yaml` | Kustomize entrypoint |

## Troubleshooting

### Vector Edge not collecting logs
```bash
# Check if the source is discovering pods
kubectl exec -n observability ds/vector-edge -- vector tap kubernetes_logs
```

### Aggregator disk buffer filling up
```bash
# Check ClickHouse connectivity from aggregator
kubectl exec -n observability sts/vector-aggregator-0 -- \
  curl -s http://clickhouse.observability.svc.cluster.local:8123/ping

# Check Vector internal metrics
kubectl exec -n observability sts/vector-aggregator-0 -- \
  curl -s http://localhost:9598/metrics | grep buffer
```

### OTel Events not collecting
```bash
# Check leader election status
kubectl get lease otel-k8s-events-leader -n observability -o yaml

# Check OTel Collector logs
kubectl logs -n observability deploy/otel-k8s-events --tail=50
```
