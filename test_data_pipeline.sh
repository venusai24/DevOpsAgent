#!/bin/bash
set -euo pipefail

echo "=========================================================="
echo "AIRS v2: Quick Observability Pipeline Test"
echo "=========================================================="
echo "This script verifies that the Vector -> ClickHouse data pipeline"
echo "is fully functional on your cluster without running a 15-minute SREGym fault."
echo ""

# Pre-flight check
echo "1. Checking ClickHouse and Vector status..."
if ! kubectl get pods -n observability | grep -q "Running"; then
    echo "❌ Observability pods are not fully running. Please fix cluster resources first."
    kubectl get pods -n observability
    exit 1
fi
echo "✅ Observability pods are Running."

echo "2. Deploying a dummy workload to generate mock FATAL logs..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: mock-error-generator
  namespace: default
  labels:
    app: mock-error-generator
spec:
  containers:
  - name: logger
    image: busybox
    command: ["/bin/sh", "-c"]
    args:
    - "while true; do echo '{\"timestamp\":\"\$(date -Iseconds)\",\"level\":\"FATAL\",\"message\":\"Simulated core meltdown in database cluster\",\"service\":\"mock-db\"}'; sleep 2; done"
  restartPolicy: Never
EOF

echo "⏳ Waiting 15 seconds for logs to route through Vector Edge -> Aggregator -> ClickHouse..."
sleep 15

echo "3. Querying ClickHouse for the mock logs..."
# The logs should be in the logs_hot table with level 'fatal'
QUERY="SELECT timestamp, service_name, severity_text, body FROM telemetry.logs_hot WHERE service_name='mock-error-generator' AND severity_text='fatal' LIMIT 5 FORMAT PrettyCompact"

RESULTS=$(kubectl exec -n observability svc/clickhouse -- clickhouse-client -n default --query="$QUERY" 2>/dev/null || echo "")

echo ""
echo "--- ClickHouse Results ---"
if [[ -z "$RESULTS" || "$RESULTS" == *"Exception"* ]]; then
    echo "❌ No logs found in ClickHouse. The pipeline might be broken or delayed."
    echo "Check vector-aggregator logs: kubectl logs statefulset/vector-aggregator -n observability"
else
    echo "$RESULTS"
    echo "✅ SUCCESS! The data pipeline is functioning correctly."
fi
echo "--------------------------"

echo "🧹 Cleaning up mock workload..."
kubectl delete pod mock-error-generator -n default --ignore-not-found >/dev/null

echo "Test complete!"
