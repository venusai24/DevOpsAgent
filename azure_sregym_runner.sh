#!/bin/bash
set -euo pipefail

# Configuration
RG_NAME="sregym-experiment-rg"
LOCATION="koreacentral" # Changed to an allowed region
AKS_NAME="sregym-target-cluster"
VM_NAME="sregym-telemetry-vm"

# ==========================================
# FAIL-SAFE CLEANUP TRAP
# ==========================================
# This function runs automatically whenever the script exits.
# Whether it finishes successfully, fails on an error, or you press Ctrl+C, 
# it will delete the Resource Group to prevent wasting Azure credits.
cleanup() {
    echo "=========================================================="
    echo "⚠️ CLEANUP TRIGGERED: Deleting Resource Group '$RG_NAME'..."
    echo "=========================================================="
    # --no-wait returns immediately while Azure deletes in the background
    az group delete --name "$RG_NAME" --yes --no-wait || true
    echo "Cleanup initiated. Check the Azure Portal to confirm deletion."
}
# trap cleanup EXIT
echo "⚠️ Auto-cleanup is DISABLED. Remember to manually delete the resource group when done: az group delete -n $RG_NAME -y"
echo "Starting SREGym Azure Provisioning..."

echo "1. Creating Resource Group: $RG_NAME in $LOCATION"
az group create --name "$RG_NAME" --location "$LOCATION"

echo "2. Creating Telemetry VM ($VM_NAME) with Docker, Prometheus & Jaeger..."
# Cloud-init script to install docker and start observability containers on boot
cat << 'EOF' > cloud-init-telemetry.txt
#cloud-config
package_upgrade: true
packages:
  - docker.io
runcmd:
  - systemctl enable docker
  - systemctl start docker
  # Start Prometheus with remote-write enabled
  - docker run -d -p 9090:9090 --name prometheus prom/prometheus --web.enable-remote-write-receiver
  # Start Jaeger all-in-one with OTLP enabled
  - docker run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one:latest
EOF

az vm create \
  --resource-group "$RG_NAME" \
  --name "$VM_NAME" \
  --image Ubuntu2204 \
  --size Standard_D2s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --custom-data cloud-init-telemetry.txt \
  --public-ip-sku Standard

# Get the Telemetry VM's public IP
TELEMETRY_IP=$(az vm show -d -g "$RG_NAME" -n "$VM_NAME" --query publicIps -o tsv)
echo "Telemetry VM created with IP: $TELEMETRY_IP"
echo "Prometheus UI: http://$TELEMETRY_IP:9090"
echo "Jaeger UI: http://$TELEMETRY_IP:16686"

echo "3. Creating Target AKS Cluster ($AKS_NAME)..."
# Using Standard_D4s_v5 (4 vCPUs, 16GB RAM) to save credits compared to massive servers
az aks create \
  --resource-group "$RG_NAME" \
  --name "$AKS_NAME" \
  --node-count 2 \
  --node-vm-size Standard_B2as_v2 \
  --generate-ssh-keys \
  --enable-managed-identity

echo "4. Getting AKS Credentials..."
# This configures your local kubectl to point to the new AKS cluster
az aks get-credentials --resource-group "$RG_NAME" --name "$AKS_NAME" --overwrite-existing

echo "=========================================================="
echo "Infrastructure is Ready!"
echo "Your kubectl context is now set to $AKS_NAME."
echo "Telemetry Endpoint (Prometheus Remote Write): http://$TELEMETRY_IP:9090/api/v1/write"
echo "=========================================================="

# ---------------------------------------------------------
# SREGYM WORKLOAD EXECUTION
# ---------------------------------------------------------
echo "5. Executing live_harness.py workload..."

# IMPORTANT: 
# 1. You must update your telemetry agents in K8s to push to http://$TELEMETRY_IP:9090/api/v1/write
# 2. You should scale down the tput so the smaller cluster doesn't crash instantly.

python3 live_harness.py --duration 600s --tput 500 --multiplier 1 --telemetry-endpoint http://$TELEMETRY_IP:9090/api/v1/write --prometheus-url http://$TELEMETRY_IP:9090 --zscore-mode benchmark


echo "Experiment workflow completed."
# The cleanup trap will automatically fire after this line.
