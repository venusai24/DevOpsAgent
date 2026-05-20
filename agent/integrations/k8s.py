import logging
import os
from config import settings
try:
    from kubernetes import client, config
except ImportError:
    client = None
    config = None

logger = logging.getLogger(__name__)

def apply_kubectl_command(command: str) -> str:
    """
    Executes a Kubernetes command safely using the Python SDK.
    In a real system, we parse the raw command (e.g., 'kubectl rollout restart deployment/X -n prod')
    and map it to the corresponding SDK method.
    """
    if not client:
        return f"[Simulated execution: Kubernetes SDK not installed] {command}"
        
    try:
        # Load kube config based on environment (in-cluster or local)
        if settings.KUBECONFIG_PATH:
            config.load_kube_config(config_file=settings.KUBECONFIG_PATH)
        elif "KUBERNETES_SERVICE_HOST" in os.environ:
            config.load_incluster_config()
        else:
            return f"[Simulated execution: No KUBECONFIG present] {command}"
            
        apps_v1 = client.AppsV1Api()
        
        # Simple string matching for demo purposes
        # E.g. "kubectl rollout restart deployment/payments-service -n prod"
        if "rollout restart" in command and "deployment" in command:
            parts = command.split()
            # extract deployment name
            deploy_part = next(p for p in parts if "deployment/" in p)
            deployment_name = deploy_part.split("/")[1]
            # extract namespace
            namespace = "default"
            if "-n" in parts:
                namespace = parts[parts.index("-n") + 1]
            
            # Using patch to trigger rollout restart
            import datetime
            patch_body = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": datetime.datetime.now().isoformat()
                            }
                        }
                    }
                }
            }
            logger.info(f"Applying rollout restart to deployment {deployment_name} in {namespace}")
            apps_v1.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=patch_body)
            return f"[Success] Executed rollout restart on {deployment_name}"
        
        return f"[Simulated execution: Command not supported by SDK wrapper] {command}"

    except Exception as e:
        logger.error(f"Failed to execute Kubernetes command: {e}")
        raise
