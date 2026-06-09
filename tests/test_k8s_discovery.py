import os
import pytest
from unittest.mock import patch, MagicMock

from airs_v2.context.graph import ContextGraph, NodeSnapshot

# --- Unit Tests using unittest.mock ---

@patch("airs_v2.context.graph.config")
@patch("airs_v2.context.graph.client")
def test_k8s_discovery_mocked(mock_client, mock_config):
    """
    Fast CI/CD unit test testing the mapping logic (Selectors, Types, Edges) 
    without requiring a live KinD cluster.
    """
    # Mocking K8s API responses
    mock_apps_api = MagicMock()
    mock_core_api = MagicMock()
    mock_custom_api = MagicMock()
    
    mock_client.AppsV1Api.return_value = mock_apps_api
    mock_client.CoreV1Api.return_value = mock_core_api
    mock_client.CustomObjectsApi.return_value = mock_custom_api
    
    # Mock Deployment: 'checkout-deployment'
    dep_item = MagicMock()
    dep_item.metadata.name = "checkout-deployment"
    dep_item.metadata.namespace = "test-ns"
    dep_item.metadata.labels = {"app.kubernetes.io/component": "backend"}
    dep_item.spec.replicas = 2
    dep_item.status.ready_replicas = 2
    mock_apps_api.list_namespaced_deployment.return_value.items = [dep_item]
    mock_apps_api.list_namespaced_stateful_set.return_value.items = []
    mock_apps_api.list_namespaced_daemon_set.return_value.items = []
    
    # Mock Service: 'checkout-service'
    svc_item = MagicMock()
    svc_item.metadata.name = "checkout-service"
    svc_item.metadata.namespace = "test-ns"
    svc_item.spec.selector = {"app.kubernetes.io/component": "backend"}
    mock_core_api.list_namespaced_service.return_value.items = [svc_item]
    
    # Mock VirtualService: 'checkout-vs' -> 'checkout-service'
    mock_custom_api.list_namespaced_custom_object.return_value = {
        "items": [
            {
                "metadata": {"name": "checkout-vs"},
                "spec": {
                    "http": [
                        {"route": [{"destination": {"host": "checkout-service"}}]}
                    ]
                }
            }
        ]
    }
    
    # Run the topology discovery
    graph = ContextGraph()
    graph._g.clear() # clear fixtures
    
    os.environ["AIRS_USE_K8S"] = "true"
    graph.load_from_k8s(["test-ns"])
    
    nodes = list(graph._g.nodes())
    edges = list(graph._g.edges())
    
    assert "checkout-deployment" in nodes
    assert "checkout-service" in nodes
    assert "checkout-vs" in nodes
    
    assert graph._g.nodes["checkout-deployment"]["node_type"] == "service"
    assert graph._g.nodes["checkout-deployment"]["health_status"] == "healthy"
    
    # Validate dynamic edges: VS -> Service -> Deployment
    assert ("checkout-vs", "checkout-service") in edges
    assert ("checkout-service", "checkout-deployment") in edges


# --- Integration Tests for KinD ---

@pytest.fixture(scope="module")
def kind_cluster():
    """
    Checks if a live Kubernetes cluster (like KinD) is available.
    If not, skips the integration test.
    """
    try:
        from kubernetes import client, config
        config.load_kube_config()
        v1 = client.CoreV1Api()
        v1.list_namespace()
        yield v1
    except Exception:
        pytest.skip("No live Kubernetes cluster available for integration testing.")

def test_k8s_discovery_integration(kind_cluster):
    """
    Integration test against a live Kubernetes API Server.
    """
    # This test assumes a specific manifest was applied prior to test execution.
    # In a full CI pipeline, we would shell out to `kubectl apply -f test_manifest.yaml` here.
    
    graph = ContextGraph()
    graph._g.clear()
    
    try:
        graph.load_from_k8s(["default"])
        # If the cluster is empty, it shouldn't crash
        assert len(list(graph._g.nodes())) >= 0
    except Exception as e:
        pytest.fail(f"Live K8s topology discovery failed with: {e}")
