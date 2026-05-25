"""
tests/test_guardrails.py

Tests for the expanded 4-layer guardrail system in agent/security/guardrails.py.

Covers:
  Layer 1 — Destructive pattern blocklist
  Layer 2 — kubectl RBAC verb allowlist + delete resource restriction
  Layer 3 — Protected namespace blocking
  Layer 4 — KB command grounding check (is_command_grounded)
"""

from __future__ import annotations

import pytest
from agent.security.guardrails import is_safe_command, is_command_grounded
from agent.state import KBEntry, KBRemediationStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry_with_command(command: str) -> KBEntry:
    return KBEntry(
        entry_id="kb-guard-001",
        incident_taxonomy="Errors:OOM",
        pattern_type="exact",
        error_pattern="Exit Code 137",
        affected_services=[],
        severity="P1",
        root_cause_narrative="Pod was OOMKilled.",
        remediation_steps=[
            KBRemediationStep(
                order=1,
                action="Restart deployment.",
                environment="kubectl",
                command=command,
                risk="medium",
            ),
        ],
        rollback_command="kubectl rollout undo deployment/svc -n prod",
    )


# ---------------------------------------------------------------------------
# Layer 1 — Destructive pattern blocklist
# ---------------------------------------------------------------------------

class TestDestructiveBlocklist:
    def test_rm_rf_blocked(self):
        assert is_safe_command("rm -rf /var/data") is False

    def test_drop_table_blocked(self):
        assert is_safe_command("psql -c 'DROP TABLE users;'") is False

    def test_drop_database_blocked(self):
        assert is_safe_command("psql -c 'DROP DATABASE production'") is False

    def test_delete_namespace_blocked(self):
        assert is_safe_command("kubectl delete namespace prod") is False

    def test_truncate_blocked(self):
        assert is_safe_command("psql -c 'TRUNCATE logs'") is False

    def test_redirect_devnull_blocked(self):
        assert is_safe_command("cat /etc/passwd > /dev/null") is False

    def test_wipefs_blocked(self):
        assert is_safe_command("wipefs -a /dev/sda") is False

    def test_kill_9_blocked(self):
        assert is_safe_command("kill -9 1234") is False

    def test_safe_kubectl_not_blocked(self):
        assert is_safe_command("kubectl get pods -n prod") is True


# ---------------------------------------------------------------------------
# Layer 2 — kubectl RBAC verb allowlist
# ---------------------------------------------------------------------------

class TestKubectlVerbAllowlist:
    def test_get_allowed(self):
        assert is_safe_command("kubectl get pods -n prod") is True

    def test_describe_allowed(self):
        assert is_safe_command("kubectl describe deployment payment-service -n prod") is True

    def test_rollout_restart_allowed(self):
        assert is_safe_command("kubectl rollout restart deployment/payment-service -n prod") is True

    def test_rollout_undo_allowed(self):
        assert is_safe_command("kubectl rollout undo deployment/payment-service -n prod") is True

    def test_scale_allowed(self):
        assert is_safe_command("kubectl scale deployment/svc -n prod --replicas=3") is True

    def test_set_resources_allowed(self):
        assert is_safe_command("kubectl set resources deployment/svc -n prod --limits=memory=2Gi") is True

    def test_patch_allowed(self):
        assert is_safe_command("kubectl patch deployment/svc -n prod --type=merge -p '{}'") is True

    def test_exec_allowed(self):
        assert is_safe_command("kubectl exec -n prod deployment/svc -- psql -c 'SELECT 1'") is True

    def test_logs_allowed(self):
        assert is_safe_command("kubectl logs -n prod deployment/svc --previous") is True

    def test_top_allowed(self):
        assert is_safe_command("kubectl top pod -n prod -l app=svc") is True

    def test_cordon_allowed(self):
        assert is_safe_command("kubectl cordon node-1") is True

    def test_drain_allowed(self):
        assert is_safe_command("kubectl drain node-1 --ignore-daemonsets") is True

    # Blocked verbs
    def test_create_blocked(self):
        """'kubectl create' is not in the allowlist — unknown verbs must be blocked."""
        assert is_safe_command("kubectl create secret generic mysecret --from-literal=key=val") is False

    def test_run_blocked(self):
        assert is_safe_command("kubectl run shell --image=ubuntu -it --rm --restart=Never") is False

    def test_replace_blocked(self):
        assert is_safe_command("kubectl replace -f /tmp/malicious.yaml") is False

    def test_proxy_blocked(self):
        assert is_safe_command("kubectl proxy --port=8001 &") is False


# ---------------------------------------------------------------------------
# Layer 2b — kubectl delete resource-type restriction
# ---------------------------------------------------------------------------

class TestKubectlDeleteRestriction:
    def test_delete_pod_allowed(self):
        assert is_safe_command("kubectl delete pod payment-service-7d9f8b-xkzp2 -n prod") is True

    def test_delete_pvc_allowed(self):
        assert is_safe_command("kubectl delete pvc payment-data -n prod") is True

    def test_delete_configmap_allowed(self):
        assert is_safe_command("kubectl delete configmap payment-config -n prod") is True

    def test_delete_job_allowed(self):
        assert is_safe_command("kubectl delete job cleanup-job -n prod") is True

    def test_delete_deployment_blocked(self):
        """Deleting a Deployment is disallowed — use rollout undo instead."""
        assert is_safe_command("kubectl delete deployment payment-service -n prod") is False

    def test_delete_service_blocked(self):
        assert is_safe_command("kubectl delete service payment-svc -n prod") is False

    def test_delete_clusterrole_blocked(self):
        assert is_safe_command("kubectl delete clusterrole admin") is False


# ---------------------------------------------------------------------------
# Layer 3 — Protected namespace blocking
# ---------------------------------------------------------------------------

class TestNamespaceProtection:
    def test_get_in_kube_system_allowed(self):
        """Read-only verbs in kube-system are permitted."""
        assert is_safe_command("kubectl get pods -n kube-system") is True

    def test_describe_in_kube_system_allowed(self):
        assert is_safe_command("kubectl describe deployment coredns -n kube-system") is True

    def test_rollout_restart_in_kube_system_allowed(self):
        """rollout is in the namespace-safe verb set."""
        assert is_safe_command("kubectl rollout restart deployment/coredns -n kube-system") is True

    def test_scale_in_kube_system_allowed(self):
        """scale is in the namespace-safe verb set."""
        assert is_safe_command("kubectl scale deployment/coredns -n kube-system --replicas=4") is True

    def test_delete_pod_in_kube_system_blocked(self):
        """Deleting pods in kube-system should be blocked."""
        assert is_safe_command("kubectl delete pod coredns-abc -n kube-system") is False

    def test_exec_in_kube_public_blocked(self):
        """exec in protected namespace should be blocked."""
        assert is_safe_command("kubectl exec -n kube-public deployment/svc -- bash") is False

    def test_drain_kube_system_blocked(self):
        """drain is not in the namespace-safe verb set."""
        assert is_safe_command("kubectl drain node-1 --ignore-daemonsets -n kube-system") is False

    def test_prod_namespace_not_protected(self):
        """Non-system namespaces should not be restricted."""
        assert is_safe_command("kubectl delete pod stale-pod -n prod") is True


# ---------------------------------------------------------------------------
# Layer 4 — KB command grounding check
# ---------------------------------------------------------------------------

class TestKBGrounding:
    def test_grounded_command_passes(self):
        kb_cmd = "kubectl rollout restart deployment/{service} -n prod"
        entry = _make_entry_with_command(kb_cmd)
        # Generated command matches the KB command (with service substituted)
        assert is_command_grounded(
            "kubectl rollout restart deployment/payment-service -n prod",
            entry,
        ) is True

    def test_ungrounded_command_fails(self):
        kb_cmd = "kubectl rollout restart deployment/{service} -n prod"
        entry = _make_entry_with_command(kb_cmd)
        # Generated command uses a completely different verb
        assert is_command_grounded(
            "kubectl scale deployment/payment-service -n prod --replicas=0",
            entry,
        ) is False

    def test_no_kb_commands_returns_true(self):
        """When KB entry has no commands, grounding check passes by default."""
        entry = KBEntry(
            entry_id="kb-no-cmd-001",
            incident_taxonomy="Errors:TLS",
            pattern_type="exact",
            error_pattern="certificate has expired",
            affected_services=[],
            severity="P1",
            root_cause_narrative="TLS cert expired.",
            remediation_steps=[
                KBRemediationStep(
                    order=1,
                    action="Renew the certificate.",
                    environment="kubectl",
                    command=None,  # No command
                    risk="low",
                ),
            ],
            rollback_command="kubectl rollout undo deployment/{service} -n prod",
        )
        assert is_command_grounded("any command here", entry) is True

    def test_service_name_substitution_grounded(self):
        """A command with the runtime service name replacing {service} should be grounded."""
        kb_cmd = "kubectl rollout restart deployment/{service} -n prod"
        entry = _make_entry_with_command(kb_cmd)
        # This is the exact runtime command with service name substituted
        assert is_command_grounded(
            "kubectl rollout restart deployment/auth-service -n prod",
            entry,
        ) is True

    def test_different_verb_not_grounded(self):
        """A command with a completely different verb from the KB is not grounded."""
        kb_cmd = "kubectl rollout restart deployment/{service} -n prod"
        entry = _make_entry_with_command(kb_cmd)
        # scale is completely different from rollout restart
        assert is_command_grounded(
            "kubectl scale deployment/auth-service --replicas=0 -n prod",
            entry,
        ) is False
