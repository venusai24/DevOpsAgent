"""
agent/reasoning/graph_schema.py

Pydantic v2 type definitions for all nodes and edges in the Enterprise Knowledge Graph (EKG).

The EKG models the enterprise infrastructure as a directed graph:
  Nodes  — Services, Databases, Caches, ExternalDependencies, OnCallSchedules, HistoricalIncidents
  Edges  — DEPENDS_ON, READS_FROM, WRITES_TO, DEPLOYED_ON, OWNED_BY, HAS_INCIDENT, CORRELATES_WITH

These types are used by knowledge_graph.py to build the NetworkX / Neo4j graph, and by the
TopologyAgent and BlastRadiusEstimator to traverse and analyze the infrastructure topology.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Discriminator for EKG node categories."""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL = "external"
    ON_CALL = "on_call"
    INCIDENT = "incident"


class HealthStatus(str, Enum):
    """Live health state of an infrastructure node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ServiceTier(int, Enum):
    """Business criticality tier for blast-radius risk scoring."""
    TIER_1 = 1  # Revenue-critical, user-facing
    TIER_2 = 2  # Internal platform services
    TIER_3 = 3  # Batch, async, non-critical


class EdgeType(str, Enum):
    """Discriminator for EKG directed edge types."""
    DEPENDS_ON = "DEPENDS_ON"
    READS_FROM = "READS_FROM"
    WRITES_TO = "WRITES_TO"
    OWNED_BY = "OWNED_BY"
    HAS_INCIDENT = "HAS_INCIDENT"
    CORRELATES_WITH = "CORRELATES_WITH"


# ---------------------------------------------------------------------------
# Node Models
# ---------------------------------------------------------------------------


class BaseNode(BaseModel):
    """Common fields shared by all EKG node types."""
    name: str = Field(..., description="Unique node identifier (canonical service name).")
    node_type: NodeType
    tier: int = Field(default=3, ge=1, le=3, description="Service tier (1=critical, 2=internal, 3=batch).")
    health_status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    owner: str = Field(default="unknown", description="Owning team or individual.")
    namespace: str = Field(default="prod", description="Kubernetes namespace.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata.")

    @property
    def is_tier1(self) -> bool:
        """True if this node is business-critical (tier 1)."""
        return self.tier == 1

    @property
    def is_healthy(self) -> bool:
        """True if the node is fully operational."""
        return self.health_status == HealthStatus.HEALTHY


class ServiceNode(BaseNode):
    """Represents a microservice or application workload."""
    node_type: NodeType = NodeType.SERVICE
    replicas: int = Field(default=1, ge=0, description="Desired replica count.")
    current_replicas: int = Field(default=1, ge=0, description="Currently running replicas.")
    on_call: str = Field(default="", description="On-call engineer email for this service.")
    deployment_version: str = Field(default="unknown", description="Currently deployed image tag.")


class DatabaseNode(BaseNode):
    """Represents a database instance."""
    node_type: NodeType = NodeType.DATABASE
    engine: str = Field(default="postgresql", description="Database engine (postgresql, mysql, etc.).")
    max_connections: int = Field(default=100, description="Maximum allowed connections.")
    current_connections: int = Field(default=0, description="Live active connection count.")

    @property
    def connection_saturation(self) -> float:
        """Returns the connection pool utilisation as a 0.0–1.0 ratio."""
        if self.max_connections == 0:
            return 0.0
        return self.current_connections / self.max_connections


class CacheNode(BaseNode):
    """Represents an in-memory cache (Redis, Memcached)."""
    node_type: NodeType = NodeType.CACHE
    engine: str = Field(default="redis", description="Cache engine.")
    memory_usage_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class ExternalNode(BaseNode):
    """Represents an external third-party dependency."""
    node_type: NodeType = NodeType.EXTERNAL
    endpoint: str = Field(default="", description="Base URL or hostname of the external service.")


class OnCallScheduleNode(BaseNode):
    """Represents an on-call rotation entry."""
    node_type: NodeType = NodeType.ON_CALL
    engineer_email: str
    schedule: str = Field(default="24/7", description="Rotation schedule (e.g., '24/7', 'business_hours').")
    escalation_contact: str = Field(default="")


# ---------------------------------------------------------------------------
# Edge Models
# ---------------------------------------------------------------------------


class DependencyEdge(BaseModel):
    """Directed DEPENDS_ON edge between two EKG nodes."""
    source: str = Field(..., description="Name of the upstream (dependent) node.")
    target: str = Field(..., description="Name of the downstream (dependency) node.")
    edge_type: EdgeType = EdgeType.DEPENDS_ON
    protocol: str = Field(default="http", description="Communication protocol.")
    latency_p99_ms: float = Field(default=0.0, description="99th percentile call latency in milliseconds.")
    error_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Current error rate %.")
    is_critical_path: bool = Field(default=True, description="If False, failure here won't cascade.")


class CorrelationEdge(BaseModel):
    """CORRELATES_WITH edge mapping services that tend to fail together."""
    source: str
    target: str
    edge_type: EdgeType = EdgeType.CORRELATES_WITH
    co_failure_count: int = Field(default=0, description="Number of times both failed within same window.")
    typical_lag_seconds: float = Field(default=0.0, description="Typical delay between primary and secondary failure.")
    description: str = Field(default="", description="Human-readable correlation description.")


# ---------------------------------------------------------------------------
# Graph-level aggregates
# ---------------------------------------------------------------------------


class SubGraph(BaseModel):
    """A bounded subgraph extracted from the EKG for a specific analysis."""
    focus_service: str = Field(..., description="The service that is the analysis focus.")
    nodes: list[dict[str, Any]] = Field(default_factory=list, description="Serialized node list.")
    edges: list[dict[str, Any]] = Field(default_factory=list, description="Serialized edge list.")
    depth: int = Field(default=2, description="Traversal depth used to extract this subgraph.")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    def to_markdown(self) -> str:
        """Render the subgraph as a human-readable markdown summary for LLM consumption."""
        lines = [f"## Topology Map: `{self.focus_service}` (depth={self.depth})\n"]
        lines.append("### Nodes")
        for node in self.nodes:
            tier_str = f"Tier {node.get('tier', '?')}"
            health = node.get('health_status', 'unknown').upper()
            health_icon = {"HEALTHY": "🟢", "DEGRADED": "🟡", "CRITICAL": "🔴"}.get(health, "⚪")
            lines.append(f"- {health_icon} **{node['name']}** [{tier_str}] [{node.get('node_type', 'service')}] — {health}")
        lines.append("\n### Dependencies")
        for edge in self.edges:
            lines.append(f"- `{edge['source']}` → `{edge['target']}` (err={edge.get('error_rate_pct', 0):.1f}%)")
        return "\n".join(lines)


class BlastRadiusReport(BaseModel):
    """Impact analysis output from the BlastRadiusEstimator."""
    target_service: str
    affected_services: list[str] = Field(default_factory=list)
    tier1_services: list[str] = Field(default_factory=list, description="Tier-1 services in blast radius.")
    tier1_impact: bool = Field(default=False)
    propagation_paths: list[list[str]] = Field(default_factory=list, description="All paths from target to tier-1.")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Aggregate risk score 0.0–1.0.")
    recommendation: str = Field(default="require_approval", description="auto_execute | require_approval | block")
    on_call_contacts: list[str] = Field(default_factory=list)
    estimated_user_impact_pct: float = Field(default=0.0, description="Estimated % of users affected.")

    def to_markdown(self) -> str:
        """Render the blast radius report as markdown for LLM and operator consumption."""
        icon = "🔴" if self.risk_score >= 0.7 else ("🟡" if self.risk_score >= 0.3 else "🟢")
        lines = [
            f"## Blast Radius Report: `{self.target_service}`",
            f"**Risk Score**: {icon} {self.risk_score:.2f}/1.00",
            f"**Recommendation**: `{self.recommendation.upper()}`",
            f"**Tier-1 Impact**: {'⚠️ YES' if self.tier1_impact else '✅ NO'}",
            f"**Estimated User Impact**: {self.estimated_user_impact_pct:.0f}%",
            "",
            "### Affected Services",
        ]
        for svc in self.affected_services:
            tier1_flag = " ⚠️ TIER-1" if svc in self.tier1_services else ""
            lines.append(f"- `{svc}`{tier1_flag}")
        if self.propagation_paths:
            lines.append("\n### Failure Propagation Paths")
            for path in self.propagation_paths:
                lines.append(f"- {' → '.join(path)}")
        if self.on_call_contacts:
            lines.append(f"\n**On-call contacts**: {', '.join(self.on_call_contacts)}")
        return "\n".join(lines)
