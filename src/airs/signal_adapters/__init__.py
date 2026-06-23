"""AIRS Signal Adapters — public exports and registry."""
from airs.signal_adapters.base import (
    FilteredSignalPayload,
    RawSignalPayload,
    SignalAdapter,
)
from airs.signal_adapters.logs import LogsSignalAdapter
from airs.signal_adapters.metrics import MetricsSignalAdapter
from airs.signal_adapters.prefilters import (
    dedup_log_patterns,
    filter_error_spans,
    filter_k8s_events,
    zscore_filter,
)
from airs.signal_adapters.traces import K8sStateSignalAdapter, TracesSignalAdapter
from airs.models.intents import SignalType


def get_adapter(signal_type: SignalType) -> SignalAdapter:
    """Return the appropriate SignalAdapter for a given signal type."""
    _registry = {
        SignalType.METRICS: MetricsSignalAdapter,
        SignalType.LOGS: LogsSignalAdapter,
        SignalType.TRACES: TracesSignalAdapter,
        SignalType.K8S_STATE: K8sStateSignalAdapter,
    }
    adapter_cls = _registry.get(signal_type)
    if adapter_cls is None:
        raise ValueError(f"No SignalAdapter registered for signal type: {signal_type}")
    return adapter_cls()


__all__ = [
    "SignalAdapter",
    "RawSignalPayload",
    "FilteredSignalPayload",
    "MetricsSignalAdapter",
    "LogsSignalAdapter",
    "TracesSignalAdapter",
    "K8sStateSignalAdapter",
    "zscore_filter",
    "dedup_log_patterns",
    "filter_error_spans",
    "filter_k8s_events",
    "get_adapter",
]
