"""Additional Tools for Stage 5-8."""

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from devops_agent.core.db.duckdb_client import DuckDBClient

from ..base import BaseTool
from .baseline_tools import _BASELINE_STORE


# 1. query_app_stats_detailed
class QueryAppStatsDetailedInput(BaseModel):
    app_stats_path: str
    tc_values: list[str]
    time_range_start: str
    time_range_end: str
    baseline_ref: str

class DetailedAppStatSeries(BaseModel):
    timestamps: list[str]
    z_scores: list[float]
    raw_values: list[float]

class QueryAppStatsDetailedOutput(BaseModel):
    detailed_series: dict[str, dict[str, DetailedAppStatSeries]] = Field(..., description="{tc -> {metric -> DetailedAppStatSeries}}")

class QueryAppStatsDetailedTool(BaseTool[QueryAppStatsDetailedInput, QueryAppStatsDetailedOutput]):
    """Fetches detailed temporal progression of app stat anomalies using DuckDB."""
    
    @property
    def name(self) -> str:
        return "query_app_stats_detailed"
        
    def _execute(self, params: QueryAppStatsDetailedInput) -> QueryAppStatsDetailedOutput:
        try:
            t_start = pd.to_datetime(params.time_range_start).timestamp()
            t_end = pd.to_datetime(params.time_range_end).timestamp()
        except ValueError:
            t_start = float(params.time_range_start)
            t_end = float(params.time_range_end)

        placeholders = ','.join(['?'] * len(params.tc_values))
        sql = f"""
            SELECT * FROM read_csv_auto('{params.app_stats_path}')
            WHERE timestamp >= ? AND timestamp <= ?
            AND tc IN ({placeholders})
            ORDER BY timestamp ASC
        """
        db = DuckDBClient.get_instance()
        try:
            df = db.query(sql, (t_start, t_end) + tuple(params.tc_values))
        except Exception:
            return QueryAppStatsDetailedOutput(detailed_series={})

        baseline_data = _BASELINE_STORE.get(params.baseline_ref, {}).get("app_metrics", {})
        detailed_series = {}
        metrics = ['rr', 'sr', 'cnt', 'mrt']

        for tc, group in df.groupby('tc'):
            if tc not in baseline_data:
                continue
                
            tc_baseline = baseline_data[tc]
            detailed_series[tc] = {}
            timestamps = [str(datetime.fromtimestamp(ts)) for ts in group['timestamp']]
            
            for m in metrics:
                if m not in group: continue
                mean = tc_baseline[m]['mean']
                std = tc_baseline[m]['std']
                
                raw_vals = group[m].tolist()
                if std == 0:
                    z_scores = [0.0 if v == mean else 3.0 for v in raw_vals]
                else:
                    z_scores = [abs(v - mean) / std for v in raw_vals]
                    
                detailed_series[tc][m] = DetailedAppStatSeries(
                    timestamps=timestamps,
                    raw_values=raw_vals,
                    z_scores=z_scores
                )

        return QueryAppStatsDetailedOutput(detailed_series=detailed_series)

# 2. compute_metric_latency_correlation

class MrtPoint(BaseModel):
    """A single timestamped MRT sample."""
    timestamp: str = Field(..., description="ISO-8601 or Unix-epoch timestamp string for this MRT sample.")
    value: float = Field(..., description="Mean response time value at this timestamp.")

class ComputeMetricLatencyCorrelationInput(BaseModel):
    metrics_path: str
    cmdb_ids: list[str]
    mrt_series: list[MrtPoint] = Field(
        ...,
        description=(
            "Mean response time series to correlate against. "
            "Each element carries an explicit timestamp so the tool can perform "
            "temporal alignment regardless of sampling frequency differences "
            "(e.g. 5 s MRT vs 1 min infrastructure KPI)."
        ),
    )
    time_range_start: str
    time_range_end: str

class CorrelationResult(BaseModel):
    pearson_r: float
    cross_correlation_lag: int
    aligned_points: int = Field(default=0, description="Number of time-aligned samples used in the computation.")
    common_freq_seconds: float = Field(default=0.0, description="Common resampling frequency (seconds) chosen for alignment.")

class ComputeMetricLatencyCorrelationOutput(BaseModel):
    correlations: dict[str, dict[str, CorrelationResult]] = Field(..., description="{cmdb_id -> {kpi -> CorrelationResult}}")


def _build_time_indexed_series(timestamps: list, values: list, name: str) -> pd.Series:
    """Convert parallel timestamp/value lists into a DatetimeIndex pd.Series.

    Accepts either ISO-8601 strings or numeric Unix-epoch seconds.
    """
    idx = pd.to_datetime(timestamps, unit="s", errors="coerce")
    # If coerce produced all-NaT the input was ISO strings, not epoch floats.
    if idx.isna().all():
        idx = pd.to_datetime(timestamps, errors="coerce")
    s = pd.Series(values, index=idx, name=name, dtype=float)
    # Drop rows whose index could not be parsed.
    return s.loc[s.index.notna()].sort_index()


def _detect_median_freq_seconds(series: pd.Series) -> float:
    """Return the median interval (in seconds) between consecutive index timestamps."""
    if len(series) < 2:
        return 0.0
    diffs = series.index.to_series().diff().dropna().dt.total_seconds()
    return float(diffs.median())


def _align_series(
    s_mrt: pd.Series,
    s_metric: pd.Series,
) -> tuple[pd.Series, pd.Series, float]:
    """Align two time-indexed Series to a common frequency.

    Strategy
    --------
    1. Detect the median sampling interval of each series.
    2. Resample *both* to the **coarser** frequency (max of the two intervals)
       using the mean aggregator so no synthetic points are introduced.
    3. Inner-join on the resulting DatetimeIndex so only overlapping windows
       are retained.
    4. Drop any remaining NaNs produced by sparse data.

    Returns
    -------
    (aligned_mrt, aligned_metric, common_freq_seconds)
    """
    freq_mrt = _detect_median_freq_seconds(s_mrt)
    freq_metric = _detect_median_freq_seconds(s_metric)

    # Fall back to 60 s when detection fails (< 2 points).
    if freq_mrt <= 0:
        freq_mrt = 60.0
    if freq_metric <= 0:
        freq_metric = 60.0

    # Choose the coarser bucket to avoid inventing data points.
    common_freq_s = max(freq_mrt, freq_metric)
    freq_str = f"{int(common_freq_s)}s"

    r_mrt = s_mrt.resample(freq_str).mean()
    r_metric = s_metric.resample(freq_str).mean()

    # Inner join → only time buckets present in both.
    aligned = pd.concat([r_mrt.rename("mrt"), r_metric.rename("metric")], axis=1, join="inner")
    aligned = aligned.dropna()

    return aligned["mrt"], aligned["metric"], common_freq_s


class ComputeMetricLatencyCorrelationTool(BaseTool[ComputeMetricLatencyCorrelationInput, ComputeMetricLatencyCorrelationOutput]):
    """Computes Pearson r and cross-correlation lag for the Stage 5.1b fallback path.

    Temporal alignment
    ------------------
    Instead of blindly truncating both series to the same length (which silently
    destroys alignment when the two signals have different sampling frequencies or
    sparse/missing data), the tool:

    1. Builds a DatetimeIndex pd.Series for each signal.
    2. Resamples both to the coarser of the two detected frequencies.
    3. Inner-joins on the resulting DatetimeIndex so only overlapping windows
       are used.
    4. Computes Pearson r and lag on the aligned, NaN-free pairs.
    """

    @property
    def name(self) -> str:
        return "compute_metric_latency_correlation"

    def _execute(self, params: ComputeMetricLatencyCorrelationInput) -> ComputeMetricLatencyCorrelationOutput:
        try:
            t_start = pd.to_datetime(params.time_range_start).timestamp()
            t_end = pd.to_datetime(params.time_range_end).timestamp()
        except ValueError:
            t_start = float(params.time_range_start)
            t_end = float(params.time_range_end)

        if not params.cmdb_ids or not params.mrt_series:
            return ComputeMetricLatencyCorrelationOutput(correlations={})

        # ── Build MRT series with DatetimeIndex ──────────────────────────────
        mrt_timestamps = [p.timestamp for p in params.mrt_series]
        mrt_values = [p.value for p in params.mrt_series]
        s_mrt = _build_time_indexed_series(mrt_timestamps, mrt_values, "mrt")

        if s_mrt.empty:
            return ComputeMetricLatencyCorrelationOutput(correlations={})

        # ── Fetch infra metric series from DuckDB ────────────────────────────
        placeholders = ",".join(["?"] * len(params.cmdb_ids))
        sql = f"""
            SELECT timestamp, cmdb_id, kpi_name, value
            FROM read_csv_auto('{params.metrics_path}')
            WHERE timestamp >= ? AND timestamp <= ?
              AND cmdb_id IN ({placeholders})
            ORDER BY timestamp ASC
        """
        db = DuckDBClient.get_instance()
        try:
            df = db.query(sql, (t_start, t_end) + tuple(params.cmdb_ids))
        except Exception:
            return ComputeMetricLatencyCorrelationOutput(correlations={})

        if df.empty:
            return ComputeMetricLatencyCorrelationOutput(correlations={})

        correlations: dict[str, dict[str, CorrelationResult]] = {}

        for cmdb_id, cmdb_group in df.groupby("cmdb_id"):
            correlations[cmdb_id] = {}
            for kpi_name, kpi_group in cmdb_group.groupby("kpi_name"):

                # Build the metric Series with a DatetimeIndex.
                s_metric = _build_time_indexed_series(
                    kpi_group["timestamp"].tolist(),
                    kpi_group["value"].tolist(),
                    name=kpi_name,
                )

                # ── Temporal alignment ────────────────────────────────────────
                a_mrt, a_metric, common_freq_s = _align_series(s_mrt, s_metric)

                if len(a_mrt) < 3:
                    # Insufficient overlap after alignment — skip this KPI.
                    continue

                # ── Pearson r ─────────────────────────────────────────────────
                pearson_r = a_mrt.corr(a_metric)
                if pd.isna(pearson_r):
                    pearson_r = 0.0

                # ── Cross-correlation lag ─────────────────────────────────────
                # Search window: ±3 buckets of the *coarser* frequency so the
                # lag search remains physically meaningful regardless of freq.
                max_lag = min(3, len(a_mrt) // 2)
                lags = range(-max_lag, max_lag + 1)
                corrs = [a_mrt.corr(a_metric.shift(lag)) for lag in lags]
                valid = [(c, lag) for c, lag in zip(corrs, lags) if not pd.isna(c)]
                best_lag = max(valid, key=lambda x: x[0])[1] if valid else 0

                correlations[cmdb_id][kpi_name] = CorrelationResult(
                    pearson_r=float(pearson_r),
                    cross_correlation_lag=int(best_lag),
                    aligned_points=int(len(a_mrt)),
                    common_freq_seconds=float(common_freq_s),
                )

        return ComputeMetricLatencyCorrelationOutput(correlations=correlations)
