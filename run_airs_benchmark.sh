#!/usr/bin/env bash
# =============================================================================
#  run_airs_benchmark.sh
#  AIRS Benchmarking Pipeline — OTel Demo (Docker Compose, No Kubernetes)
#
#  Architecture:
#    • Spins up opentelemetry-demo via Docker Compose + observability stack
#    • Injects faults by editing demo.flagd.json with `sed` (no browser, no K8s)
#    • Waits 300 s for Prometheus to accumulate anomalous telemetry
#    • Invokes live_harness.py --no-k8s against http://localhost:9090
#    • Restores each feature flag to "off" when the scenario completes
#
#  Usage:
#    chmod +x run_airs_benchmark.sh
#    ./run_airs_benchmark.sh [OPTIONS]
#
#  Options:
#    -s, --scenario FAULT_ID   Run only the named scenario (e.g. cartFailure)
#    -w, --wait SECONDS        Override the 300 s metric-accumulation window
#    -p, --prometheus-url URL  Override Prometheus URL (default: http://localhost:9090)
#    -n, --no-spinup           Skip `docker compose up` (stack already running)
#    -d, --dry-run             Print sed commands without executing them
#    -h, --help                Show this help text
#
#  Dependencies:
#    docker, docker compose (v2), python3, jq, sed, curl
# =============================================================================

set -Eeuo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
YEL='\033[1;33m'
GRN='\033[0;32m'
BLU='\033[0;34m'
CYN='\033[0;36m'
BLD='\033[1m'
RST='\033[0m'

info()    { echo -e "${BLU}[INFO ]${RST} $*"; }
success() { echo -e "${GRN}[OK   ]${RST} $*"; }
warn()    { echo -e "${YEL}[WARN ]${RST} $*"; }
error()   { echo -e "${RED}[ERROR]${RST} $*" >&2; }
banner()  { echo -e "\n${BLD}${CYN}══════════════════════════════════════════════════════════════${RST}"; \
            echo -e "${BLD}${CYN}  $*${RST}"; \
            echo -e "${BLD}${CYN}══════════════════════════════════════════════════════════════${RST}\n"; }

# ── Defaults ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OTEL_DEMO_DIR="${SCRIPT_DIR}/opentelemetry-demo"
FLAGD_JSON="${OTEL_DEMO_DIR}/src/flagd/demo.flagd.json"
SCENARIOS_FILE="${SCRIPT_DIR}/otel_scenarios.json"
HARNESS="${SCRIPT_DIR}/live_harness.py"
RESULTS_FILE="${SCRIPT_DIR}/benchmark_results.json"

PROMETHEUS_URL="http://localhost:9090"
WAIT_SECONDS=300
SPINUP=true
DRY_RUN=false
ONLY_SCENARIO=""   # empty = run all

# Determine which python command to use. If default python3 lacks pydantic,
# check if the 'genai' conda environment is available.
PYTHON_CMD="python3"
if ! python3 -c "import pydantic" &>/dev/null; then
  if command -v conda &>/dev/null && conda env list | grep -q -E '\bgenai\b'; then
    PYTHON_CMD="conda run --no-capture-output -n genai python3"
  fi
fi


# ── CLI Argument Parsing ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--scenario)      ONLY_SCENARIO="$2"; shift 2 ;;
    -w|--wait)          WAIT_SECONDS="$2";  shift 2 ;;
    -p|--prometheus-url) PROMETHEUS_URL="$2"; shift 2 ;;
    -n|--no-spinup)     SPINUP=false; shift ;;
    -d|--dry-run)       DRY_RUN=true; shift ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | grep -v '#!/' | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) error "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Sanity Checks ────────────────────────────────────────────────────────────
banner "AIRS Benchmark Pipeline — Pre-flight Checks"

for cmd in docker python3 sed curl; do
  if ! command -v "$cmd" &>/dev/null; then
    error "Required command not found: $cmd"
    exit 1
  fi
done
success "All required tools present: docker, python3, sed, curl"

if [[ ! -f "${FLAGD_JSON}" ]]; then
  error "flagd JSON not found at: ${FLAGD_JSON}"
  error "Make sure the opentelemetry-demo repo is cloned inside ${SCRIPT_DIR}"
  exit 1
fi
success "flagd config found: ${FLAGD_JSON}"

if [[ ! -f "${SCENARIOS_FILE}" ]]; then
  error "Scenario manifest not found: ${SCENARIOS_FILE}"
  exit 1
fi
success "Scenario manifest found: ${SCENARIOS_FILE}"

if [[ ! -f "${HARNESS}" ]]; then
  error "live_harness.py not found at: ${HARNESS}"
  exit 1
fi
success "Harness found: ${HARNESS}"

# Backup the original flagd config once — restored on EXIT
FLAGD_BACKUP="${FLAGD_JSON}.bak"
cp "${FLAGD_JSON}" "${FLAGD_BACKUP}"
info "Created flagd backup: ${FLAGD_BACKUP}"

# ── Cleanup Trap ─────────────────────────────────────────────────────────────
# Restore the original flagd file no matter how the script exits.
cleanup() {
  local exit_code=$?
  echo ""
  warn "🧹  Cleanup triggered (exit code: ${exit_code})"
  if [[ -f "${FLAGD_BACKUP}" ]]; then
    cp "${FLAGD_BACKUP}" "${FLAGD_JSON}"
    success "Restored original flagd config from backup."
    rm -f "${FLAGD_BACKUP}"
  fi
  # Give flagd a moment to hot-reload the restored config
  sleep 3
  info "Cleanup complete."
}
trap cleanup EXIT

# ── Helper: Toggle a Standard On/Off Flag ────────────────────────────────────
# Usage: toggle_flag <flag_key> <on|off>
# Handles flags whose variants are { "on": true/any, "off": false/0 }
toggle_flag() {
  local flag_key="$1"
  local state="$2"    # "on" or "off"

  if [[ "${DRY_RUN}" == "true" ]]; then
    warn "[DRY-RUN] Would set '${flag_key}' → '${state}'"
    return 0
  fi

  # The sed pattern targets the defaultVariant field within the block that
  # starts with the flag key. This is identical to the approach mandated in
  # the architecture spec, extended to handle both directions.
  if [[ "${state}" == "on" ]]; then
    sed -i "/\"${flag_key}\": {/,/\"defaultVariant\":/ \
      s/\"defaultVariant\": \"off\"/\"defaultVariant\": \"on\"/" \
      "${FLAGD_JSON}"
  else
    sed -i "/\"${flag_key}\": {/,/\"defaultVariant\":/ \
      s/\"defaultVariant\": \"on\"/\"defaultVariant\": \"off\"/" \
      "${FLAGD_JSON}"
  fi

  # Verify the write landed
  local actual
  actual=$(${PYTHON_CMD} -c "
import json, sys
with open('${FLAGD_JSON}') as f:
    d = json.load(f)
flag = d['flags'].get('${flag_key}', {})
print(flag.get('defaultVariant','MISSING'))
")
  if [[ "${actual}" == "${state}" ]]; then
    success "Flag '${flag_key}' → '${state}' (verified)"
  else
    warn "Flag '${flag_key}': expected '${state}', got '${actual}'. Check for multi-variant flag."
  fi
}

# ── Helper: Toggle paymentFailure (multi-variant flag) ───────────────────────
toggle_payment_failure() {
  local state="$1"   # "100%" or "off"

  if [[ "${DRY_RUN}" == "true" ]]; then
    warn "[DRY-RUN] Would set 'paymentFailure' → '${state}'"
    return 0
  fi

  ${PYTHON_CMD} - <<PYEOF
import json

flagd_path = "${FLAGD_JSON}"
target_state = "${state}"

with open(flagd_path) as f:
    data = json.load(f)

data["flags"]["paymentFailure"]["defaultVariant"] = target_state

with open(flagd_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"paymentFailure.defaultVariant set to '{target_state}'")
PYEOF
}

# ── Helper: Wait for Prometheus to be ready ──────────────────────────────────
wait_for_prometheus() {
  info "Waiting for Prometheus at ${PROMETHEUS_URL} to become ready..."
  local attempts=0
  local max_attempts=40  # 40 × 15 s = 10 minutes max wait

  while [[ ${attempts} -lt ${max_attempts} ]]; do
    if curl -sf "${PROMETHEUS_URL}/-/ready" &>/dev/null; then
      success "Prometheus is ready."
      return 0
    fi
    attempts=$(( attempts + 1 ))
    info "  [${attempts}/${max_attempts}] Not ready yet, retrying in 15 s..."
    sleep 15
  done

  error "Prometheus did not become ready within $((max_attempts * 15)) seconds."
  return 1
}

# ── Helper: Wait for core OTel Demo services to be healthy ───────────────────
wait_for_demo_health() {
  info "Waiting for OTel Demo frontend-proxy to be reachable..."
  local attempts=0
  local max_attempts=48  # 48 × 10 s = 8 minutes

  while [[ ${attempts} -lt ${max_attempts} ]]; do
    # The frontend-proxy typically binds on port 8080
    if curl -sf --max-time 5 "http://localhost:8080/" &>/dev/null; then
      success "OTel Demo frontend is reachable."
      return 0
    fi
    attempts=$(( attempts + 1 ))
    info "  [${attempts}/${max_attempts}] Not ready yet, retrying in 10 s..."
    sleep 10
  done

  warn "Frontend did not become reachable — stack may still be initialising."
  warn "Proceeding anyway; some metrics may not be available yet."
}

# ── Step 1: Spin Up the Stack ─────────────────────────────────────────────────
if [[ "${SPINUP}" == "true" ]]; then
  banner "Step 1 — Spinning Up OTel Demo + Observability Stack"

  info "Changing to OTel Demo directory: ${OTEL_DEMO_DIR}"
  pushd "${OTEL_DEMO_DIR}" > /dev/null

  info "Running: docker compose -f compose.yaml -f compose.observability.yaml up --no-build --detach"
  if [[ "${DRY_RUN}" == "false" ]]; then
    docker compose -f compose.yaml -f compose.observability.yaml up --no-build --detach
    success "Docker Compose stack launched."
  else
    warn "[DRY-RUN] Skipping docker compose up"
  fi

  popd > /dev/null

  wait_for_demo_health
  wait_for_prometheus

  # Allow the load generator to produce an initial baseline in Prometheus before
  # any fault is injected (short 60 s head-start so Z-Score has a baseline mean).
  info "Allowing 60 s baseline accumulation before fault injection..."
  sleep 60
else
  banner "Step 1 — Skipping Stack Spinup (--no-spinup mode)"
  info "Assuming OTel Demo is already running."
  wait_for_prometheus
fi

# ── Step 2–5: Scenario Loop ───────────────────────────────────────────────────
banner "Step 2 — Loading Scenarios"

mapfile -t ALL_SCENARIOS < <(${PYTHON_CMD} -c "import json; [print(json.dumps(s)) for s in json.load(open('${SCENARIOS_FILE}'))]")
info "Found ${#ALL_SCENARIOS[@]} scenario(s) in ${SCENARIOS_FILE}"

RESULTS=()
PASS=0
FAIL=0

for scenario_json in "${ALL_SCENARIOS[@]}"; do
  fault_id=$(echo "${scenario_json}" | ${PYTHON_CMD} -c "import sys, json; print(json.load(sys.stdin).get('fault_id', ''))")
  flag_key=$(echo "${scenario_json}" | ${PYTHON_CMD} -c "import sys, json; print(json.load(sys.stdin).get('feature_flag_key', ''))")
  description=$(echo "${scenario_json}" | ${PYTHON_CMD} -c "import sys, json; print(json.load(sys.stdin).get('description', ''))")
  expected_rc=$(echo "${scenario_json}" | ${PYTHON_CMD} -c "import sys, json; print(json.load(sys.stdin).get('expected_root_cause', ''))")
  variant_on=$(echo "${scenario_json}" | ${PYTHON_CMD} -c "import sys, json; val = json.load(sys.stdin).get('variant_on'); print(val if val is not None else 'on')")

  # Filter to a single scenario if --scenario was specified
  if [[ -n "${ONLY_SCENARIO}" && "${fault_id}" != "${ONLY_SCENARIO}" ]]; then
    continue
  fi

  banner "Scenario: ${fault_id}"
  echo -e "${BLD}Description:${RST} ${description}"
  echo -e "${BLD}Expected RC: ${RST} ${expected_rc}"
  echo ""

  # ── Ensure all flags are OFF before starting ──────────────────────────────
  info "Ensuring all flags are in baseline 'off' state..."
  cp "${FLAGD_BACKUP}" "${FLAGD_JSON}"
  sleep 2  # give flagd hot-reload time

  # ── Step 2: Inject Fault ─────────────────────────────────────────────────
  info "Step 2: Injecting fault → '${fault_id}' (flag: ${flag_key}, variant: ${variant_on})"

  if [[ "${flag_key}" == "paymentFailure" ]]; then
    toggle_payment_failure "${variant_on}"
  else
    toggle_flag "${flag_key}" "${variant_on}"
  fi

  # ── Step 3: Wait for Metric Spike ────────────────────────────────────────
  info "Step 3: Waiting ${WAIT_SECONDS} s for Prometheus to capture anomalous telemetry..."
  echo -e "  ${YEL}⏳  $(date '+%H:%M:%S') — fault is now ACTIVE. Will analyse at $(date -d "+${WAIT_SECONDS} seconds" '+%H:%M:%S')${RST}"

  if [[ "${DRY_RUN}" == "false" ]]; then
    # Countdown with a progress indicator every 30 s
    elapsed=0
    while [[ ${elapsed} -lt ${WAIT_SECONDS} ]]; do
      remaining=$(( WAIT_SECONDS - elapsed ))
      printf "  ⏳  %3d s remaining...\r" "${remaining}"
      sleep_chunk=30
      if [[ ${remaining} -lt ${sleep_chunk} ]]; then
        sleep_chunk=${remaining}
      fi
      sleep "${sleep_chunk}"
      elapsed=$(( elapsed + sleep_chunk ))
    done
    echo ""
  else
    warn "[DRY-RUN] Skipping ${WAIT_SECONDS} s wait"
  fi

  # ── Step 4: Invoke live_harness.py ────────────────────────────────────────
  info "Step 4: Running AIRS via live_harness.py..."
  START_TS=$(date +%s)

  set +e
  ${PYTHON_CMD} "${HARNESS}" \
    --prometheus-url "${PROMETHEUS_URL}" \
    --zscore-mode    benchmark \
    --no-k8s \
    --fault-id       "${fault_id}" \
    --output-json    "${SCRIPT_DIR}/.airs_single_run.json"
  HARNESS_EXIT=$?
  set -e

  END_TS=$(date +%s)
  DURATION=$(( END_TS - START_TS ))

  if [[ ${HARNESS_EXIT} -eq 0 ]]; then
    AIRS_STATUS="success"
    success "AIRS analysis completed in ${DURATION} s."
  else
    AIRS_STATUS="harness_error"
    warn "live_harness.py exited with code ${HARNESS_EXIT}. Marking scenario as FAIL."
  fi

  # ── Step 5: Restore Flag to OFF ──────────────────────────────────────────
  info "Step 5: Restoring flag '${flag_key}' → off..."
  if [[ "${flag_key}" == "paymentFailure" ]]; then
    toggle_payment_failure "off"
  else
    toggle_flag "${flag_key}" "off"
  fi
  success "Flag restored. Sleeping 10 s for flagd hot-reload to propagate..."
  sleep 10

  # ── Collect Result ────────────────────────────────────────────────────────
  airs_log="N/A"
  diagnosis_correct=false
  if [[ -f "${SCRIPT_DIR}/.airs_single_run.json" ]]; then
    airs_log=$(${PYTHON_CMD} -c "import json; d = json.load(open('${SCRIPT_DIR}/.airs_single_run.json')); print(d.get('chain_of_thought_log', 'N/A'))" 2>/dev/null || echo "N/A")
    diagnosis_correct=$(${PYTHON_CMD} -c "import json; d = json.load(open('${SCRIPT_DIR}/.airs_single_run.json')); print(str(d.get('diagnosis_correct', False)).lower())" 2>/dev/null || echo "false")
    rm -f "${SCRIPT_DIR}/.airs_single_run.json"
  fi

  if [[ "${diagnosis_correct}" == "true" ]]; then
    echo -e "${GRN}${BLD}  ✅  PASS — ${fault_id} (${DURATION} s)${RST}"
    PASS=$(( PASS + 1 ))
  else
    echo -e "${RED}${BLD}  ❌  FAIL — ${fault_id} (${DURATION} s, harness_status=${AIRS_STATUS})${RST}"
    FAIL=$(( FAIL + 1 ))
  fi

  RESULTS+=("{\"fault_id\":\"${fault_id}\",\"diagnosis_correct\":${diagnosis_correct},\"time_to_resolve_seconds\":${DURATION},\"harness_status\":\"${AIRS_STATUS}\",\"airs_log\":$(echo "${airs_log}" | ${PYTHON_CMD} -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}")

  # Brief inter-scenario cooldown so metrics settle back to baseline
  if [[ -n "${ONLY_SCENARIO}" ]]; then
    info "Single-scenario mode: no inter-scenario cooldown needed."
  else
    info "Inter-scenario cooldown: sleeping 60 s for metrics to settle..."
    sleep 60
  fi
done

# ── Final Report ──────────────────────────────────────────────────────────────
banner "Benchmark Complete"

TOTAL=$(( PASS + FAIL ))
echo -e "${BLD}Results: ${GRN}${PASS} PASS${RST} / ${RED}${FAIL} FAIL${RST} / ${TOTAL} Total${RST}"

# Write JSON results file
RESULTS_JSON="[$(IFS=,; echo "${RESULTS[*]}")]"
echo "${RESULTS_JSON}" | ${PYTHON_CMD} -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read()), indent=2))" \
  > "${RESULTS_FILE}"

success "Results saved to: ${RESULTS_FILE}"
echo ""
info "Tip: View with:  cat ${RESULTS_FILE} | ${PYTHON_CMD} -m json.tool"
