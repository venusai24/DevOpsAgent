import streamlit as st
from datetime import datetime, time, date
import os
import sys
import zipfile
import io

# Add the project root to PYTHONPATH so we can import src.airs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.airs.gui.telemetry_clients import get_client
from src.airs.gui.json_to_csv import convert_metrics_to_csv, convert_logs_to_csv, convert_traces_to_csv

# Configure page
st.set_page_config(page_title="Telemetry Extractor", page_icon="📡", layout="wide")

st.title("📡 Observability Telemetry Extractor")
st.markdown("Extract metrics, logs, and traces from your observability stack into unified CSV formats.")

# Directories
OUTPUT_DIR = "data/telemetry_exports"

with st.sidebar:
    st.header("Time Range")
    start_date = st.date_input("Start Date", value=date.today())
    start_time_val = st.time_input("Start Time", value=time(0, 0))
    
    end_date = st.date_input("End Date", value=date.today())
    end_time_val = st.time_input("End Time", value=time(23, 59))
    
    st.divider()
    
    st.header("Select Services")
    use_prometheus = st.checkbox("Prometheus (Metrics)", value=True)
    use_jaeger = st.checkbox("Jaeger (Traces)", value=True)
    use_tempo = st.checkbox("Tempo (Traces)", value=False)
    use_loki = st.checkbox("Loki (Logs)", value=True)
    use_opensearch = st.checkbox("OpenSearch (Logs)", value=False)

st.header("Connection Settings")
col1, col2 = st.columns(2)

urls = {}
with col1:
    if use_prometheus:
        urls['prometheus'] = st.text_input("Prometheus URL", value="http://localhost:9090")
    if use_jaeger:
        urls['jaeger'] = st.text_input("Jaeger URL", value="http://localhost:16686")
    if use_tempo:
        urls['tempo'] = st.text_input("Tempo URL", value="http://localhost:3200")

with col2:
    if use_loki:
        urls['loki'] = st.text_input("Loki URL", value="http://localhost:3100")
    if use_opensearch:
        urls['opensearch'] = st.text_input("OpenSearch URL", value="http://localhost:9200")

if st.button("Fetch and Export Data", type="primary"):
    start_dt = datetime.combine(start_date, start_time_val)
    end_dt = datetime.combine(end_date, end_time_val)
    
    if start_dt >= end_dt:
        st.error("Start time must be before end time.")
    elif not urls:
        st.warning("Please select at least one service to query.")
    else:
        with st.spinner("Fetching data from selected services..."):
            generated_files = []
            
            # Fetch Prometheus
            if use_prometheus:
                st.write("Fetching Prometheus metrics...")
                client = get_client("prometheus", urls['prometheus'])
                data = client.fetch(start_dt, end_dt)
                csv_path = convert_metrics_to_csv(data, OUTPUT_DIR, "prometheus_metrics.csv")
                if csv_path: generated_files.append(csv_path)

            # Fetch Logs (Loki & OpenSearch combined)
            logs_data = []
            if use_loki:
                st.write("Fetching Loki logs...")
                client = get_client("loki", urls['loki'])
                logs_data.extend(client.fetch(start_dt, end_dt))
            if use_opensearch:
                st.write("Fetching OpenSearch logs...")
                client = get_client("opensearch", urls['opensearch'])
                logs_data.extend(client.fetch(start_dt, end_dt))
                
            if logs_data:
                csv_path = convert_logs_to_csv(logs_data, OUTPUT_DIR, "cluster_incident_logs.csv")
                if csv_path: generated_files.append(csv_path)
                
            # Fetch Traces (Jaeger & Tempo combined)
            traces_data = []
            if use_jaeger:
                st.write("Fetching Jaeger traces...")
                client = get_client("jaeger", urls['jaeger'])
                traces_data.extend(client.fetch(start_dt, end_dt))
            if use_tempo:
                st.write("Fetching Tempo traces...")
                client = get_client("tempo", urls['tempo'])
                traces_data.extend(client.fetch(start_dt, end_dt))
                
            if traces_data:
                csv_path = convert_traces_to_csv(traces_data, OUTPUT_DIR, "incident_traces.csv")
                if csv_path: generated_files.append(csv_path)

            if generated_files:
                st.success(f"Successfully generated {len(generated_files)} CSV file(s)!")
                
                # Create a ZIP file for easy download
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for file_path in generated_files:
                        zip_file.write(file_path, os.path.basename(file_path))
                
                st.download_button(
                    label="Download All CSVs (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="telemetry_exports.zip",
                    mime="application/zip",
                )
            else:
                st.warning("No data was retrieved from the selected services.")
