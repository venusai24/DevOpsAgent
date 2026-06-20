import json
import random
import uuid
import argparse
from datetime import datetime, timedelta

def generate_trace_id():
    return uuid.uuid4().hex

def create_log_entry(timestamp, service, level, message, context=None):
    log = {
        "@timestamp": timestamp.isoformat() + "Z",
        "service": service,
        "level": level,
        "message": message,
    }
    if context:
        log["context"] = context
    return log

def generate_normal_traffic(start_time, end_time, service, endpoint, count):
    logs = []
    duration = (end_time - start_time).total_seconds()
    for _ in range(count):
        log_time = start_time + timedelta(seconds=random.uniform(0, duration))
        trace_id = generate_trace_id()
        logs.append(create_log_entry(
            log_time, service, "INFO", f"Successfully processed request to {endpoint}",
            {"trace_id": trace_id, "status": 200, "latency_ms": random.randint(10, 50)}
        ))
    return logs

def synthesize_valkey_memory_disruption(start_time, duration_minutes):
    """Generates Valkey (Redis fork) OOM errors."""
    logs = []
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    # Normal background cache reads
    logs.extend(generate_normal_traffic(start_time, end_time, "cache_client_service", "valkey://get", 50))
    
    # Fault Injection: Valkey OOM write failures
    fault_start = start_time + timedelta(minutes=1)
    for i in range(20):
        log_time = fault_start + timedelta(seconds=i*3)
        logs.append(create_log_entry(
            log_time, "cache_client_service", "ERROR",
            "RedisException: OOM command not allowed when used memory > 'maxmemory'",
            {"command": "SETEX", "key_pattern": "session:*", "trace_id": generate_trace_id()}
        ))
    return sorted(logs, key=lambda x: x["@timestamp"])

def synthesize_conntrack_exhaustion(start_time, duration_minutes):
    """Generates Linux kernel nf_conntrack full errors and resulting timeouts."""
    logs = []
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    # Normal traffic
    logs.extend(generate_normal_traffic(start_time, end_time, "hotel_reservation_frontend", "/api/reserve", 30))
    
    fault_start = start_time + timedelta(minutes=1)
    for i in range(15):
        log_time = fault_start + timedelta(seconds=i*4)
        trace_id = generate_trace_id()
        # Kernel log
        logs.append(create_log_entry(
            log_time, "linux_kernel", "CRITICAL",
            "kernel: nf_conntrack: table full, dropping packet",
            {"host": "worker-node-04"}
        ))
        # App log symptom
        logs.append(create_log_entry(
            log_time + timedelta(milliseconds=500), "hotel_reservation_service", "ERROR",
            "ConnectionTimeout: Failed to establish connection to upstream payment_service (EAGAIN)",
            {"trace_id": trace_id, "host": "worker-node-04"}
        ))
    return sorted(logs, key=lambda x: x["@timestamp"])

def synthesize_kafka_queue_problems(start_time, duration_minutes):
    """Generates Kafka Consumer max.poll.interval.ms exceed and rebalancing errors."""
    logs = []
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    fault_start = start_time + timedelta(minutes=1)
    for i in range(10):
        log_time = fault_start + timedelta(seconds=i*15)
        
        # High processing time warning
        logs.append(create_log_entry(
            log_time, "order_processor_consumer", "WARN",
            "Message processing took longer than max.poll.interval.ms (300000ms)",
            {"partition": random.randint(0, 5), "topic": "orders_queue"}
        ))
        
        # Commit failed exception due to group rebalance
        logs.append(create_log_entry(
            log_time + timedelta(seconds=1), "order_processor_consumer", "ERROR",
            "org.apache.kafka.clients.consumer.CommitFailedException: Commit cannot be completed since the group has already rebalanced and assigned the partitions to another member.",
            {"consumer_id": "order-consumer-group-1"}
        ))
    return sorted(logs, key=lambda x: x["@timestamp"])

def synthesize_loadgenerator_flood(start_time, duration_minutes):
    """Generates massive frontend traffic spikes causing thread exhaustion."""
    logs = []
    fault_start = start_time + timedelta(minutes=1)
    
    # The Flood
    for i in range(200): # High volume
        log_time = fault_start + timedelta(milliseconds=i*100)
        logs.append(create_log_entry(
            log_time, "frontend_ingress", "WARN",
            "HTTP 429 Too Many Requests - Rate limit exceeded for IP",
            {"client_ip": "10.0.0.45", "path": "/"}
        ))
    
    # Cascading failure: thread exhaustion
    for i in range(20):
        log_time = fault_start + timedelta(seconds=i)
        logs.append(create_log_entry(
            log_time, "frontend_app", "FATAL",
            "Worker thread pool exhausted. Unable to accept new HTTP connections.",
            {"active_threads": 500, "max_threads": 500}
        ))
    return sorted(logs, key=lambda x: x["@timestamp"])

def synthesize_ingress_misroute(start_time, duration_minutes):
    """Generates API gateway misconfiguration logging."""
    logs = []
    fault_start = start_time + timedelta(minutes=1)
    
    for i in range(25):
        log_time = fault_start + timedelta(seconds=i*2)
        trace_id = generate_trace_id()
        
        # Ingress controller log
        logs.append(create_log_entry(
            log_time, "nginx_ingress_controller", "ERROR",
            "upstream connect error or disconnect/reset before headers. reset reason: connection failure",
            {"trace_id": trace_id, "status": 502, "upstream": "invalid-backend-svc:8080"}
        ))
        # Client facing error
        logs.append(create_log_entry(
            log_time + timedelta(milliseconds=10), "nginx_ingress_controller", "WARN",
            "HTTP 404 Not Found - No route matches path",
            {"trace_id": trace_id, "path": "/api/v1/checkout/process"}
        ))
    return sorted(logs, key=lambda x: x["@timestamp"])

def synthesize_rpc_retry_storm(start_time, duration_minutes, is_capacity_driven=False):
    """Generates a gRPC DEADLINE_EXCEEDED cascade causing retry storms."""
    logs = []
    fault_start = start_time + timedelta(minutes=1)
    
    if is_capacity_driven:
        # Precede the storm with CPU containment symptoms
        for i in range(5):
            logs.append(create_log_entry(
                fault_start - timedelta(seconds=10 - i), "checkout_service", "WARN",
                f"High CPU Usage detected: {95 + i}% - Throttling enabled",
                {"cpu_quota": "limited", "gc_pause_ms": random.randint(200, 500)}
            ))

    for i in range(15):
        log_time = fault_start + timedelta(seconds=i*2)
        trace_id = generate_trace_id()
        
        # Initial RPC Timeout
        logs.append(create_log_entry(
            log_time, "frontend_rpc_client", "ERROR",
            "rpc error: code = DeadlineExceeded desc = context deadline exceeded",
            {"trace_id": trace_id, "method": "/checkout.CheckoutService/ProcessPayment", "timeout_ms": 2000}
        ))
        
        # The Retry Storm (Exponential amplification)
        for retry in range(1, 4):
            logs.append(create_log_entry(
                log_time + timedelta(milliseconds=retry * 150), "frontend_rpc_client", "WARN",
                f"Retrying request ({retry}/3) due to DeadlineExceeded",
                {"trace_id": trace_id, "method": "/checkout.CheckoutService/ProcessPayment"}
            ))
            # Backends getting slammed by retries
            logs.append(create_log_entry(
                log_time + timedelta(milliseconds=retry * 160), "payment_backend_service", "WARN",
                "Received duplicate trace_id request. Backend queue overloaded.",
                {"trace_id": trace_id, "queue_depth": random.randint(150, 300)}
            ))

    return sorted(logs, key=lambda x: x["@timestamp"])

def main():
    parser = argparse.ArgumentParser(description="SRE Mock Telemetry Synthesizer")
    parser.add_argument("--fault", required=True, choices=[
        "valkey_memory_disruption", 
        "node_conntrack_exhaustion", 
        "kafka_queue_problems", 
        "loadgenerator_flood", 
        "ingress_misroute", 
        "load_spike_rpc_retry_storm", 
        "capacity_decrease_rpc_retry_storm"
    ], help="The fault scenario to synthesize.")
    
    args = parser.parse_args()
    
    start_time = datetime.utcnow()
    duration = 5 # Generate 5 minutes of logs
    
    if args.fault == "valkey_memory_disruption":
        logs = synthesize_valkey_memory_disruption(start_time, duration)
    elif args.fault == "node_conntrack_exhaustion":
        logs = synthesize_conntrack_exhaustion(start_time, duration)
    elif args.fault == "kafka_queue_problems":
        logs = synthesize_kafka_queue_problems(start_time, duration)
    elif args.fault == "loadgenerator_flood":
        logs = synthesize_loadgenerator_flood(start_time, duration)
    elif args.fault == "ingress_misroute":
        logs = synthesize_ingress_misroute(start_time, duration)
    elif args.fault == "load_spike_rpc_retry_storm":
        logs = synthesize_rpc_retry_storm(start_time, duration, is_capacity_driven=False)
    elif args.fault == "capacity_decrease_rpc_retry_storm":
        logs = synthesize_rpc_retry_storm(start_time, duration, is_capacity_driven=True)

    # Output to file
    filename = f"{args.fault}_logs.json"
    with open(filename, "w") as f:
        json.dump(logs, f, indent=2)
        
    print(f"✅ Successfully synthesized {len(logs)} log entries for '{args.fault}'.")
    print(f"📁 Saved to: {filename}")

if __name__ == "__main__":
    main()