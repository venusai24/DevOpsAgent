# DevOpsAgent (AIRS v2)

Welcome to the repository for DevOpsAgent, an autonomous incident response system.

## Setup Instructions

These instructions will help you set up and run the system locally on your device as quickly and easily as possible.

### Prerequisites

Make sure you have the following installed on your system:
- **Python 3.10+**
- **Git**
- **Docker** and **Docker Compose** (if you want to run the full stack including databases and workers)

### 1. Clone the Repository

First, clone the project to your local machine:

```bash
git clone https://github.com/your-username/DevOpsAgent.git
cd DevOpsAgent
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage your Python dependencies.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

### 3. Install Dependencies

With your virtual environment active, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Start Infrastructure Services

The system relies on external services like PostgreSQL, Redis, and Neo4j. You can easily start these up using the provided Docker Compose file:

```bash
docker-compose up -d
```

This will spin up the database layer, knowledge graph, and messaging queues in the background.

### 5. Running the System

You're all set! You can now run the agent components (for example, the local test harness or specific tests inside the `airs_v2/` directory):

```bash
# Example: Run the local harness
python harness.py
```

### Stopping Services

When you are done, you can stop the background services with:

```bash
docker-compose down
```
