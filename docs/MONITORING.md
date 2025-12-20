# Ordinis Monitoring Stack

This document outlines the observability and monitoring infrastructure for the Ordinis trading system. The stack provides real-time metrics, structured logging, alerting, and a unified dashboard for operations.

## 🚀 Quick Start

### 1. Installation
The monitoring tools (Prometheus, Grafana, Loki, etc.) are installed locally in the `bin/` directory. No system-wide installation or admin rights are required.

```powershell
# Download and install tools to ./bin
.\scripts\install_monitoring.ps1
```

### 2. Start Services
Launch the entire stack in background processes.

```powershell
# Starts all services on ports 3000-3006
.\scripts\start_monitoring.ps1
```

### 3. Stop Services
Gracefully shut down all monitoring processes.

```powershell
.\scripts\stop_monitoring.ps1
```

---

## 📊 Architecture & Ports

The system uses a "3000-series" port convention for easy management.

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Grafana** | `3010` | [http://localhost:3010](http://localhost:3010) | **Main Dashboard**. Visualizes all data. Login: `admin`/`admin`. |
| **Prometheus** | `3001` | [http://localhost:3001](http://localhost:3001) | **Metrics Database**. Scrapes and stores time-series data. |
| **Alertmanager** | `3002` | [http://localhost:3002](http://localhost:3002) | **Alert Routing**. Sends notifications (Slack, Email) when rules break. |
| **Loki** | `3003` | [http://localhost:3003](http://localhost:3003) | **Log Database**. Stores and indexes structured logs. |
| **Promtail** | `3004` | [http://localhost:3004](http://localhost:3004) | **Log Shipper**. Reads local log files and sends them to Loki. |
| **Trading Bot** | `3005` | [http://localhost:3005/metrics](http://localhost:3005/metrics) | **Metrics Exporter**. The raw data endpoint exposed by the Python script. |
| **Health Check** | `3006` | [http://localhost:3006/health](http://localhost:3006/health) | **Health Endpoint**. Returns JSON status for uptime monitoring. |

---

## � AI Monitoring

The system includes dedicated monitoring for the Cortex AI Engine to track model usage, latency, and costs.

### Metrics
The following metrics are exposed on port `3005`:

- `ai_requests_total`: Counter of total AI requests (labels: `model`, `operation`).
- `ai_errors_total`: Counter of failed requests (labels: `model`, `operation`).
- `ai_request_duration_seconds`: Histogram of request latency.

### Dashboards
- **AI Development Metrics**: A dedicated dashboard visualizing:
  - Request Rate (per second)
  - Latency (p95)
  - Requests by Model & Operation
  - Error Rates

---

## �🧩 Component Roles

### 1. Grafana (The Dashboard)
*   **Role:** The "Single Pane of Glass". It connects to Prometheus and Loki to visualize everything in one place.
*   **Usage:** Use this to watch account equity, live signals, and system health.
*   **Configuration:** `configs/grafana/` (Dashboards are auto-provisioned).

### 2. Prometheus (The Metrics Engine)
*   **Role:** Scrapes numeric data from the trading bot every 15 seconds (e.g., "Equity = $105,000").
*   **Usage:** Powers the graphs in Grafana. Handles alert rule evaluation.
*   **Configuration:** `configs/prometheus/prometheus.yml` & `alert_rules.yml`.

### 3. Alertmanager (The Notifier)
*   **Role:** Receives alerts from Prometheus (e.g., "Risk Breach Detected") and routes them to the correct channel (Slack, Email, PagerDuty).
*   **Usage:** Configure this to set up your notification channels.
*   **Configuration:** `configs/alertmanager/alertmanager.yml`.

### 4. Loki (The Log Store)
*   **Role:** Stores structured logs (JSON) efficiently. Like Prometheus, but for text.
*   **Usage:** Allows you to filter logs by strategy, symbol, or error type directly in Grafana.
*   **Configuration:** `configs/loki/loki-config.yml`.

### 5. Promtail (The Log Shipper)
*   **Role:** A lightweight agent that tails the `logs/trading.json` file and pushes new lines to Loki.
*   **Usage:** Runs silently in the background.
*   **Configuration:** `configs/promtail/promtail-windows.yml`.

---

## 🛠️ Configuration

All configuration files are located in the `configs/` directory:

*   `configs/prometheus/alert_rules.yml`: Define what triggers an alert (e.g., Drawdown > 5%).
*   `configs/grafana/trading_dashboard.json`: The layout of the Grafana dashboard.
*   `configs/alertmanager/alertmanager.yml`: Setup Slack webhooks or SMTP settings here.

## 🔍 Troubleshooting

*   **Grafana Login:** Default is `admin` / `admin`.
*   **No Data in Graphs:**
    1.  Ensure the trading script is running (`python scripts/trading/live_trading_production.py`).
    2.  Check if `http://localhost:3005/metrics` is accessible.
    3.  Check Prometheus targets at `http://localhost:3001/targets`.
*   **No Logs in Grafana:**
    1.  Ensure the bot is writing to `logs/trading.json`.
    2.  Check Promtail status at `http://localhost:3004/targets`.
