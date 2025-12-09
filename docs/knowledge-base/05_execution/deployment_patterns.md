# Deployment Patterns for Algorithmic Trading Systems

## Overview

This document covers deployment architectures, patterns, and operational considerations for algorithmic trading systems. Proper deployment is critical for reliability, performance, and regulatory compliance.

**Last Updated**: December 8, 2025

---

## 1. Deployment Architectures

### 1.1 Architecture Options

| Architecture | Latency | Cost | Complexity | Use Case |
|--------------|---------|------|------------|----------|
| **Local workstation** | Variable | Low | Low | Development, paper trading |
| **Cloud VM** | 10-50ms | Medium | Medium | Retail algo trading |
| **Colocation** | < 1ms | High | High | HFT, professional trading |
| **Hybrid** | Variable | Medium | High | Multi-strategy firms |

### 1.2 Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT ARCHITECTURE OPTIONS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LOCAL WORKSTATION              CLOUD DEPLOYMENT           COLOCATION       │
│  ┌─────────────┐               ┌─────────────┐           ┌─────────────┐   │
│  │  Trading    │               │  Cloud VM   │           │  Exchange   │   │
│  │  Software   │               │  (AWS/GCP)  │           │  Data Ctr   │   │
│  │             │               │             │           │             │   │
│  │  ┌───────┐  │               │  ┌───────┐  │           │  ┌───────┐  │   │
│  │  │ Algo  │  │               │  │ Algo  │  │           │  │ Algo  │  │   │
│  │  │Engine │  │               │  │Engine │  │           │  │Engine │  │   │
│  │  └───────┘  │               │  └───────┘  │           │  └───────┘  │   │
│  └──────┬──────┘               └──────┬──────┘           └──────┬──────┘   │
│         │                             │                         │          │
│    Home ISP                      Cloud Network              Cross-connect  │
│    (20-100ms)                    (10-50ms)                   (< 1ms)       │
│         │                             │                         │          │
│         └─────────────────────────────┴─────────────────────────┘          │
│                                   │                                         │
│                              BROKER/EXCHANGE                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cloud Deployment

### 2.1 Cloud Provider Selection

| Provider | Trading-Relevant Features | Financial Region Options |
|----------|--------------------------|-------------------------|
| **AWS** | Direct Connect, low-latency zones | US-East (NY), EU-West |
| **GCP** | Partner Interconnect, Anthos | US-East, Europe |
| **Azure** | ExpressRoute, Proximity Placement | US-East, UK |

### 2.2 AWS Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AWS TRADING ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          VPC (us-east-1)                             │   │
│  │                                                                       │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │   │
│  │  │   Public    │    │   Private   │    │   Private   │              │   │
│  │  │   Subnet    │    │   Subnet    │    │   Subnet    │              │   │
│  │  │             │    │             │    │             │              │   │
│  │  │  ┌───────┐  │    │  ┌───────┐  │    │  ┌───────┐  │              │   │
│  │  │  │ NAT   │  │    │  │Trading│  │    │  │  RDS  │  │              │   │
│  │  │  │Gateway│  │    │  │ EC2   │  │    │  │Postgre│  │              │   │
│  │  │  └───────┘  │    │  └───────┘  │    │  └───────┘  │              │   │
│  │  │             │    │             │    │             │              │   │
│  │  │  ┌───────┐  │    │  ┌───────┐  │    │  ┌───────┐  │              │   │
│  │  │  │  ALB  │  │    │  │Redis  │  │    │  │  S3   │  │              │   │
│  │  │  │       │  │    │  │Cache  │  │    │  │Bucket │  │              │   │
│  │  │  └───────┘  │    │  └───────┘  │    │  └───────┘  │              │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                     ┌─────────────────────────┐                             │
│                     │    Direct Connect       │                             │
│                     │    (to broker/exchange) │                             │
│                     └─────────────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Infrastructure as Code

```python
# terraform/main.tf equivalent in Python/Pulumi

from dataclasses import dataclass
from typing import List


@dataclass
class TradingInfraConfig:
    """Configuration for trading infrastructure."""
    region: str = "us-east-1"
    instance_type: str = "c5.xlarge"  # Compute optimized
    availability_zones: List[str] = None
    enable_monitoring: bool = True
    backup_retention_days: int = 30


# Example Pulumi/CDK-style definition
class TradingInfrastructure:
    """
    Trading system infrastructure definition.
    """

    def __init__(self, config: TradingInfraConfig):
        self.config = config

    def create_vpc(self):
        """
        Create VPC with public/private subnets.
        """
        return {
            "cidr_block": "10.0.0.0/16",
            "enable_dns_hostnames": True,
            "subnets": {
                "public": ["10.0.1.0/24", "10.0.2.0/24"],
                "private": ["10.0.10.0/24", "10.0.11.0/24"]
            }
        }

    def create_trading_instance(self):
        """
        Create EC2 instance for trading engine.

        Key considerations:
        - Compute optimized (c5/c6i) for CPU-bound algos
        - Memory optimized (r5/r6i) for large datasets
        - EBS optimized for disk I/O
        - Enhanced networking for lower latency
        """
        return {
            "instance_type": self.config.instance_type,
            "ami": "ami-trading-optimized",
            "ebs_optimized": True,
            "monitoring": self.config.enable_monitoring,
            "security_groups": ["sg-trading"],
            "user_data": self._trading_startup_script()
        }

    def _trading_startup_script(self) -> str:
        """Startup script for trading instance."""
        return """#!/bin/bash
# Update system
yum update -y

# Install dependencies
yum install -y python3.11 git docker

# Configure time sync (critical for trading)
yum install -y chrony
systemctl enable chronyd
systemctl start chronyd

# Set CPU governor to performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu
done

# Disable CPU frequency scaling
systemctl disable ondemand

# Pull and start trading container
docker pull trading-system:latest
docker run -d --name trading \\
    --restart unless-stopped \\
    -v /data:/app/data \\
    trading-system:latest
"""
```

---

## 3. Containerized Deployment

### 3.1 Docker Configuration

```dockerfile
# Dockerfile for trading system

FROM python:3.11-slim

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 trader
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=trader:trader . .

# Switch to non-root user
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.health import check_health; check_health()"

# Run trading engine
CMD ["python", "-m", "src.main"]
```

### 3.2 Docker Compose for Development

```yaml
# docker-compose.yml

version: '3.8'

services:
  trading:
    build: .
    container_name: ordinis-trading
    restart: unless-stopped
    environment:
      - ENVIRONMENT=development
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
      - DATABASE_URL=postgresql://trader:password@db:5432/trading
      - REDIS_URL=redis://cache:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - db
      - cache
    ports:
      - "8501:8501"  # Streamlit dashboard

  db:
    image: timescale/timescaledb:latest-pg14
    container_name: ordinis-db
    environment:
      - POSTGRES_USER=trader
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=trading
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  cache:
    image: redis:7-alpine
    container_name: ordinis-cache
    volumes:
      - redisdata:/data
    ports:
      - "6379:6379"

  monitoring:
    image: grafana/grafana:latest
    container_name: ordinis-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafanadata:/var/lib/grafana

volumes:
  pgdata:
  redisdata:
  grafanadata:
```

### 3.3 Kubernetes Deployment

```yaml
# kubernetes/trading-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-engine
  labels:
    app: ordinis-trading
spec:
  replicas: 1  # Single instance for trading (no split brain)
  strategy:
    type: Recreate  # Not RollingUpdate for trading
  selector:
    matchLabels:
      app: ordinis-trading
  template:
    metadata:
      labels:
        app: ordinis-trading
    spec:
      containers:
        - name: trading
          image: ordinis/trading:latest
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          env:
            - name: ALPACA_API_KEY
              valueFrom:
                secretKeyRef:
                  name: trading-secrets
                  key: alpaca-api-key
            - name: ALPACA_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: trading-secrets
                  key: alpaca-secret-key
          ports:
            - containerPort: 8501
          livenessProbe:
            httpGet:
              path: /health
              port: 8501
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8501
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: data
              mountPath: /app/data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: trading-data-pvc
      nodeSelector:
        workload-type: trading  # Dedicated node pool
      tolerations:
        - key: "trading"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
```

---

## 4. High Availability Patterns

### 4.1 Active-Passive Failover

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ACTIVE-PASSIVE FAILOVER PATTERN                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐         ┌───────────────────┐                       │
│  │   PRIMARY (AZ-A)  │         │  STANDBY (AZ-B)   │                       │
│  │   ┌───────────┐   │         │   ┌───────────┐   │                       │
│  │   │  Trading  │   │         │   │  Trading  │   │                       │
│  │   │  Engine   │───┼────────▶│   │  Engine   │   │                       │
│  │   │  ACTIVE   │   │ Repl.   │   │  STANDBY  │   │                       │
│  │   └───────────┘   │         │   └───────────┘   │                       │
│  │         │         │         │         │         │                       │
│  │         ▼         │         │         ▼         │                       │
│  │   ┌───────────┐   │         │   ┌───────────┐   │                       │
│  │   │    DB     │───┼────────▶│   │    DB     │   │                       │
│  │   │  Primary  │   │ Sync    │   │  Replica  │   │                       │
│  │   └───────────┘   │         │   └───────────┘   │                       │
│  └───────────────────┘         └───────────────────┘                       │
│                                                                             │
│                        ┌─────────────────┐                                  │
│                        │  Health Monitor │                                  │
│                        │  (Watchdog)     │                                  │
│                        └─────────────────┘                                  │
│                               │                                             │
│                               ▼                                             │
│                    On PRIMARY failure:                                      │
│                    1. Detect failure (health checks)                        │
│                    2. Promote STANDBY to ACTIVE                             │
│                    3. Update DNS/routing                                    │
│                    4. Alert operators                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Failover Implementation

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import asyncio


class NodeRole(Enum):
    PRIMARY = "primary"
    STANDBY = "standby"
    TRANSITIONING = "transitioning"


class NodeHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class NodeStatus:
    role: NodeRole
    health: NodeHealth
    last_heartbeat: datetime
    positions_synced: bool
    orders_synced: bool


class FailoverManager:
    """
    Manages active-passive failover for trading system.
    """

    def __init__(
        self,
        heartbeat_interval: float = 1.0,
        failure_threshold: int = 3,
        recovery_threshold: int = 5
    ):
        self.heartbeat_interval = heartbeat_interval
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold

        self._role = NodeRole.STANDBY
        self._consecutive_failures = 0
        self._consecutive_successes = 0

    async def start_heartbeat_loop(self, peer_url: str):
        """
        Monitor peer node health and manage failover.
        """
        while True:
            try:
                peer_healthy = await self._check_peer_health(peer_url)

                if peer_healthy:
                    self._consecutive_successes += 1
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
                    self._consecutive_successes = 0

                # Handle state transitions
                await self._evaluate_role_transition()

            except Exception as e:
                self._consecutive_failures += 1

            await asyncio.sleep(self.heartbeat_interval)

    async def _check_peer_health(self, peer_url: str) -> bool:
        """Check if peer node is healthy."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{peer_url}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _evaluate_role_transition(self):
        """Evaluate and execute role transitions."""
        if self._role == NodeRole.STANDBY:
            # Check if should promote to primary
            if self._consecutive_failures >= self.failure_threshold:
                await self._promote_to_primary()

        elif self._role == NodeRole.PRIMARY:
            # Check if peer recovered and should demote
            if self._consecutive_successes >= self.recovery_threshold:
                # Original primary recovered - coordinate handoff
                pass

    async def _promote_to_primary(self):
        """
        Promote this node to primary role.
        Critical section - must be atomic.
        """
        self._role = NodeRole.TRANSITIONING

        try:
            # 1. Acquire distributed lock
            lock_acquired = await self._acquire_leader_lock()
            if not lock_acquired:
                self._role = NodeRole.STANDBY
                return

            # 2. Verify peer is truly down (split-brain prevention)
            peer_status = await self._verify_peer_status()
            if peer_status == NodeHealth.HEALTHY:
                await self._release_leader_lock()
                self._role = NodeRole.STANDBY
                return

            # 3. Sync latest state from database
            await self._sync_state_from_db()

            # 4. Cancel any in-flight orders from old primary
            await self._cancel_orphaned_orders()

            # 5. Activate trading engine
            await self._activate_trading()

            # 6. Update health check endpoint
            self._role = NodeRole.PRIMARY

            # 7. Alert operations team
            await self._send_failover_alert()

        except Exception as e:
            await self._release_leader_lock()
            self._role = NodeRole.STANDBY
            raise

    async def _acquire_leader_lock(self) -> bool:
        """
        Acquire distributed lock for leader election.
        Uses Redis or Consul for coordination.
        """
        # Implementation depends on coordination service
        pass

    async def _sync_state_from_db(self):
        """Sync positions and orders from database."""
        pass

    async def _cancel_orphaned_orders(self):
        """Cancel orders that may be orphaned from failed primary."""
        pass

    async def _activate_trading(self):
        """Activate trading engine on this node."""
        pass

    async def _send_failover_alert(self):
        """Alert operations team of failover event."""
        pass
```

---

## 5. Configuration Management

### 5.1 Environment-Based Configuration

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import os


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class BrokerConfig:
    """Broker connection configuration."""
    api_key: str
    secret_key: str
    base_url: str
    paper_trading: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    name: str
    user: str
    password: str
    ssl_mode: str = "require"


@dataclass
class TradingConfig:
    """Trading system configuration."""
    environment: Environment
    broker: BrokerConfig
    database: DatabaseConfig
    max_position_size: float = 0.10
    max_daily_loss: float = 0.03
    enable_live_trading: bool = False
    log_level: str = "INFO"


class ConfigLoader:
    """
    Load configuration from environment variables.
    """

    @staticmethod
    def load() -> TradingConfig:
        """Load configuration based on environment."""
        env = Environment(os.getenv("ENVIRONMENT", "development"))

        broker = BrokerConfig(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            base_url=ConfigLoader._get_broker_url(env),
            paper_trading=env != Environment.PRODUCTION
        )

        database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "trading"),
            user=os.getenv("DB_USER", "trader"),
            password=os.getenv("DB_PASSWORD", ""),
            ssl_mode="require" if env == Environment.PRODUCTION else "disable"
        )

        return TradingConfig(
            environment=env,
            broker=broker,
            database=database,
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.10")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.03")),
            enable_live_trading=env == Environment.PRODUCTION,
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )

    @staticmethod
    def _get_broker_url(env: Environment) -> str:
        """Get broker URL based on environment."""
        urls = {
            Environment.DEVELOPMENT: "https://paper-api.alpaca.markets",
            Environment.STAGING: "https://paper-api.alpaca.markets",
            Environment.PRODUCTION: "https://api.alpaca.markets"
        }
        return urls[env]
```

### 5.2 Secrets Management

```python
from abc import ABC, abstractmethod
from typing import Dict
import json


class SecretsProvider(ABC):
    """Abstract base for secrets providers."""

    @abstractmethod
    def get_secret(self, name: str) -> str:
        pass


class AWSSecretsManager(SecretsProvider):
    """AWS Secrets Manager integration."""

    def __init__(self, region: str = "us-east-1"):
        import boto3
        self.client = boto3.client("secretsmanager", region_name=region)

    def get_secret(self, name: str) -> str:
        """Retrieve secret from AWS Secrets Manager."""
        response = self.client.get_secret_value(SecretId=name)
        return response["SecretString"]

    def get_trading_secrets(self) -> Dict[str, str]:
        """Get all trading-related secrets."""
        secret_string = self.get_secret("ordinis/trading")
        return json.loads(secret_string)


class HashiCorpVault(SecretsProvider):
    """HashiCorp Vault integration."""

    def __init__(self, url: str, token: str):
        import hvac
        self.client = hvac.Client(url=url, token=token)

    def get_secret(self, name: str) -> str:
        """Retrieve secret from Vault."""
        response = self.client.secrets.kv.v2.read_secret_version(path=name)
        return response["data"]["data"]["value"]
```

---

## 6. Deployment Pipeline

### 6.1 CI/CD Pipeline

{% raw %}
```yaml
# .github/workflows/deploy.yml

name: Deploy Trading System

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ -v --cov=src

      - name: Run linting
        run: |
          ruff check src/
          mypy src/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}

    steps:
      - uses: actions/checkout@v4

      - name: Log in to registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - name: Deploy to staging
        run: |
          # Update Kubernetes deployment
          kubectl set image deployment/trading-engine \
            trading=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            --namespace=staging

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/trading-engine \
            --namespace=staging \
            --timeout=300s

      - name: Run smoke tests
        run: |
          # Verify system is healthy
          curl -f https://staging.ordinis.local/health

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: Production deployment gate
        run: |
          # Check market hours - don't deploy during trading
          HOUR=$(TZ=America/New_York date +%H)
          if [ $HOUR -ge 9 ] && [ $HOUR -lt 16 ]; then
            echo "Cannot deploy during market hours"
            exit 1
          fi

      - name: Deploy to production
        run: |
          kubectl set image deployment/trading-engine \
            trading=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            --namespace=production

      - name: Verify deployment
        run: |
          kubectl rollout status deployment/trading-engine \
            --namespace=production \
            --timeout=600s
```
{% endraw %}

### 6.2 Blue-Green Deployment

```python
class BlueGreenDeployer:
    """
    Blue-green deployment for zero-downtime updates.
    Critical for trading systems that cannot have downtime.
    """

    def __init__(self, k8s_client):
        self.k8s = k8s_client

    async def deploy(self, new_image: str) -> bool:
        """
        Execute blue-green deployment.

        1. Deploy new version to inactive environment
        2. Warm up and test new version
        3. Switch traffic
        4. Monitor for issues
        5. Rollback if problems detected
        """
        current_env = await self._get_active_environment()
        new_env = "green" if current_env == "blue" else "blue"

        try:
            # Deploy to inactive environment
            await self._deploy_to_environment(new_env, new_image)

            # Wait for pods to be ready
            await self._wait_for_ready(new_env)

            # Run smoke tests
            smoke_passed = await self._run_smoke_tests(new_env)
            if not smoke_passed:
                raise DeploymentError("Smoke tests failed")

            # Switch traffic
            await self._switch_traffic(new_env)

            # Monitor for errors
            await self._monitor_deployment(duration_seconds=300)

            # Success - cleanup old environment
            await self._scale_down_environment(current_env)

            return True

        except Exception as e:
            # Rollback
            await self._switch_traffic(current_env)
            await self._scale_down_environment(new_env)
            raise

    async def _get_active_environment(self) -> str:
        """Determine which environment is currently active."""
        pass

    async def _switch_traffic(self, target_env: str):
        """Switch ingress to target environment."""
        pass

    async def _monitor_deployment(self, duration_seconds: int):
        """
        Monitor new deployment for errors.
        Automatically rollback if error rate exceeds threshold.
        """
        pass
```

---

## 7. Operational Considerations

### 7.1 Deployment Checklist

| Phase | Check | Critical |
|-------|-------|----------|
| **Pre-deploy** | All tests passing | Yes |
| **Pre-deploy** | No open positions (if possible) | Yes |
| **Pre-deploy** | Outside market hours | Recommended |
| **Pre-deploy** | Backup current state | Yes |
| **Deploy** | Staged rollout | Yes |
| **Deploy** | Health checks passing | Yes |
| **Post-deploy** | Smoke tests passing | Yes |
| **Post-deploy** | Monitoring active | Yes |
| **Post-deploy** | Rollback plan ready | Yes |

### 7.2 Market Hours Awareness

```python
from datetime import datetime, time
import pytz


class MarketHoursChecker:
    """
    Check market hours for deployment decisions.
    """

    NYSE_OPEN = time(9, 30)
    NYSE_CLOSE = time(16, 0)
    NYSE_TZ = pytz.timezone("America/New_York")

    @classmethod
    def is_market_open(cls) -> bool:
        """Check if US market is currently open."""
        now = datetime.now(cls.NYSE_TZ)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Holiday check (simplified)
        # Full implementation would check holiday calendar

        # Hours check
        current_time = now.time()
        return cls.NYSE_OPEN <= current_time < cls.NYSE_CLOSE

    @classmethod
    def can_deploy(cls) -> tuple[bool, str]:
        """
        Check if deployment is safe.

        Returns:
            (can_deploy, reason)
        """
        if cls.is_market_open():
            return False, "Market is open - deployment blocked"

        now = datetime.now(cls.NYSE_TZ)

        # Pre-market buffer
        if now.time() >= time(7, 0) and now.time() < cls.NYSE_OPEN:
            return False, "Pre-market period - deploy with caution"

        return True, "Safe to deploy"
```

---

## 8. References

- Beyer et al. (2016): "Site Reliability Engineering" (Google SRE book)
- Burns, B. (2018): "Designing Distributed Systems"
- AWS Well-Architected Framework: https://aws.amazon.com/architecture/well-architected/
- Kubernetes Documentation: https://kubernetes.io/docs/
- HashiCorp Vault: https://www.vaultproject.io/docs
