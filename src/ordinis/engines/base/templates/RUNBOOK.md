# {ENGINE_NAME} - Operations Runbook

> **Document ID:** {ENGINE_ID}-RUNBOOK-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Status:** Draft | Review | Approved
> **On-Call Contact:** {Contact info}

---

## 1. Quick Reference

### 1.1 Key Commands

```bash
# Check engine status
ordinis status {engine}

# View logs
ordinis logs {engine} --tail 100

# Health check
ordinis health {engine}

# Restart engine
ordinis restart {engine}

# Emergency stop
ordinis stop {engine} --force
```

### 1.2 Key Metrics

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| `{engine}_latency_p95` | < {X}ms | < {Y}ms | > {Y}ms |
| `{engine}_error_rate` | < 0.1% | < 1% | > 1% |
| `{engine}_queue_depth` | < 100 | < 500 | > 500 |

### 1.3 Dashboard Links

- Grafana: `{URL}`
- Logs: `{URL}`
- Alerts: `{URL}`

---

## 2. Architecture Overview

### 2.1 Component Diagram

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │ Instance 1│  │ Instance 2│  │ Instance 3│
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │    Database     │
                    └─────────────────┘
```

### 2.2 Dependencies

| Dependency | Type | Required | Health Check |
|------------|------|----------|--------------|
| {Service} | API | Yes | `GET /health` |
| {Database} | Storage | Yes | Connection pool |
| {Queue} | Messaging | No | Queue depth |

---

## 3. Standard Procedures

### 3.1 Startup Procedure

**When:** System startup, after deployment, after maintenance

**Steps:**
1. Verify dependencies are healthy
   ```bash
   ordinis check-deps {engine}
   ```

2. Start engine in dry-run mode
   ```bash
   ordinis start {engine} --dry-run
   ```

3. Verify health check passes
   ```bash
   ordinis health {engine}
   ```

4. Enable production mode
   ```bash
   ordinis enable {engine}
   ```

5. Verify metrics are flowing
   ```bash
   ordinis metrics {engine}
   ```

### 3.2 Shutdown Procedure

**When:** Maintenance, deployment, emergency

**Steps:**
1. Disable new requests (graceful)
   ```bash
   ordinis disable {engine}
   ```

2. Wait for in-flight requests (max 30s)
   ```bash
   ordinis drain {engine} --timeout 30
   ```

3. Stop engine
   ```bash
   ordinis stop {engine}
   ```

4. Verify stopped
   ```bash
   ordinis status {engine}
   # Expected: STOPPED
   ```

### 3.3 Configuration Update

**When:** Config changes required

**Steps:**
1. Validate new config
   ```bash
   ordinis config validate {engine} --file new-config.yaml
   ```

2. Backup current config
   ```bash
   ordinis config backup {engine}
   ```

3. Apply new config
   ```bash
   ordinis config apply {engine} --file new-config.yaml
   ```

4. Restart engine
   ```bash
   ordinis restart {engine}
   ```

5. Verify health
   ```bash
   ordinis health {engine}
   ```

**Rollback:**
```bash
ordinis config restore {engine} --backup latest
ordinis restart {engine}
```

---

## 4. Troubleshooting

### 4.1 High Latency

**Symptoms:**
- `{engine}_latency_p95` > {threshold}ms
- Alert: `{ENGINE_NAME}HighLatency`

**Diagnosis:**
```bash
# Check resource usage
ordinis metrics {engine} --resource

# Check queue depth
ordinis metrics {engine} --queues

# Check dependency latency
ordinis trace {engine} --slow
```

**Resolution:**
1. If queue backed up → Scale up instances
2. If dependency slow → Check dependency health
3. If CPU high → Check for hot loops, increase resources
4. If memory high → Check for leaks, restart

### 4.2 High Error Rate

**Symptoms:**
- `{engine}_error_rate` > 1%
- Alert: `{ENGINE_NAME}HighErrorRate`

**Diagnosis:**
```bash
# Check error breakdown
ordinis errors {engine} --by-type

# View recent errors
ordinis logs {engine} --level ERROR --tail 50

# Check dependency health
ordinis deps {engine}
```

**Resolution:**
1. If dependency errors → Fix/restart dependency
2. If validation errors → Check input source
3. If resource errors → Increase limits
4. If unknown → Escalate to engineering

### 4.3 Engine Not Starting

**Symptoms:**
- State stuck in `INITIALIZING`
- Alert: `{ENGINE_NAME}StartupFailed`

**Diagnosis:**
```bash
# Check startup logs
ordinis logs {engine} --since startup

# Check config validity
ordinis config validate {engine}

# Check port availability
netstat -an | grep {PORT}
```

**Resolution:**
1. If config error → Fix config, restart
2. If port in use → Kill conflicting process
3. If dependency unavailable → Start dependency first
4. If resource unavailable → Check disk/memory

### 4.4 Memory Leak

**Symptoms:**
- Memory usage steadily increasing
- OOM kills

**Diagnosis:**
```bash
# Memory trend
ordinis metrics {engine} --memory --history 24h

# Heap dump (if supported)
ordinis debug {engine} --heap-dump
```

**Resolution:**
1. Short-term: Restart engine
2. Long-term: File bug, analyze heap dump

---

## 5. Alert Playbooks

### 5.1 Alert: {ENGINE_NAME}Down

**Severity:** Critical
**Impact:** {What breaks when this is down}

**Immediate Actions:**
1. Check engine status
   ```bash
   ordinis status {engine}
   ```

2. Check recent deployments
   ```bash
   ordinis deploy history --last 5
   ```

3. Attempt restart
   ```bash
   ordinis restart {engine}
   ```

4. If restart fails, check logs for root cause
   ```bash
   ordinis logs {engine} --level ERROR --since 10m
   ```

**Escalation:**
- If not resolved in 15 minutes → Page on-call engineer
- If customer-impacting → Notify support team

### 5.2 Alert: {ENGINE_NAME}Degraded

**Severity:** Warning
**Impact:** {Degraded functionality}

**Immediate Actions:**
1. Check health details
   ```bash
   ordinis health {engine} --verbose
   ```

2. Identify degraded component
3. Apply targeted fix

**Escalation:**
- If not resolved in 30 minutes → Page on-call

---

## 6. Maintenance Procedures

### 6.1 Database Maintenance

**When:** Weekly, during low-traffic period

**Steps:**
1. Put engine in maintenance mode
   ```bash
   ordinis maintenance {engine} --enable
   ```

2. Run database maintenance
   ```bash
   ordinis db vacuum {engine}
   ordinis db reindex {engine}
   ```

3. Exit maintenance mode
   ```bash
   ordinis maintenance {engine} --disable
   ```

### 6.2 Log Rotation

**Automated:** Yes, via logrotate

**Manual trigger:**
```bash
ordinis logs rotate {engine}
```

### 6.3 Cache Clear

**When:** After data migration, corruption suspected

```bash
ordinis cache clear {engine} --confirm
```

---

## 7. Disaster Recovery

### 7.1 Backup Schedule

| Data | Frequency | Retention | Location |
|------|-----------|-----------|----------|
| Config | Daily | 30 days | S3 |
| State | Hourly | 7 days | S3 |
| Logs | Real-time | 90 days | Elasticsearch |

### 7.2 Recovery Procedures

**Full Recovery:**
```bash
# Stop engine
ordinis stop {engine}

# Restore from backup
ordinis restore {engine} --backup {BACKUP_ID}

# Restart
ordinis start {engine}

# Verify
ordinis health {engine}
```

**Recovery Time Objective (RTO):** {X} minutes
**Recovery Point Objective (RPO):** {X} minutes

---

## 8. Contact Information

| Role | Name | Contact |
|------|------|---------|
| Primary On-Call | {Name} | {Phone/Slack} |
| Secondary On-Call | {Name} | {Phone/Slack} |
| Engineering Lead | {Name} | {Email} |
| Product Owner | {Name} | {Email} |

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial runbook |
