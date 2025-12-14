# {ENGINE_NAME} - State Machine Specification

> **Document ID:** {ENGINE_ID}-STATES-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Status:** Draft | Review | Approved

---

## 1. State Diagram

```
                    ┌─────────────────┐
                    │  UNINITIALIZED  │
                    └────────┬────────┘
                             │ initialize()
                             ▼
                    ┌─────────────────┐
                    │  INITIALIZING   │
                    └────────┬────────┘
                             │ success
              ┌──────────────┼──────────────┐
              │              ▼              │
              │     ┌─────────────────┐     │
              │     │     READY       │     │
              │     └────────┬────────┘     │
              │              │ start()      │
              │              ▼              │
              │     ┌─────────────────┐     │
     pause()  │     │    RUNNING      │◄────┤ resume()
              │     └────────┬────────┘     │
              │              │              │
              │              ▼              │
              │     ┌─────────────────┐     │
              └────►│     PAUSED      │─────┘
                    └────────┬────────┘
                             │ shutdown()
                             ▼
                    ┌─────────────────┐
                    │    STOPPING     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    STOPPED      │
                    └─────────────────┘

        Any State ──────────────────────► ERROR
                     (on unrecoverable error)
```

---

## 2. State Definitions

### 2.1 Core States

| State | Description | Valid Operations |
|-------|-------------|------------------|
| `UNINITIALIZED` | Engine created but not initialized | `initialize()` |
| `INITIALIZING` | Setting up resources | None (wait) |
| `READY` | Initialized, ready to process | `start()`, `shutdown()` |
| `RUNNING` | Actively processing | All domain ops, `pause()`, `shutdown()` |
| `PAUSED` | Temporarily suspended | `resume()`, `shutdown()` |
| `STOPPING` | Shutting down | None (wait) |
| `STOPPED` | Fully shut down | `initialize()` (restart) |
| `ERROR` | Unrecoverable error | `shutdown()`, `reset()` |

### 2.2 State Properties

| State | Is Running | Accepts Requests | Resources Held |
|-------|------------|------------------|----------------|
| `UNINITIALIZED` | No | No | No |
| `INITIALIZING` | No | No | Acquiring |
| `READY` | Yes | Yes | Yes |
| `RUNNING` | Yes | Yes | Yes |
| `PAUSED` | Yes | No | Yes |
| `STOPPING` | No | No | Releasing |
| `STOPPED` | No | No | No |
| `ERROR` | No | No | Unknown |

---

## 3. State Transitions

### 3.1 Transition Matrix

| From \ To | UNINIT | INIT | READY | RUN | PAUSE | STOP | STOPPED | ERROR |
|-----------|--------|------|-------|-----|-------|------|---------|-------|
| UNINIT | - | ✓ | - | - | - | - | - | - |
| INIT | - | - | ✓ | - | - | - | - | ✓ |
| READY | - | - | - | ✓ | - | ✓ | - | ✓ |
| RUN | - | - | - | - | ✓ | ✓ | - | ✓ |
| PAUSE | - | - | - | ✓ | - | ✓ | - | ✓ |
| STOP | - | - | - | - | - | - | ✓ | ✓ |
| STOPPED | ✓ | - | - | - | - | - | - | - |
| ERROR | - | - | - | - | - | ✓ | - | - |

### 3.2 Transition Details

#### UNINITIALIZED → INITIALIZING
**Trigger:** `initialize()` called
**Actions:**
1. Validate configuration
2. Acquire resources
3. Register with orchestrator

**Postconditions:**
- Resources acquired
- Health check passes

#### INITIALIZING → READY
**Trigger:** Initialization completes successfully
**Actions:**
1. Mark engine as ready
2. Emit `engine.ready` event

#### INITIALIZING → ERROR
**Trigger:** Initialization fails
**Actions:**
1. Release any acquired resources
2. Log error details
3. Emit `engine.error` event

#### READY → RUNNING
**Trigger:** `start()` called or first request received
**Actions:**
1. Begin accepting requests
2. Start background tasks (if any)

#### RUNNING → PAUSED
**Trigger:** `pause()` called
**Actions:**
1. Stop accepting new requests
2. Complete in-flight requests
3. Suspend background tasks

#### PAUSED → RUNNING
**Trigger:** `resume()` called
**Actions:**
1. Resume accepting requests
2. Restart background tasks

#### * → STOPPING
**Trigger:** `shutdown()` called
**Actions:**
1. Stop accepting requests
2. Wait for in-flight requests (with timeout)
3. Release resources

#### STOPPING → STOPPED
**Trigger:** Shutdown completes
**Actions:**
1. Emit `engine.stopped` event
2. Deregister from orchestrator

#### * → ERROR
**Trigger:** Unrecoverable error
**Actions:**
1. Log error with full context
2. Emit `engine.error` event
3. Alert operations team

---

## 4. Engine-Specific States

### 4.1 Additional States (if applicable)

| State | Description | Transitions |
|-------|-------------|-------------|
| `{CUSTOM_STATE}` | {Description} | From: {states}, To: {states} |

### 4.2 Substates

```
RUNNING
├── IDLE          # Waiting for requests
├── PROCESSING    # Handling request
└── BACKPRESSURE  # Queue full, throttling
```

---

## 5. Error Recovery

### 5.1 Recovery Procedures

| Error Type | Recovery Action | Max Attempts |
|------------|-----------------|--------------|
| Transient | Retry with backoff | 3 |
| Resource exhaustion | Pause, wait, resume | 1 |
| Configuration | Reload config | 1 |
| Unrecoverable | Shutdown, alert | 0 |

### 5.2 Circuit Breaker States

```
CLOSED ──(failures > threshold)──► OPEN
   ▲                                  │
   │                            (timeout)
   │                                  ▼
   └────(success)──── HALF_OPEN ◄────┘
```

---

## 6. Monitoring

### 6.1 State Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `engine_state` | Gauge | Current state (enum value) |
| `engine_state_transitions_total` | Counter | State transition count |
| `engine_state_duration_seconds` | Histogram | Time spent in each state |

### 6.2 State Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| EngineError | State = ERROR | Critical |
| EngineStopped | State = STOPPED (unexpected) | Warning |
| EngineStuck | State unchanged > 5min | Warning |

---

## 7. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial state specification |
