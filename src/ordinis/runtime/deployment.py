"""
Production Deployment and System Integration.

Implements production-grade deployment infrastructure:
- System health monitoring
- Component integration testing
- Deployment configuration
- Graceful shutdown/startup
- Environment management

Step 10 of Trade Enhancement Roadmap.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import os
import signal
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component health status."""
    
    UNKNOWN = auto()
    STARTING = auto()
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    STOPPED = auto()


class DeploymentEnvironment(Enum):
    """Deployment environments."""
    
    DEVELOPMENT = auto()
    STAGING = auto()
    PAPER = auto()  # Paper trading
    PRODUCTION = auto()


@dataclass
class ComponentHealth:
    """Health status for a component."""
    
    component_name: str
    status: ComponentStatus
    last_check: datetime
    latency_ms: float | None = None
    error_message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return self.status in (ComponentStatus.HEALTHY, ComponentStatus.STARTING)
        
    @property
    def is_operational(self) -> bool:
        return self.status in (ComponentStatus.HEALTHY, ComponentStatus.DEGRADED)


@dataclass
class SystemHealth:
    """Overall system health status."""
    
    timestamp: datetime
    environment: DeploymentEnvironment
    overall_status: ComponentStatus
    components: dict[str, ComponentHealth]
    uptime_seconds: float
    warnings: list[str]
    errors: list[str]
    
    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.components.values() if c.is_healthy)
        
    @property
    def total_count(self) -> int:
        return len(self.components)


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    
    environment: DeploymentEnvironment
    
    # Component toggles
    enable_live_trading: bool = False
    enable_paper_trading: bool = True
    enable_backtesting: bool = True
    enable_websockets: bool = True
    enable_learning: bool = True
    enable_synapse: bool = True
    
    # Safety
    kill_switch_enabled: bool = True
    circuit_breaker_enabled: bool = True
    max_daily_loss_pct: float = 0.05
    max_position_pct: float = 0.10
    
    # Health checks
    health_check_interval_seconds: int = 30
    unhealthy_threshold: int = 3  # Consecutive failures
    
    # Startup/shutdown
    startup_timeout_seconds: int = 60
    shutdown_timeout_seconds: int = 30
    graceful_shutdown: bool = True
    
    # Monitoring
    metrics_enabled: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_environment(cls) -> "DeploymentConfig":
        """Create config from environment variables."""
        env_str = os.getenv("ORDINIS_ENVIRONMENT", "development").upper()
        try:
            env = DeploymentEnvironment[env_str]
        except KeyError:
            env = DeploymentEnvironment.DEVELOPMENT
            
        return cls(
            environment=env,
            enable_live_trading=os.getenv("ORDINIS_LIVE_TRADING", "false").lower() == "true",
            enable_paper_trading=os.getenv("ORDINIS_PAPER_TRADING", "true").lower() == "true",
            log_level=os.getenv("ORDINIS_LOG_LEVEL", "INFO"),
            max_daily_loss_pct=float(os.getenv("ORDINIS_MAX_DAILY_LOSS", "0.05")),
            max_position_pct=float(os.getenv("ORDINIS_MAX_POSITION", "0.10")),
        )


class HealthMonitor:
    """
    System health monitoring.
    
    Continuously monitors component health and triggers alerts.
    """
    
    def __init__(
        self,
        config: DeploymentConfig,
    ) -> None:
        """Initialize health monitor."""
        self.config = config
        self._components: dict[str, Callable[[], ComponentHealth]] = {}
        self._health_history: dict[str, list[ComponentHealth]] = {}
        self._failure_counts: dict[str, int] = {}
        self._start_time = datetime.utcnow()
        self._running = False
        self._check_task: asyncio.Task | None = None
        
        # Callbacks
        self._on_unhealthy: list[Callable[[str, ComponentHealth], None]] = []
        self._on_recovery: list[Callable[[str, ComponentHealth], None]] = []
        
    def register_component(
        self,
        name: str,
        health_check: Callable[[], ComponentHealth],
    ) -> None:
        """Register a component for health monitoring."""
        self._components[name] = health_check
        self._health_history[name] = []
        self._failure_counts[name] = 0
        
        logger.debug(f"Registered component for health monitoring: {name}")
        
    async def start(self) -> None:
        """Start health monitoring."""
        self._running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health monitor started")
        
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")
        
    async def _health_check_loop(self) -> None:
        """Continuous health check loop."""
        while self._running:
            try:
                await self._run_health_checks()
            except Exception as e:
                logger.error(f"Health check error: {e}")
                
            await asyncio.sleep(self.config.health_check_interval_seconds)
            
    async def _run_health_checks(self) -> None:
        """Run all health checks."""
        for name, check_fn in self._components.items():
            try:
                start = datetime.utcnow()
                health = check_fn()
                health.latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
                
            except Exception as e:
                health = ComponentHealth(
                    component_name=name,
                    status=ComponentStatus.UNHEALTHY,
                    last_check=datetime.utcnow(),
                    error_message=str(e),
                )
                
            # Track history
            self._health_history[name].append(health)
            if len(self._health_history[name]) > 100:
                self._health_history[name] = self._health_history[name][-100:]
                
            # Track failures
            prev_count = self._failure_counts[name]
            
            if not health.is_healthy:
                self._failure_counts[name] += 1
                
                if self._failure_counts[name] >= self.config.unhealthy_threshold:
                    for callback in self._on_unhealthy:
                        try:
                            callback(name, health)
                        except Exception as e:
                            logger.error(f"Unhealthy callback error: {e}")
            else:
                if prev_count >= self.config.unhealthy_threshold:
                    # Recovered
                    for callback in self._on_recovery:
                        try:
                            callback(name, health)
                        except Exception as e:
                            logger.error(f"Recovery callback error: {e}")
                            
                self._failure_counts[name] = 0
                
    def get_system_health(self) -> SystemHealth:
        """Get overall system health."""
        components = {}
        warnings = []
        errors = []
        
        for name, history in self._health_history.items():
            if history:
                health = history[-1]
                components[name] = health
                
                if health.status == ComponentStatus.DEGRADED:
                    warnings.append(f"{name} is degraded")
                elif health.status == ComponentStatus.UNHEALTHY:
                    errors.append(f"{name} is unhealthy: {health.error_message}")
            else:
                components[name] = ComponentHealth(
                    component_name=name,
                    status=ComponentStatus.UNKNOWN,
                    last_check=datetime.utcnow(),
                )
                
        # Calculate overall status
        if errors:
            overall = ComponentStatus.UNHEALTHY
        elif warnings:
            overall = ComponentStatus.DEGRADED
        elif all(c.is_healthy for c in components.values()):
            overall = ComponentStatus.HEALTHY
        else:
            overall = ComponentStatus.UNKNOWN
            
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        return SystemHealth(
            timestamp=datetime.utcnow(),
            environment=self.config.environment,
            overall_status=overall,
            components=components,
            uptime_seconds=uptime,
            warnings=warnings,
            errors=errors,
        )
        
    def on_unhealthy(
        self,
        callback: Callable[[str, ComponentHealth], None],
    ) -> None:
        """Register callback for unhealthy events."""
        self._on_unhealthy.append(callback)
        
    def on_recovery(
        self,
        callback: Callable[[str, ComponentHealth], None],
    ) -> None:
        """Register callback for recovery events."""
        self._on_recovery.append(callback)


class GracefulShutdownManager:
    """
    Manages graceful system shutdown.
    
    Ensures all components are properly stopped.
    """
    
    def __init__(
        self,
        config: DeploymentConfig,
    ) -> None:
        """Initialize shutdown manager."""
        self.config = config
        self._shutdown_handlers: list[tuple[int, str, Callable[[], Any]]] = []
        self._is_shutting_down = False
        self._shutdown_event = asyncio.Event()
        
    def register_handler(
        self,
        name: str,
        handler: Callable[[], Any],
        priority: int = 50,
    ) -> None:
        """
        Register shutdown handler.
        
        Lower priority = earlier shutdown.
        """
        self._shutdown_handlers.append((priority, name, handler))
        self._shutdown_handlers.sort(key=lambda x: x[0])
        
        logger.debug(f"Registered shutdown handler: {name} (priority: {priority})")
        
    def setup_signal_handlers(self) -> None:
        """Set up OS signal handlers."""
        if os.name == "nt":
            # Windows
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        else:
            # Unix
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self.shutdown(str(s))),
                )
                
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle OS signal (Windows)."""
        asyncio.create_task(self.shutdown(f"signal_{signum}"))
        
    async def shutdown(self, reason: str = "unknown") -> None:
        """
        Execute graceful shutdown.
        
        Args:
            reason: Reason for shutdown
        """
        if self._is_shutting_down:
            logger.warning("Shutdown already in progress")
            return
            
        self._is_shutting_down = True
        logger.info(f"Starting graceful shutdown (reason: {reason})")
        
        start = datetime.utcnow()
        deadline = start + timedelta(seconds=self.config.shutdown_timeout_seconds)
        
        for priority, name, handler in self._shutdown_handlers:
            remaining = (deadline - datetime.utcnow()).total_seconds()
            
            if remaining <= 0:
                logger.warning(f"Shutdown timeout - skipping remaining handlers")
                break
                
            try:
                logger.info(f"Stopping {name}...")
                
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=remaining)
                else:
                    handler()
                    
                logger.info(f"Stopped {name}")
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout stopping {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
                
        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.info(f"Shutdown completed in {elapsed:.1f}s")
        
        self._shutdown_event.set()
        
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self._shutdown_event.wait()


class SystemIntegrator:
    """
    Integrates all system components.
    
    Main entry point for system startup and coordination.
    """
    
    def __init__(
        self,
        config: DeploymentConfig | None = None,
    ) -> None:
        """Initialize system integrator."""
        self.config = config or DeploymentConfig.from_environment()
        self.health_monitor = HealthMonitor(self.config)
        self.shutdown_manager = GracefulShutdownManager(self.config)
        
        self._components: dict[str, Any] = {}
        self._started = False
        
        # Register shutdown for health monitor
        self.shutdown_manager.register_handler(
            "health_monitor",
            self.health_monitor.stop,
            priority=90,  # Stop late
        )
        
    def register_component(
        self,
        name: str,
        component: Any,
        health_check: Callable[[], ComponentHealth] | None = None,
        startup: Callable[[], Any] | None = None,
        shutdown: Callable[[], Any] | None = None,
        priority: int = 50,
    ) -> None:
        """
        Register a system component.
        
        Args:
            name: Component name
            component: Component instance
            health_check: Health check function
            startup: Startup function
            shutdown: Shutdown function
            priority: Shutdown priority (lower = earlier)
        """
        self._components[name] = {
            "instance": component,
            "health_check": health_check,
            "startup": startup,
            "shutdown": shutdown,
            "priority": priority,
        }
        
        if health_check:
            self.health_monitor.register_component(name, health_check)
            
        if shutdown:
            self.shutdown_manager.register_handler(name, shutdown, priority)
            
        logger.info(f"Registered component: {name}")
        
    async def startup(self) -> None:
        """
        Start all system components.
        
        Respects startup order and timeout.
        """
        if self._started:
            logger.warning("System already started")
            return
            
        logger.info(f"Starting Ordinis ({self.config.environment.name})")
        
        start = datetime.utcnow()
        deadline = start + timedelta(seconds=self.config.startup_timeout_seconds)
        
        # Sort by priority (reverse - higher priority starts first)
        sorted_components = sorted(
            self._components.items(),
            key=lambda x: x[1]["priority"],
            reverse=True,
        )
        
        for name, comp in sorted_components:
            remaining = (deadline - datetime.utcnow()).total_seconds()
            
            if remaining <= 0:
                raise TimeoutError("Startup timeout exceeded")
                
            startup_fn = comp.get("startup")
            if startup_fn:
                try:
                    logger.info(f"Starting {name}...")
                    
                    if asyncio.iscoroutinefunction(startup_fn):
                        await asyncio.wait_for(startup_fn(), timeout=remaining)
                    else:
                        startup_fn()
                        
                    logger.info(f"Started {name}")
                    
                except Exception as e:
                    logger.error(f"Failed to start {name}: {e}")
                    raise
                    
        # Start health monitoring
        await self.health_monitor.start()
        
        # Set up signal handlers
        self.shutdown_manager.setup_signal_handlers()
        
        self._started = True
        
        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.info(f"System startup completed in {elapsed:.1f}s")
        
    async def run(self) -> None:
        """Run the system until shutdown."""
        if not self._started:
            await self.startup()
            
        logger.info("System running - waiting for shutdown signal")
        await self.shutdown_manager.wait_for_shutdown()
        
    def get_component(self, name: str) -> Any:
        """Get a registered component."""
        comp = self._components.get(name)
        return comp["instance"] if comp else None


class IntegrationTestRunner:
    """
    Integration test runner for deployment validation.
    """
    
    def __init__(self, integrator: SystemIntegrator) -> None:
        """Initialize test runner."""
        self.integrator = integrator
        self._tests: list[tuple[str, Callable[[], bool]]] = []
        
    def add_test(self, name: str, test_fn: Callable[[], bool]) -> None:
        """Add an integration test."""
        self._tests.append((name, test_fn))
        
    async def run_tests(self) -> dict[str, Any]:
        """Run all integration tests."""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": self.integrator.config.environment.name,
            "tests": [],
            "passed": 0,
            "failed": 0,
        }
        
        for name, test_fn in self._tests:
            try:
                if asyncio.iscoroutinefunction(test_fn):
                    passed = await test_fn()
                else:
                    passed = test_fn()
                    
                result = {
                    "name": name,
                    "passed": passed,
                    "error": None,
                }
                
            except Exception as e:
                passed = False
                result = {
                    "name": name,
                    "passed": False,
                    "error": str(e),
                }
                
            results["tests"].append(result)
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                
        results["all_passed"] = results["failed"] == 0
        
        return results
        
    def add_default_tests(self) -> None:
        """Add default integration tests."""
        
        def test_health_check() -> bool:
            health = self.integrator.health_monitor.get_system_health()
            return health.overall_status in (ComponentStatus.HEALTHY, ComponentStatus.DEGRADED)
            
        def test_all_components_registered() -> bool:
            return len(self.integrator._components) > 0
            
        def test_config_valid() -> bool:
            config = self.integrator.config
            return (
                config.environment is not None
                and config.max_daily_loss_pct > 0
                and config.max_position_pct > 0
            )
            
        self.add_test("health_check", test_health_check)
        self.add_test("components_registered", test_all_components_registered)
        self.add_test("config_valid", test_config_valid)


# Factory function for easy setup
def create_production_system(
    config: DeploymentConfig | None = None,
) -> SystemIntegrator:
    """
    Create and configure production system.
    
    Returns configured SystemIntegrator ready for startup.
    """
    config = config or DeploymentConfig.from_environment()
    integrator = SystemIntegrator(config)
    
    # Log configuration
    logger.info(f"Environment: {config.environment.name}")
    logger.info(f"Live trading: {config.enable_live_trading}")
    logger.info(f"Paper trading: {config.enable_paper_trading}")
    logger.info(f"Max daily loss: {config.max_daily_loss_pct:.1%}")
    logger.info(f"Max position: {config.max_position_pct:.1%}")
    
    return integrator
