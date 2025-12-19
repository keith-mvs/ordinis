"""Tests for monitoring.health module."""

from ordinis.adapters.telemetry.health import (
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    api_health_check,
    database_health_check,
    disk_space_health_check,
    get_health_check,
    memory_health_check,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_health_check_result_initialization(self):
        """Test HealthCheckResult initialization."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
        )

        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert isinstance(result.details, dict)
        assert result.response_time == 0.0

    def test_health_check_result_with_details(self):
        """Test HealthCheckResult with details."""
        details = {"cpu_usage": 45.2, "memory_usage": 60.5}
        result = HealthCheckResult(
            name="system_check",
            status=HealthStatus.HEALTHY,
            details=details,
        )

        assert result.details == details
        assert result.details["cpu_usage"] == 45.2

    def test_health_check_result_to_dict(self):
        """Test HealthCheckResult to_dict conversion."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.DEGRADED,
            message="Performance degraded",
            details={"latency": 500},
            response_time=0.15,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["name"] == "test_check"
        assert result_dict["status"] == "degraded"
        assert result_dict["message"] == "Performance degraded"
        assert result_dict["details"]["latency"] == 500
        assert result_dict["response_time"] == 0.15
        assert "timestamp" in result_dict


class TestHealthCheck:
    """Tests for HealthCheck class."""

    def test_health_check_initialization(self):
        """Test HealthCheck initialization."""
        health = HealthCheck()

        assert len(health._checks) == 0
        assert len(health._last_results) == 0

    def test_register_check(self):
        """Test registering a health check."""
        health = HealthCheck()

        def sample_check():
            return HealthCheckResult(
                name="sample",
                status=HealthStatus.HEALTHY,
            )

        health.register_check("sample", sample_check)

        assert "sample" in health._checks
        assert health._checks["sample"] == sample_check

    def test_run_check_success(self):
        """Test running a successful health check."""
        health = HealthCheck()

        def sample_check():
            return HealthCheckResult(
                name="sample",
                status=HealthStatus.HEALTHY,
                message="Check passed",
            )

        health.register_check("sample", sample_check)
        result = health.run_check("sample")

        assert result.name == "sample"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Check passed"
        assert result.response_time > 0

    def test_run_check_not_found(self):
        """Test running a non-existent check."""
        health = HealthCheck()

        result = health.run_check("nonexistent")

        assert result.name == "nonexistent"
        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.message

    def test_run_check_with_exception(self):
        """Test running a check that raises an exception."""
        health = HealthCheck()

        def failing_check():
            raise RuntimeError("Check failed")

        health.register_check("failing", failing_check)
        result = health.run_check("failing")

        assert result.name == "failing"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message
        assert result.response_time > 0

    def test_run_all_checks(self):
        """Test running all registered checks."""
        health = HealthCheck()

        def check1():
            return HealthCheckResult(name="check1", status=HealthStatus.HEALTHY)

        def check2():
            return HealthCheckResult(name="check2", status=HealthStatus.DEGRADED)

        health.register_check("check1", check1)
        health.register_check("check2", check2)

        results = health.run_all_checks()

        assert len(results) == 2
        assert "check1" in results
        assert "check2" in results
        assert results["check1"].status == HealthStatus.HEALTHY
        assert results["check2"].status == HealthStatus.DEGRADED

    def test_get_overall_status_no_checks(self):
        """Test overall status with no checks."""
        health = HealthCheck()

        status = health.get_overall_status()

        assert status == HealthStatus.UNKNOWN

    def test_get_overall_status_all_healthy(self):
        """Test overall status when all checks are healthy."""
        health = HealthCheck()

        def healthy_check():
            return HealthCheckResult(name="test", status=HealthStatus.HEALTHY)

        health.register_check("check1", healthy_check)
        health.register_check("check2", healthy_check)
        health.run_all_checks()

        status = health.get_overall_status()

        assert status == HealthStatus.HEALTHY

    def test_get_overall_status_with_degraded(self):
        """Test overall status with degraded checks."""
        health = HealthCheck()

        def healthy_check():
            return HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY)

        def degraded_check():
            return HealthCheckResult(name="degraded", status=HealthStatus.DEGRADED)

        health.register_check("check1", healthy_check)
        health.register_check("check2", degraded_check)
        health.run_all_checks()

        status = health.get_overall_status()

        assert status == HealthStatus.DEGRADED

    def test_get_overall_status_with_unhealthy(self):
        """Test overall status with unhealthy checks."""
        health = HealthCheck()

        def healthy_check():
            return HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY)

        def unhealthy_check():
            return HealthCheckResult(name="unhealthy", status=HealthStatus.UNHEALTHY)

        health.register_check("check1", healthy_check)
        health.register_check("check2", unhealthy_check)
        health.run_all_checks()

        status = health.get_overall_status()

        assert status == HealthStatus.UNHEALTHY

    def test_get_health_report(self):
        """Test getting comprehensive health report."""
        health = HealthCheck()

        def sample_check():
            return HealthCheckResult(
                name="sample",
                status=HealthStatus.HEALTHY,
                message="All systems operational",
            )

        health.register_check("sample", sample_check)
        report = health.get_health_report()

        assert isinstance(report, dict)
        assert "overall_status" in report
        assert "timestamp" in report
        assert "checks" in report
        assert "sample" in report["checks"]
        assert report["checks"]["sample"]["status"] == "healthy"

    def test_get_last_results(self):
        """Test getting last check results."""
        health = HealthCheck()

        def sample_check():
            return HealthCheckResult(name="sample", status=HealthStatus.HEALTHY)

        health.register_check("sample", sample_check)
        health.run_check("sample")

        last_results = health.get_last_results()

        assert isinstance(last_results, dict)
        assert "sample" in last_results
        assert last_results["sample"].status == HealthStatus.HEALTHY

    def test_last_results_updated_after_run(self):
        """Test that last results are updated after each run."""
        health = HealthCheck()

        counter = {"value": 0}

        def changing_check():
            counter["value"] += 1
            status = HealthStatus.HEALTHY if counter["value"] == 1 else HealthStatus.DEGRADED
            return HealthCheckResult(name="changing", status=status)

        health.register_check("changing", changing_check)

        result1 = health.run_check("changing")
        assert result1.status == HealthStatus.HEALTHY

        result2 = health.run_check("changing")
        assert result2.status == HealthStatus.DEGRADED

        last_results = health.get_last_results()
        assert last_results["changing"].status == HealthStatus.DEGRADED


class TestBuiltInHealthChecks:
    """Tests for built-in health check functions."""

    def test_database_health_check(self):
        """Test database health check."""
        result = database_health_check()

        assert result.name == "database"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
        assert isinstance(result.message, str)

    def test_api_health_check(self):
        """Test API health check."""
        result = api_health_check()

        assert result.name == "api"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
        assert isinstance(result.message, str)

    def test_disk_space_health_check(self):
        """Test disk space health check."""
        result = disk_space_health_check()

        assert result.name == "disk_space"
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert "free" in result.message.lower()
        assert "total_gb" in result.details
        assert "free_percent" in result.details

    def test_memory_health_check(self):
        """Test memory health check."""
        result = memory_health_check()

        assert result.name == "memory"
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert "available" in result.message.lower()
        assert "total_gb" in result.details
        assert "available_percent" in result.details


class TestGlobalHealthCheck:
    """Tests for global health check instance."""

    def test_get_health_check_singleton(self):
        """Test that get_health_check returns singleton."""
        health1 = get_health_check()
        health2 = get_health_check()

        assert health1 is health2

    def test_get_health_check_returns_health_check(self):
        """Test that get_health_check returns HealthCheck."""
        health = get_health_check()

        assert isinstance(health, HealthCheck)

    def test_get_health_check_has_default_checks(self):
        """Test that global instance has default checks registered."""
        health = get_health_check()

        # Should have disk_space and memory checks by default
        assert "disk_space" in health._checks
        assert "memory" in health._checks

    def test_global_health_check_persistence(self):
        """Test that global health check persists data."""
        health = get_health_check()

        def custom_check():
            return HealthCheckResult(name="custom", status=HealthStatus.HEALTHY)

        health.register_check("custom_test", custom_check)

        # Get health check again
        health2 = get_health_check()

        assert "custom_test" in health2._checks


class TestHealthCheckIntegration:
    """Integration tests for health check system."""

    def test_complete_health_check_workflow(self):
        """Test complete health check workflow."""
        health = HealthCheck()

        # Register multiple checks
        def cpu_check():
            return HealthCheckResult(
                name="cpu",
                status=HealthStatus.HEALTHY,
                message="CPU usage normal",
                details={"usage_percent": 45.2},
            )

        def memory_check():
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.DEGRADED,
                message="Memory usage elevated",
                details={"usage_percent": 78.5},
            )

        def disk_check():
            return HealthCheckResult(
                name="disk",
                status=HealthStatus.HEALTHY,
                message="Disk space available",
                details={"free_percent": 45.0},
            )

        health.register_check("cpu", cpu_check)
        health.register_check("memory", memory_check)
        health.register_check("disk", disk_check)

        # Run all checks
        results = health.run_all_checks()

        assert len(results) == 3
        assert all(r.response_time > 0 for r in results.values())

        # Get overall status (should be DEGRADED due to memory)
        overall = health.get_overall_status()
        assert overall == HealthStatus.DEGRADED

        # Get health report
        report = health.get_health_report()

        assert report["overall_status"] == "degraded"
        assert len(report["checks"]) == 3
        assert report["checks"]["cpu"]["status"] == "healthy"
        assert report["checks"]["memory"]["status"] == "degraded"
        assert report["checks"]["disk"]["status"] == "healthy"

    def test_health_check_error_handling(self):
        """Test health check system handles errors gracefully."""
        health = HealthCheck()

        def working_check():
            return HealthCheckResult(name="working", status=HealthStatus.HEALTHY)

        def broken_check():
            raise ValueError("Simulated failure")

        health.register_check("working", working_check)
        health.register_check("broken", broken_check)

        # Should not raise, should handle gracefully
        results = health.run_all_checks()

        assert len(results) == 2
        assert results["working"].status == HealthStatus.HEALTHY
        assert results["broken"].status == HealthStatus.UNHEALTHY
        assert "Simulated failure" in results["broken"].message

        # Overall status should be UNHEALTHY
        overall = health.get_overall_status()
        assert overall == HealthStatus.UNHEALTHY
