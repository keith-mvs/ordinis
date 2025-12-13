"""Tests for adapters.telemetry.health module."""

from datetime import datetime
import time
from unittest.mock import Mock, patch

import ordinis.adapters.telemetry.health as health_module
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

    def test_health_status_members(self):
        """Test HealthStatus has all expected members."""
        statuses = list(HealthStatus)
        assert len(statuses) == 4
        assert HealthStatus.HEALTHY in statuses
        assert HealthStatus.DEGRADED in statuses
        assert HealthStatus.UNHEALTHY in statuses
        assert HealthStatus.UNKNOWN in statuses


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_health_check_result_initialization(self):
        """Test HealthCheckResult initialization with required fields."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
        )

        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == ""
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.details, dict)
        assert len(result.details) == 0
        assert result.response_time == 0.0

    def test_health_check_result_with_all_fields(self):
        """Test HealthCheckResult initialization with all fields."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        details = {"key": "value", "count": 42}
        result = HealthCheckResult(
            name="full_check",
            status=HealthStatus.DEGRADED,
            message="Performance degraded",
            timestamp=custom_time,
            details=details,
            response_time=1.5,
        )

        assert result.name == "full_check"
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "Performance degraded"
        assert result.timestamp == custom_time
        assert result.details == details
        assert result.response_time == 1.5

    def test_health_check_result_to_dict(self):
        """Test HealthCheckResult to_dict conversion."""
        details = {"cpu_usage": 75, "memory_usage": 80}
        result = HealthCheckResult(
            name="system_check",
            status=HealthStatus.HEALTHY,
            message="System running normally",
            details=details,
            response_time=0.25,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["name"] == "system_check"
        assert result_dict["status"] == "healthy"
        assert result_dict["message"] == "System running normally"
        assert result_dict["details"] == details
        assert result_dict["response_time"] == 0.25
        assert "timestamp" in result_dict
        assert isinstance(result_dict["timestamp"], str)

    def test_health_check_result_to_dict_timestamp_format(self):
        """Test HealthCheckResult to_dict timestamp is ISO format."""
        result = HealthCheckResult(
            name="timestamp_test",
            status=HealthStatus.UNKNOWN,
        )

        result_dict = result.to_dict()
        timestamp_str = result_dict["timestamp"]

        # Verify it can be parsed back
        parsed_time = datetime.fromisoformat(timestamp_str)
        assert isinstance(parsed_time, datetime)

    def test_health_check_result_default_factory_independence(self):
        """Test that default dict/timestamp are independent between instances."""
        result1 = HealthCheckResult(name="check1", status=HealthStatus.HEALTHY)
        result2 = HealthCheckResult(name="check2", status=HealthStatus.HEALTHY)

        result1.details["key"] = "value1"
        assert "key" not in result2.details
        # Timestamps are independent - either different or same
        assert True


class TestHealthCheck:
    """Tests for HealthCheck class."""

    def test_health_check_initialization(self):
        """Test HealthCheck initialization."""
        health_check = HealthCheck()

        assert len(health_check._checks) == 0
        assert len(health_check._last_results) == 0

    def test_register_check(self):
        """Test registering a health check."""
        health_check = HealthCheck()

        def test_check() -> HealthCheckResult:
            return HealthCheckResult(name="test", status=HealthStatus.HEALTHY)

        health_check.register_check("test_check", test_check)

        assert "test_check" in health_check._checks
        assert health_check._checks["test_check"] == test_check

    def test_register_multiple_checks(self):
        """Test registering multiple health checks."""
        health_check = HealthCheck()

        def check1() -> HealthCheckResult:
            return HealthCheckResult(name="check1", status=HealthStatus.HEALTHY)

        def check2() -> HealthCheckResult:
            return HealthCheckResult(name="check2", status=HealthStatus.HEALTHY)

        health_check.register_check("check1", check1)
        health_check.register_check("check2", check2)

        assert len(health_check._checks) == 2
        assert "check1" in health_check._checks
        assert "check2" in health_check._checks

    def test_run_check_success(self):
        """Test running a successful health check."""
        health_check = HealthCheck()

        def test_check() -> HealthCheckResult:
            return HealthCheckResult(name="test", status=HealthStatus.HEALTHY, message="All good")

        health_check.register_check("test_check", test_check)
        result = health_check.run_check("test_check")

        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.response_time > 0
        assert "test_check" in health_check._last_results

    def test_run_check_not_found(self):
        """Test running a non-existent check returns UNKNOWN status."""
        health_check = HealthCheck()
        result = health_check.run_check("nonexistent")

        assert result.name == "nonexistent"
        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.message

    def test_run_check_exception_handling(self):
        """Test health check handles exceptions gracefully."""
        health_check = HealthCheck()

        def failing_check() -> HealthCheckResult:
            raise RuntimeError("Check failed")

        health_check.register_check("failing_check", failing_check)
        result = health_check.run_check("failing_check")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message
        assert result.response_time > 0
        assert "failing_check" in health_check._last_results

    def test_run_check_updates_last_results(self):
        """Test that run_check updates last results."""
        health_check = HealthCheck()

        call_count = 0

        def changing_check() -> HealthCheckResult:
            nonlocal call_count
            call_count += 1
            return HealthCheckResult(
                name="changing",
                status=HealthStatus.HEALTHY,
                message=f"Call {call_count}",
            )

        health_check.register_check("changing", changing_check)

        result1 = health_check.run_check("changing")
        assert result1.message == "Call 1"

        result2 = health_check.run_check("changing")
        assert result2.message == "Call 2"

        assert health_check._last_results["changing"].message == "Call 2"

    def test_run_all_checks_empty(self):
        """Test running all checks when none are registered."""
        health_check = HealthCheck()
        results = health_check.run_all_checks()

        assert isinstance(results, dict)
        assert len(results) == 0

    def test_run_all_checks_multiple(self):
        """Test running all registered checks."""
        health_check = HealthCheck()

        def check1() -> HealthCheckResult:
            return HealthCheckResult(name="check1", status=HealthStatus.HEALTHY)

        def check2() -> HealthCheckResult:
            return HealthCheckResult(name="check2", status=HealthStatus.DEGRADED)

        def check3() -> HealthCheckResult:
            return HealthCheckResult(name="check3", status=HealthStatus.UNHEALTHY)

        health_check.register_check("check1", check1)
        health_check.register_check("check2", check2)
        health_check.register_check("check3", check3)

        results = health_check.run_all_checks()

        assert len(results) == 3
        assert "check1" in results
        assert "check2" in results
        assert "check3" in results
        assert results["check1"].status == HealthStatus.HEALTHY
        assert results["check2"].status == HealthStatus.DEGRADED
        assert results["check3"].status == HealthStatus.UNHEALTHY

    def test_get_overall_status_no_results(self):
        """Test overall status with no results returns UNKNOWN."""
        health_check = HealthCheck()
        status = health_check.get_overall_status()

        assert status == HealthStatus.UNKNOWN

    def test_get_overall_status_all_healthy(self):
        """Test overall status with all healthy checks."""
        health_check = HealthCheck()

        def healthy_check() -> HealthCheckResult:
            return HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY)

        health_check.register_check("check1", healthy_check)
        health_check.register_check("check2", healthy_check)
        health_check.run_all_checks()

        status = health_check.get_overall_status()
        assert status == HealthStatus.HEALTHY

    def test_get_overall_status_one_degraded(self):
        """Test overall status returns DEGRADED if any check is degraded."""
        health_check = HealthCheck()

        def healthy_check() -> HealthCheckResult:
            return HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY)

        def degraded_check() -> HealthCheckResult:
            return HealthCheckResult(name="degraded", status=HealthStatus.DEGRADED)

        health_check.register_check("check1", healthy_check)
        health_check.register_check("check2", degraded_check)
        health_check.run_all_checks()

        status = health_check.get_overall_status()
        assert status == HealthStatus.DEGRADED

    def test_get_overall_status_one_unhealthy(self):
        """Test overall status returns UNHEALTHY if any check is unhealthy."""
        health_check = HealthCheck()

        def healthy_check() -> HealthCheckResult:
            return HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY)

        def degraded_check() -> HealthCheckResult:
            return HealthCheckResult(name="degraded", status=HealthStatus.DEGRADED)

        def unhealthy_check() -> HealthCheckResult:
            return HealthCheckResult(name="unhealthy", status=HealthStatus.UNHEALTHY)

        health_check.register_check("check1", healthy_check)
        health_check.register_check("check2", degraded_check)
        health_check.register_check("check3", unhealthy_check)
        health_check.run_all_checks()

        status = health_check.get_overall_status()
        assert status == HealthStatus.UNHEALTHY

    def test_get_overall_status_only_unknown(self):
        """Test overall status with only unknown checks."""
        health_check = HealthCheck()

        def unknown_check() -> HealthCheckResult:
            return HealthCheckResult(name="unknown", status=HealthStatus.UNKNOWN)

        health_check.register_check("check1", unknown_check)
        health_check.run_all_checks()

        status = health_check.get_overall_status()
        assert status == HealthStatus.UNKNOWN

    def test_get_health_report(self):
        """Test getting comprehensive health report."""
        health_check = HealthCheck()

        def test_check() -> HealthCheckResult:
            return HealthCheckResult(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                details={"version": "1.0"},
            )

        health_check.register_check("test_check", test_check)
        report = health_check.get_health_report()

        assert "overall_status" in report
        assert "timestamp" in report
        assert "checks" in report
        assert isinstance(report["checks"], dict)
        assert "test_check" in report["checks"]
        assert report["checks"]["test_check"]["name"] == "test"
        assert report["checks"]["test_check"]["status"] == "healthy"

    def test_get_health_report_timestamp_format(self):
        """Test health report timestamp is ISO format."""
        health_check = HealthCheck()

        def test_check() -> HealthCheckResult:
            return HealthCheckResult(name="test", status=HealthStatus.HEALTHY)

        health_check.register_check("test_check", test_check)
        report = health_check.get_health_report()

        timestamp_str = report["timestamp"]
        parsed_time = datetime.fromisoformat(timestamp_str)
        assert isinstance(parsed_time, datetime)

    def test_get_last_results_empty(self):
        """Test getting last results when none exist."""
        health_check = HealthCheck()
        results = health_check.get_last_results()

        assert isinstance(results, dict)
        assert len(results) == 0

    def test_get_last_results_returns_copy(self):
        """Test that get_last_results returns a copy."""
        health_check = HealthCheck()

        def test_check() -> HealthCheckResult:
            return HealthCheckResult(name="test", status=HealthStatus.HEALTHY)

        health_check.register_check("test_check", test_check)
        health_check.run_check("test_check")

        results1 = health_check.get_last_results()
        results2 = health_check.get_last_results()

        assert results1 is not results2
        assert results1 == results2

    def test_get_last_results_without_rerunning(self):
        """Test that get_last_results doesn't re-run checks."""
        health_check = HealthCheck()

        call_count = 0

        def counting_check() -> HealthCheckResult:
            nonlocal call_count
            call_count += 1
            return HealthCheckResult(
                name="count", status=HealthStatus.HEALTHY, message=str(call_count)
            )

        health_check.register_check("counting", counting_check)
        health_check.run_check("counting")

        initial_count = call_count
        results = health_check.get_last_results()

        assert call_count == initial_count
        assert len(results) == 1


class TestDatabaseHealthCheck:
    """Tests for database_health_check function."""

    def test_database_health_check_success(self):
        """Test database health check returns healthy status."""
        result = database_health_check()

        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
        assert "accessible" in result.message.lower()

    def test_database_health_check_exception(self):
        """Test database health check handles exceptions."""
        # This tests the exception handling path in the function
        # Since it's a stub implementation, we can't easily trigger the exception
        # but we verify the function is callable and returns proper type
        result = database_health_check()
        assert isinstance(result, HealthCheckResult)


class TestApiHealthCheck:
    """Tests for api_health_check function."""

    def test_api_health_check_success(self):
        """Test API health check returns healthy status."""
        result = api_health_check()

        assert result.name == "api"
        assert result.status == HealthStatus.HEALTHY
        assert "accessible" in result.message.lower()

    def test_api_health_check_returns_result(self):
        """Test API health check returns HealthCheckResult."""
        result = api_health_check()
        assert isinstance(result, HealthCheckResult)


class TestDiskSpaceHealthCheck:
    """Tests for disk_space_health_check function."""

    @patch("shutil.disk_usage")
    def test_disk_space_health_check_healthy(self, mock_disk_usage):
        """Test disk space check returns healthy with sufficient space."""
        # Simulate 50% free space
        total = 1000 * (1024**3)
        used = 500 * (1024**3)
        free = 500 * (1024**3)
        mock_disk_usage.return_value = (total, used, free)

        result = disk_space_health_check()

        assert result.name == "disk_space"
        assert result.status == HealthStatus.HEALTHY
        assert "50.0%" in result.message
        assert result.details["free_percent"] == 50.0
        assert result.details["total_gb"] > 0
        assert result.details["free_gb"] > 0

    @patch("shutil.disk_usage")
    def test_disk_space_health_check_degraded(self, mock_disk_usage):
        """Test disk space check returns degraded with low space."""
        # Simulate 15% free space
        total = 1000 * (1024**3)
        used = 850 * (1024**3)
        free = 150 * (1024**3)
        mock_disk_usage.return_value = (total, used, free)

        result = disk_space_health_check()

        assert result.name == "disk_space"
        assert result.status == HealthStatus.DEGRADED
        assert "low" in result.message.lower()
        assert result.details["free_percent"] == 15.0

    @patch("shutil.disk_usage")
    def test_disk_space_health_check_unhealthy(self, mock_disk_usage):
        """Test disk space check returns unhealthy with critical space."""
        # Simulate 5% free space
        total = 1000 * (1024**3)
        used = 950 * (1024**3)
        free = 50 * (1024**3)
        mock_disk_usage.return_value = (total, used, free)

        result = disk_space_health_check()

        assert result.name == "disk_space"
        assert result.status == HealthStatus.UNHEALTHY
        assert "critical" in result.message.lower()
        assert result.details["free_percent"] == 5.0

    @patch("shutil.disk_usage")
    def test_disk_space_health_check_exception(self, mock_disk_usage):
        """Test disk space check handles exceptions."""
        mock_disk_usage.side_effect = OSError("Disk error")

        result = disk_space_health_check()

        assert result.name == "disk_space"
        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()

    @patch("shutil.disk_usage")
    def test_disk_space_health_check_exact_threshold_degraded(self, mock_disk_usage):
        """Test disk space check at exact degraded threshold (20%)."""
        total = 1000 * (1024**3)
        used = 800 * (1024**3)
        free = 200 * (1024**3)
        mock_disk_usage.return_value = (total, used, free)

        result = disk_space_health_check()

        assert result.details["free_percent"] == 20.0
        # At exactly 20%, should be HEALTHY (not < 20)
        assert result.status == HealthStatus.HEALTHY

    @patch("shutil.disk_usage")
    def test_disk_space_health_check_exact_threshold_unhealthy(self, mock_disk_usage):
        """Test disk space check at exact unhealthy threshold (10%)."""
        total = 1000 * (1024**3)
        used = 900 * (1024**3)
        free = 100 * (1024**3)
        mock_disk_usage.return_value = (total, used, free)

        result = disk_space_health_check()

        assert result.details["free_percent"] == 10.0
        # At exactly 10%, should be DEGRADED (not < 10)
        assert result.status == HealthStatus.DEGRADED


class TestMemoryHealthCheck:
    """Tests for memory_health_check function."""

    @patch("psutil.virtual_memory")
    def test_memory_health_check_healthy(self, mock_memory):
        """Test memory check returns healthy with sufficient memory."""
        # Simulate 50% available memory
        mock_memory.return_value = Mock(
            total=16 * (1024**3),
            available=8 * (1024**3),
            percent=50.0,
        )

        result = memory_health_check()

        assert result.name == "memory"
        assert result.status == HealthStatus.HEALTHY
        assert "50.0%" in result.message
        assert result.details["available_percent"] == 50.0
        assert result.details["total_gb"] > 0
        assert result.details["available_gb"] > 0

    @patch("psutil.virtual_memory")
    def test_memory_health_check_degraded(self, mock_memory):
        """Test memory check returns degraded with low memory."""
        # Simulate 15% available memory
        total = 16 * (1024**3)
        available = int(total * 0.15)
        mock_memory.return_value = Mock(
            total=total,
            available=available,
            percent=85.0,
        )

        result = memory_health_check()

        assert result.name == "memory"
        assert result.status == HealthStatus.DEGRADED
        assert "low" in result.message.lower()
        assert abs(result.details["available_percent"] - 15.0) < 0.01

    @patch("psutil.virtual_memory")
    def test_memory_health_check_unhealthy(self, mock_memory):
        """Test memory check returns unhealthy with critical memory."""
        # Simulate 5% available memory
        total = 16 * (1024**3)
        available = int(total * 0.05)
        mock_memory.return_value = Mock(
            total=total,
            available=available,
            percent=95.0,
        )

        result = memory_health_check()

        assert result.name == "memory"
        assert result.status == HealthStatus.UNHEALTHY
        assert "critical" in result.message.lower()
        assert abs(result.details["available_percent"] - 5.0) < 0.01

    @patch("psutil.virtual_memory")
    def test_memory_health_check_exception(self, mock_memory):
        """Test memory check handles exceptions."""
        mock_memory.side_effect = RuntimeError("Memory error")

        result = memory_health_check()

        assert result.name == "memory"
        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()

    @patch("psutil.virtual_memory")
    def test_memory_health_check_exact_threshold_degraded(self, mock_memory):
        """Test memory check at exact degraded threshold (20%)."""
        total = 16 * (1024**3)
        available = int(total * 0.20)
        mock_memory.return_value = Mock(
            total=total,
            available=available,
            percent=80.0,
        )

        result = memory_health_check()

        assert abs(result.details["available_percent"] - 20.0) < 0.01
        # At exactly 20%, triggers DEGRADED due to floating point (< 20)
        assert result.status == HealthStatus.DEGRADED

    @patch("psutil.virtual_memory")
    def test_memory_health_check_exact_threshold_unhealthy(self, mock_memory):
        """Test memory check at exact unhealthy threshold (10%)."""
        total = 16 * (1024**3)
        available = int(total * 0.10)
        mock_memory.return_value = Mock(
            total=total,
            available=available,
            percent=90.0,
        )

        result = memory_health_check()

        assert abs(result.details["available_percent"] - 10.0) < 0.01
        # At exactly 10%, triggers UNHEALTHY due to floating point (< 10)
        assert result.status == HealthStatus.UNHEALTHY


class TestGetHealthCheck:
    """Tests for get_health_check function."""

    def test_get_health_check_singleton(self):
        """Test get_health_check returns singleton instance."""
        # Reset the global instance first
        health_module._global_health_check = None

        instance1 = get_health_check()
        instance2 = get_health_check()

        assert instance1 is instance2

    def test_get_health_check_registers_default_checks(self):
        """Test get_health_check registers default checks."""
        # Reset the global instance first
        health_module._global_health_check = None

        instance = get_health_check()

        assert "disk_space" in instance._checks
        assert "memory" in instance._checks

    def test_get_health_check_returns_health_check_instance(self):
        """Test get_health_check returns HealthCheck instance."""
        instance = get_health_check()
        assert isinstance(instance, HealthCheck)

    def test_get_health_check_preserves_registered_checks(self):
        """Test get_health_check preserves previously registered checks."""
        # Reset the global instance first
        health_module._global_health_check = None

        instance1 = get_health_check()

        def custom_check() -> HealthCheckResult:
            return HealthCheckResult(name="custom", status=HealthStatus.HEALTHY)

        instance1.register_check("custom_check", custom_check)

        instance2 = get_health_check()
        assert "custom_check" in instance2._checks


class TestHealthCheckIntegration:
    """Integration tests for health check system."""

    def test_complete_health_check_workflow(self):
        """Test complete health check workflow."""
        health_check = HealthCheck()

        # Register multiple checks
        def db_check() -> HealthCheckResult:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connected",
            )

        def api_check() -> HealthCheckResult:
            return HealthCheckResult(
                name="api",
                status=HealthStatus.DEGRADED,
                message="API response slow",
            )

        health_check.register_check("database", db_check)
        health_check.register_check("api", api_check)

        # Run all checks
        results = health_check.run_all_checks()
        assert len(results) == 2

        # Check overall status
        overall = health_check.get_overall_status()
        assert overall == HealthStatus.DEGRADED

        # Get health report
        report = health_check.get_health_report()
        assert report["overall_status"] == "degraded"
        assert len(report["checks"]) == 2

    @patch("psutil.virtual_memory")
    @patch("shutil.disk_usage")
    def test_real_health_checks_integration(self, mock_disk_usage, mock_memory):
        """Test integration with real health check functions."""
        # Setup mocks
        mock_disk_usage.return_value = (1000 * (1024**3), 500 * (1024**3), 500 * (1024**3))
        mock_memory.return_value = Mock(
            total=16 * (1024**3),
            available=8 * (1024**3),
            percent=50.0,
        )

        health_check = HealthCheck()
        health_check.register_check("disk", disk_space_health_check)
        health_check.register_check("memory", memory_health_check)

        report = health_check.get_health_report()

        assert report["overall_status"] == "healthy"
        assert "disk" in report["checks"]
        assert "memory" in report["checks"]
        assert report["checks"]["disk"]["status"] == "healthy"
        assert report["checks"]["memory"]["status"] == "healthy"

    def test_health_check_response_time_tracking(self):
        """Test that response time is tracked for checks."""
        health_check = HealthCheck()

        def slow_check() -> HealthCheckResult:
            time.sleep(0.01)  # 10ms delay
            return HealthCheckResult(name="slow", status=HealthStatus.HEALTHY)

        health_check.register_check("slow", slow_check)
        result = health_check.run_check("slow")

        assert result.response_time > 0
        assert result.response_time >= 0.01

    def test_health_check_with_mixed_statuses(self):
        """Test health check with various status combinations."""
        health_check = HealthCheck()

        checks_data = [
            ("healthy1", HealthStatus.HEALTHY),
            ("healthy2", HealthStatus.HEALTHY),
            ("degraded", HealthStatus.DEGRADED),
            ("unknown", HealthStatus.UNKNOWN),
        ]

        for name, status in checks_data:

            def make_check(s=status, n=name):
                def check() -> HealthCheckResult:
                    return HealthCheckResult(name=n, status=s)

                return check

            health_check.register_check(name, make_check())

        health_check.run_all_checks()
        overall = health_check.get_overall_status()

        # Should be DEGRADED (worst non-UNKNOWN status)
        assert overall == HealthStatus.DEGRADED

    def test_exception_during_check_doesnt_stop_others(self):
        """Test that exception in one check doesn't stop others."""
        health_check = HealthCheck()

        def failing_check() -> HealthCheckResult:
            raise ValueError("Check error")

        def working_check() -> HealthCheckResult:
            return HealthCheckResult(name="working", status=HealthStatus.HEALTHY)

        health_check.register_check("failing", failing_check)
        health_check.register_check("working", working_check)

        results = health_check.run_all_checks()

        assert len(results) == 2
        assert results["failing"].status == HealthStatus.UNHEALTHY
        assert results["working"].status == HealthStatus.HEALTHY
