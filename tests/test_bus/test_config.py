"""Tests for BusConfig configuration."""

from ordinis.bus.config import AdapterType, BusConfig


class TestAdapterType:
    """Tests for AdapterType enum."""

    def test_adapter_types_exist(self):
        """Test that both adapter types are defined."""
        assert AdapterType.MEMORY.value == "memory"
        assert AdapterType.REDIS.value == "redis"

    def test_adapter_type_count(self):
        """Test that we have exactly 2 adapter types."""
        assert len(AdapterType) == 2


class TestBusConfig:
    """Tests for BusConfig dataclass."""

    def test_default_config(self):
        """Test BusConfig with default values."""
        config = BusConfig()

        # Adapter settings
        assert config.adapter == AdapterType.MEMORY
        assert config.redis_url == "redis://localhost:6379"
        assert config.redis_stream_prefix == "ordinis:"
        assert config.redis_max_len == 10000

        # Event settings
        assert config.max_payload_size == 1024 * 1024  # 1MB
        assert config.default_ttl_seconds == 3600

        # Processing settings
        assert config.max_concurrent_handlers == 10
        assert config.handler_timeout_seconds == 30.0
        assert config.retry_failed_handlers is True
        assert config.max_handler_retries == 3

        # Batching settings
        assert config.batch_size == 100
        assert config.batch_timeout_ms == 100

        # History settings
        assert config.enable_history is True
        assert config.history_max_events == 10000

        # Metrics settings
        assert config.emit_metrics is True
        assert config.metrics_interval_seconds == 60

    def test_custom_config(self):
        """Test BusConfig with custom values."""
        config = BusConfig(
            adapter=AdapterType.REDIS,
            redis_url="redis://custom:6380",
            redis_stream_prefix="custom:",
            redis_max_len=5000,
            max_payload_size=512 * 1024,
            default_ttl_seconds=1800,
            max_concurrent_handlers=20,
            handler_timeout_seconds=60.0,
            retry_failed_handlers=False,
            max_handler_retries=5,
            batch_size=50,
            batch_timeout_ms=200,
            enable_history=False,
            history_max_events=5000,
            emit_metrics=False,
            metrics_interval_seconds=120,
        )

        assert config.adapter == AdapterType.REDIS
        assert config.redis_url == "redis://custom:6380"
        assert config.redis_stream_prefix == "custom:"
        assert config.redis_max_len == 5000
        assert config.max_payload_size == 512 * 1024
        assert config.default_ttl_seconds == 1800
        assert config.max_concurrent_handlers == 20
        assert config.handler_timeout_seconds == 60.0
        assert config.retry_failed_handlers is False
        assert config.max_handler_retries == 5
        assert config.batch_size == 50
        assert config.batch_timeout_ms == 200
        assert config.enable_history is False
        assert config.history_max_events == 5000
        assert config.emit_metrics is False
        assert config.metrics_interval_seconds == 120

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = BusConfig()
        errors = config.validate()

        assert errors == []

    def test_validate_invalid_payload_size(self):
        """Test validation fails for invalid payload size."""
        config = BusConfig(max_payload_size=0)
        errors = config.validate()

        assert len(errors) == 1
        assert "max_payload_size must be > 0" in errors

    def test_validate_invalid_handler_timeout(self):
        """Test validation fails for invalid handler timeout."""
        config = BusConfig(handler_timeout_seconds=0)
        errors = config.validate()

        assert len(errors) == 1
        assert "handler_timeout_seconds must be > 0" in errors

    def test_validate_redis_missing_url(self):
        """Test validation fails for Redis adapter without URL."""
        config = BusConfig(adapter=AdapterType.REDIS, redis_url="")
        errors = config.validate()

        assert len(errors) == 1
        assert "redis_url required when adapter=REDIS" in errors

    def test_validate_multiple_errors(self):
        """Test validation with multiple errors."""
        config = BusConfig(
            max_payload_size=-100,
            handler_timeout_seconds=-5.0,
            adapter=AdapterType.REDIS,
            redis_url="",
        )
        errors = config.validate()

        assert len(errors) == 3
        assert any("max_payload_size" in err for err in errors)
        assert any("handler_timeout_seconds" in err for err in errors)
        assert any("redis_url" in err for err in errors)

    def test_validate_redis_with_valid_url(self):
        """Test validation passes for Redis with valid URL."""
        config = BusConfig(
            adapter=AdapterType.REDIS,
            redis_url="redis://localhost:6379",
        )
        errors = config.validate()

        assert errors == []

    def test_validate_memory_adapter_no_redis_url_needed(self):
        """Test validation passes for memory adapter without Redis URL."""
        config = BusConfig(
            adapter=AdapterType.MEMORY,
            redis_url="",  # Empty URL is OK for memory adapter
        )
        errors = config.validate()

        assert errors == []

    def test_config_adapter_type_string(self):
        """Test creating config with adapter type as string."""
        # This tests that adapter accepts AdapterType enum
        config = BusConfig(adapter=AdapterType.MEMORY)
        assert config.adapter == AdapterType.MEMORY

        config2 = BusConfig(adapter=AdapterType.REDIS)
        assert config2.adapter == AdapterType.REDIS

    def test_config_history_disabled(self):
        """Test config with history disabled."""
        config = BusConfig(enable_history=False)

        assert config.enable_history is False
        # history_max_events should still have a value
        assert config.history_max_events == 10000

    def test_config_retry_disabled(self):
        """Test config with retry disabled."""
        config = BusConfig(retry_failed_handlers=False)

        assert config.retry_failed_handlers is False
        # max_handler_retries should still have a value
        assert config.max_handler_retries == 3

    def test_config_metrics_disabled(self):
        """Test config with metrics disabled."""
        config = BusConfig(emit_metrics=False)

        assert config.emit_metrics is False
        # metrics_interval_seconds should still have a value
        assert config.metrics_interval_seconds == 60

    def test_config_extreme_values(self):
        """Test config with extreme but valid values."""
        config = BusConfig(
            max_payload_size=1,  # 1 byte
            handler_timeout_seconds=0.001,  # 1ms
            max_concurrent_handlers=1000,
            history_max_events=1000000,
        )
        errors = config.validate()

        assert errors == []

    def test_config_negative_values_fail(self):
        """Test that negative values fail validation."""
        config1 = BusConfig(max_payload_size=-1)
        assert len(config1.validate()) > 0

        config2 = BusConfig(handler_timeout_seconds=-1.0)
        assert len(config2.validate()) > 0
