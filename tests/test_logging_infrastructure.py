import json
import os

from ordinis.core.logging import TraceContext, get_logger, setup_logging


def test_logging():
    # Setup logging
    setup_logging(log_dir="artifacts/logs_test", level="INFO")

    logger = get_logger("test.logger")

    # Test basic logging
    logger.info("Test message 1")

    # Test structured logging
    logger.info("Test message 2", data={"key": "value"})

    # Test trace context
    with TraceContext() as trace_id:
        logger.info("Test message 3 inside trace")
        assert trace_id is not None

        # Verify trace_id is in the log file later

    # Check file content
    log_file = "artifacts/logs_test/ordinis.jsonl"
    assert os.path.exists(log_file)

    with open(log_file) as f:
        lines = f.readlines()

    print(f"Found {len(lines)} log lines.")

    # Parse lines
    logs = [json.loads(line) for line in lines]

    # Verify message 1
    assert logs[0]["message"] == "Test message 1"
    assert logs[0]["trace_id"] is None

    # Verify message 2
    assert logs[1]["message"] == "Test message 2"
    assert logs[1]["key"] == "value"

    # Verify message 3
    assert logs[2]["message"] == "Test message 3 inside trace"
    assert logs[2]["trace_id"] is not None

    print("Logging test passed!")


if __name__ == "__main__":
    test_logging()
