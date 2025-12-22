import json
import os
import uuid

from ordinis.core.logging import TraceContext, get_logger, setup_logging


def test_logging():
    # Use unique marker to identify test messages
    test_marker = str(uuid.uuid4())[:8]

    # Setup logging
    setup_logging(log_dir="artifacts/logs_test", level="INFO")

    logger = get_logger("ordinis.test.logger")

    # Test basic logging with unique marker
    logger.info(f"Test message 1 [{test_marker}]")

    # Test structured logging
    logger.info(f"Test message 2 [{test_marker}]", data={"key": "value"})

    # Test trace context
    with TraceContext() as trace_id:
        logger.info(f"Test message 3 inside trace [{test_marker}]")
        assert trace_id is not None

    # Check file content
    log_file = "artifacts/logs_test/ordinis.jsonl"
    assert os.path.exists(log_file)

    with open(log_file) as f:
        lines = f.readlines()

    print(f"Found {len(lines)} total log lines.")

    # Parse lines and filter for our test marker
    all_logs = [json.loads(line) for line in lines]
    logs = [log for log in all_logs if test_marker in log.get("message", "")]

    print(f"Found {len(logs)} test log lines with marker {test_marker}.")

    # Verify we got at least 3 messages
    assert len(logs) >= 3, f"Expected at least 3 log messages, got {len(logs)}"

    # Verify message 1
    assert f"Test message 1 [{test_marker}]" in logs[0]["message"]

    # Verify message 2
    assert f"Test message 2 [{test_marker}]" in logs[1]["message"]
    assert logs[1].get("key") == "value"

    # Verify message 3
    assert f"Test message 3 inside trace [{test_marker}]" in logs[2]["message"]
    assert logs[2]["trace_id"] is not None

    print("Logging test passed!")


if __name__ == "__main__":
    test_logging()
