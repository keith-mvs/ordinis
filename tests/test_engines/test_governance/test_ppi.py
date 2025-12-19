"""Tests for PPI Engine."""

from ordinis.engines.governance.core.ppi import (
    MaskingMethod,
    PPICategory,
    PPIEngine,
    PPIPolicy,
)


class TestPPIEngine:
    """Tests for PPIEngine."""

    def test_engine_initialization(self) -> None:
        """Test engine initializes with default policies."""
        engine = PPIEngine()

        # Should have default patterns
        assert len(engine._compiled_patterns) > 0
        # Should have default policies
        assert len(engine._policies) > 0

    def test_detect_ssn(self) -> None:
        """Test SSN detection."""
        engine = PPIEngine()

        text = "My SSN is 123-45-6789 for tax purposes."
        masked, detections = engine.scan_text(text)

        assert len(detections) == 1
        assert detections[0].category == PPICategory.SSN
        assert "123-45-6789" not in masked
        assert "6789" in masked  # Partial mask shows last 4

    def test_detect_credit_card(self) -> None:
        """Test credit card detection."""
        engine = PPIEngine()

        text = "Card number: 4111-1111-1111-1111"
        masked, detections = engine.scan_text(text)

        assert len(detections) >= 1
        assert any(d.category == PPICategory.CREDIT_CARD for d in detections)
        assert "4111-1111-1111-1111" not in masked

    def test_detect_email(self) -> None:
        """Test email detection."""
        engine = PPIEngine()

        text = "Contact me at john.doe@example.com"
        masked, detections = engine.scan_text(text)

        assert any(d.category == PPICategory.EMAIL for d in detections)
        assert "john.doe@example.com" not in masked
        assert "@example.com" in masked  # Domain visible

    def test_detect_phone(self) -> None:
        """Test phone number detection."""
        engine = PPIEngine()

        text = "Call me at 555-123-4567"
        masked, detections = engine.scan_text(text)

        assert any(d.category == PPICategory.PHONE for d in detections)
        assert "555-123-4567" not in masked

    def test_detect_api_key(self) -> None:
        """Test API key detection."""
        engine = PPIEngine()

        text = "api_key=sk_test_1234567890abcdefghijklmnop"
        masked, detections = engine.scan_text(text)

        assert any(d.category == PPICategory.API_KEY for d in detections)
        assert "[REDACTED" in masked

    def test_masking_method_full(self) -> None:
        """Test full masking method."""
        engine = PPIEngine()
        engine.set_policy(
            PPIPolicy(
                category=PPICategory.EMAIL,
                masking_method=MaskingMethod.FULL,
            )
        )

        text = "Email: test@test.com"
        masked, _ = engine.scan_text(text, categories=[PPICategory.EMAIL])

        assert "test@test.com" not in masked

    def test_masking_method_hash(self) -> None:
        """Test hash masking method."""
        engine = PPIEngine()
        engine.set_policy(
            PPIPolicy(
                category=PPICategory.EMAIL,
                masking_method=MaskingMethod.HASH,
            )
        )

        text = "Email: test@test.com"
        masked, _ = engine.scan_text(text, categories=[PPICategory.EMAIL])

        assert "test@test.com" not in masked
        # Hash should be deterministic
        masked2, _ = engine.scan_text(text, categories=[PPICategory.EMAIL])
        assert masked == masked2

    def test_masking_method_tokenize(self) -> None:
        """Test tokenization masking method."""
        engine = PPIEngine()
        engine.set_policy(
            PPIPolicy(
                category=PPICategory.EMAIL,
                masking_method=MaskingMethod.TOKENIZE,
            )
        )

        text = "Email: unique@example.com"
        masked, detections = engine.scan_text(text, categories=[PPICategory.EMAIL])

        assert "unique@example.com" not in masked
        assert "TOK-" in masked

        # Should be reversible
        token = detections[0].masked_value
        original = engine.detokenize(token)
        assert original == "unique@example.com"

    def test_scan_dict(self) -> None:
        """Test scanning nested dictionary."""
        engine = PPIEngine()

        data = {
            "user": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "555-123-4567",
            },
            "payment": {
                "card": "4111111111111111",
            },
        }

        masked, detections = engine.scan_dict(data)

        # Original emails/cards should be masked
        assert "john@example.com" not in str(masked)
        assert "4111111111111111" not in str(masked)
        assert len(detections) >= 3  # Email, phone, card

    def test_block_transmission(self) -> None:
        """Test transmission blocking for sensitive PPI."""
        engine = PPIEngine()

        # SSN should trigger block
        text = "SSN: 123-45-6789"
        _, detections = engine.scan_text(text)

        should_block, categories = engine.should_block_transmission(detections)
        assert should_block is True
        assert PPICategory.SSN in categories

    def test_custom_pattern(self) -> None:
        """Test adding custom detection pattern."""
        engine = PPIEngine()

        # Add custom pattern for employee IDs
        engine.add_pattern(PPICategory.CUSTOM, r"EMP-\d{6}")

        text = "Employee ID: EMP-123456"
        _, detections = engine.scan_text(text, categories=[PPICategory.CUSTOM])

        assert len(detections) == 1
        assert detections[0].category == PPICategory.CUSTOM

    def test_detection_confidence(self) -> None:
        """Test detection confidence scoring."""
        engine = PPIEngine()

        # Valid email should have high confidence
        _, detections = engine.scan_text("email@example.com")
        email_detection = [d for d in detections if d.category == PPICategory.EMAIL]
        assert email_detection[0].confidence >= 0.9

    def test_alert_callback(self) -> None:
        """Test alert callback is triggered."""
        engine = PPIEngine()
        alerts = []

        engine.register_alert_callback(lambda d: alerts.append(d))

        # SSN should trigger alert
        engine.scan_text("SSN: 123-45-6789")

        assert len(alerts) > 0

    def test_detection_summary(self) -> None:
        """Test detection summary generation."""
        engine = PPIEngine()

        engine.scan_text("email@test.com")
        engine.scan_text("SSN: 123-45-6789")
        engine.scan_text("Call: 555-123-4567")

        summary = engine.get_detection_summary()

        assert summary["total_detections"] >= 3
        assert "by_category" in summary
        assert summary["high_risk_count"] >= 1  # SSN is high risk

    def test_no_false_positives_common_words(self) -> None:
        """Test that common words aren't flagged as PPI."""
        engine = PPIEngine()

        # These shouldn't trigger PPI detection
        text = "The stock price is $150.50 per share."
        _, detections = engine.scan_text(text)

        # Should not have SSN or credit card detections
        ssn_detections = [d for d in detections if d.category == PPICategory.SSN]
        cc_detections = [d for d in detections if d.category == PPICategory.CREDIT_CARD]

        assert len(ssn_detections) == 0
        assert len(cc_detections) == 0
