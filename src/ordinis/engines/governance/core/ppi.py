"""
PPI Engine - Personal/Private Information Detection and Masking.

Implements OECD AI Principle 2: Respect for human rights and democratic values
- Fairness and privacy protection
- Prevention of unauthorized data exposure

Reference: https://oecd.ai/en/ai-principles
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import re
from typing import Any
import uuid


class PPICategory(Enum):
    """Categories of Personal/Private Information."""

    # Financial identifiers
    SSN = "ssn"
    TAX_ID = "tax_id"
    ACCOUNT_NUMBER = "account_number"
    CREDIT_CARD = "credit_card"
    BANK_ROUTING = "bank_routing"

    # Personal identifiers
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"
    DOB = "date_of_birth"

    # Authentication
    PASSWORD = "password"
    API_KEY = "api_key"
    SECRET = "secret"
    TOKEN = "token"

    # Network identifiers
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"

    # Custom/Other
    CUSTOM = "custom"


class MaskingMethod(Enum):
    """Methods for masking detected PPI."""

    FULL = "full"  # Replace entirely: ****
    PARTIAL = "partial"  # Show partial: ***-**-1234
    HASH = "hash"  # SHA-256 hash
    TOKENIZE = "tokenize"  # Replace with reversible token
    REDACT = "redact"  # Replace with [REDACTED]
    ENCRYPT = "encrypt"  # AES encryption (reversible)


@dataclass
class PPIDetection:
    """Record of detected PPI."""

    detection_id: str
    timestamp: datetime
    category: PPICategory
    field_path: str  # Where in the data structure (e.g., "user.email")
    original_length: int  # Length of original value
    masked_value: str  # Masked/tokenized value
    masking_method: MaskingMethod
    confidence: float  # Detection confidence (0-1)
    pattern_matched: str  # Which pattern detected it
    context: str | None  # Surrounding context for review


@dataclass
class PPIPolicy:
    """Policy for handling specific PPI category."""

    category: PPICategory
    masking_method: MaskingMethod
    log_detection: bool = True
    alert_on_detection: bool = False
    block_transmission: bool = False  # Prevent data from being sent externally
    retention_days: int | None = None  # Auto-delete after N days


class PPIEngine:
    """
    Personal/Private Information detection and masking engine.

    Implements OECD AI Principle 2 for privacy protection.
    Detects and masks sensitive data before logging, storage, or transmission.

    Reference: https://oecd.ai/en/ai-principles
    """

    # Default detection patterns
    DEFAULT_PATTERNS: dict[PPICategory, list[str]] = {
        PPICategory.SSN: [
            r"\b\d{3}-\d{2}-\d{4}\b",  # 123-45-6789
            r"\b\d{9}\b",  # 123456789 (context-dependent)
        ],
        PPICategory.CREDIT_CARD: [
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # 1234-5678-9012-3456
            r"\b\d{16}\b",  # 1234567890123456
        ],
        PPICategory.EMAIL: [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ],
        PPICategory.PHONE: [
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # 123-456-7890
            r"\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b",  # (123) 456-7890
            r"\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # +1-123-456-7890
        ],
        PPICategory.IP_ADDRESS: [
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IPv4
        ],
        PPICategory.API_KEY: [
            r"\b[A-Za-z0-9]{32,}\b",  # Long alphanumeric strings
            r"(?:api[_-]?key|apikey)[=:]\s*['\"]?([A-Za-z0-9_-]+)['\"]?",
            r"(?:secret|token)[=:]\s*['\"]?([A-Za-z0-9_-]+)['\"]?",
        ],
        PPICategory.PASSWORD: [
            r"(?:password|passwd|pwd)[=:]\s*['\"]?([^\s'\"]+)['\"]?",
        ],
        PPICategory.ACCOUNT_NUMBER: [
            r"\b\d{8,17}\b",  # Bank account numbers (context-dependent)
        ],
    }

    # Default policies
    DEFAULT_POLICIES: dict[PPICategory, PPIPolicy] = {
        PPICategory.SSN: PPIPolicy(
            category=PPICategory.SSN,
            masking_method=MaskingMethod.PARTIAL,
            log_detection=True,
            alert_on_detection=True,
            block_transmission=True,
        ),
        PPICategory.CREDIT_CARD: PPIPolicy(
            category=PPICategory.CREDIT_CARD,
            masking_method=MaskingMethod.PARTIAL,
            log_detection=True,
            alert_on_detection=True,
            block_transmission=True,
        ),
        PPICategory.EMAIL: PPIPolicy(
            category=PPICategory.EMAIL,
            masking_method=MaskingMethod.PARTIAL,
            log_detection=True,
            alert_on_detection=False,
        ),
        PPICategory.PHONE: PPIPolicy(
            category=PPICategory.PHONE,
            masking_method=MaskingMethod.PARTIAL,
            log_detection=True,
        ),
        PPICategory.API_KEY: PPIPolicy(
            category=PPICategory.API_KEY,
            masking_method=MaskingMethod.REDACT,
            log_detection=True,
            alert_on_detection=True,
            block_transmission=True,
        ),
        PPICategory.PASSWORD: PPIPolicy(
            category=PPICategory.PASSWORD,
            masking_method=MaskingMethod.REDACT,
            log_detection=True,
            alert_on_detection=True,
            block_transmission=True,
        ),
        PPICategory.IP_ADDRESS: PPIPolicy(
            category=PPICategory.IP_ADDRESS,
            masking_method=MaskingMethod.PARTIAL,
            log_detection=False,
        ),
    }

    def __init__(
        self,
        policies: dict[PPICategory, PPIPolicy] | None = None,
        custom_patterns: dict[PPICategory, list[str]] | None = None,
        token_store: dict[str, str] | None = None,
        encryption_key: bytes | None = None,
    ) -> None:
        """
        Initialize PPI engine.

        Args:
            policies: Custom policies per category
            custom_patterns: Additional detection patterns
            token_store: Storage for tokenization (for reversibility)
            encryption_key: Key for encryption masking method
        """
        self._policies = {**self.DEFAULT_POLICIES, **(policies or {})}
        self._patterns = {**self.DEFAULT_PATTERNS, **(custom_patterns or {})}
        self._token_store = token_store if token_store is not None else {}
        self._reverse_token_store: dict[str, str] = {}
        self._encryption_key = encryption_key

        # Compiled regex patterns for performance
        self._compiled_patterns: dict[PPICategory, list[re.Pattern]] = {}
        self._compile_patterns()

        # Detection history
        self._detections: list[PPIDetection] = []
        self._alert_callbacks: list[Callable[[PPIDetection], None]] = []

    def _compile_patterns(self) -> None:
        """Compile regex patterns for all categories."""
        for category, patterns in self._patterns.items():
            self._compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def add_pattern(self, category: PPICategory, pattern: str) -> None:
        """Add custom detection pattern."""
        if category not in self._patterns:
            self._patterns[category] = []
        self._patterns[category].append(pattern)

        if category not in self._compiled_patterns:
            self._compiled_patterns[category] = []
        self._compiled_patterns[category].append(re.compile(pattern, re.IGNORECASE))

    def set_policy(self, policy: PPIPolicy) -> None:
        """Set or update policy for a category."""
        self._policies[policy.category] = policy

    def register_alert_callback(
        self,
        callback: Callable[[PPIDetection], None],
    ) -> None:
        """Register callback for PPI detection alerts."""
        self._alert_callbacks.append(callback)

    def scan_text(
        self,
        text: str,
        field_path: str = "text",
        categories: list[PPICategory] | None = None,
    ) -> tuple[str, list[PPIDetection]]:
        """
        Scan text for PPI and return masked version.

        Args:
            text: Text to scan
            field_path: Path identifier for logging
            categories: Categories to scan for (None = all)

        Returns:
            Tuple of (masked_text, list of detections)
        """
        detections = []
        masked_text = text

        scan_categories = categories or list(self._compiled_patterns.keys())

        for category in scan_categories:
            if category not in self._compiled_patterns:
                continue

            for pattern in self._compiled_patterns[category]:
                for match in pattern.finditer(text):
                    matched_text = match.group()

                    # Get policy for this category
                    policy = self._policies.get(category)
                    if not policy:
                        policy = PPIPolicy(
                            category=category,
                            masking_method=MaskingMethod.REDACT,
                        )

                    # Mask the value
                    masked_value = self._mask_value(
                        matched_text,
                        policy.masking_method,
                        category,
                    )

                    # Create detection record
                    detection = PPIDetection(
                        detection_id=self._generate_detection_id(),
                        timestamp=datetime.utcnow(),
                        category=category,
                        field_path=field_path,
                        original_length=len(matched_text),
                        masked_value=masked_value,
                        masking_method=policy.masking_method,
                        confidence=self._calculate_confidence(category, matched_text),
                        pattern_matched=pattern.pattern,
                        context=self._get_context(text, match),
                    )

                    detections.append(detection)

                    # Replace in text
                    masked_text = masked_text.replace(matched_text, masked_value, 1)

                    # Log and alert if configured
                    if policy.log_detection:
                        self._detections.append(detection)

                    if policy.alert_on_detection:
                        self._trigger_alerts(detection)

        return masked_text, detections

    def scan_dict(
        self,
        data: dict[str, Any],
        path_prefix: str = "",
        categories: list[PPICategory] | None = None,
    ) -> tuple[dict[str, Any], list[PPIDetection]]:
        """
        Recursively scan dictionary for PPI.

        Args:
            data: Dictionary to scan
            path_prefix: Current path in nested structure
            categories: Categories to scan for

        Returns:
            Tuple of (masked_dict, list of detections)
        """
        all_detections: list[PPIDetection] = []
        masked_data: dict[str, Any] = {}

        for key, value in data.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            if isinstance(value, str):
                masked_str, str_detections = self.scan_text(value, current_path, categories)
                masked_data[key] = masked_str
                all_detections.extend(str_detections)

            elif isinstance(value, dict):
                masked_dict, dict_detections = self.scan_dict(value, current_path, categories)
                masked_data[key] = masked_dict
                all_detections.extend(dict_detections)

            elif isinstance(value, list):
                masked_list: list[Any] = []
                for i, item in enumerate(value):
                    item_path = f"{current_path}[{i}]"
                    if isinstance(item, str):
                        item_str, item_detections = self.scan_text(item, item_path, categories)
                        masked_list.append(item_str)
                        all_detections.extend(item_detections)
                    elif isinstance(item, dict):
                        item_dict, item_detections = self.scan_dict(item, item_path, categories)
                        masked_list.append(item_dict)
                        all_detections.extend(item_detections)
                    else:
                        masked_list.append(item)
                masked_data[key] = masked_list

            else:
                masked_data[key] = value

        return masked_data, all_detections

    def _mask_value(  # noqa: PLR0911
        self,
        value: str,
        method: MaskingMethod,
        category: PPICategory,
    ) -> str:
        """Apply masking method to value."""
        if method == MaskingMethod.FULL:
            return "*" * len(value)

        if method == MaskingMethod.PARTIAL:
            return self._partial_mask(value, category)

        if method == MaskingMethod.HASH:
            return hashlib.sha256(value.encode()).hexdigest()[:16]

        if method == MaskingMethod.TOKENIZE:
            return self._tokenize(value)

        if method == MaskingMethod.REDACT:
            return f"[REDACTED:{category.value.upper()}]"

        if method == MaskingMethod.ENCRYPT:
            return self._encrypt(value)

        return "[MASKED]"

    def _partial_mask(self, value: str, category: PPICategory) -> str:  # noqa: PLR0911
        """Apply partial masking based on category."""
        length = len(value)

        if category == PPICategory.SSN:
            # Show last 4: ***-**-1234
            if length >= 4:
                return f"***-**-{value[-4:]}"

        elif category == PPICategory.CREDIT_CARD:
            # Show last 4: ****-****-****-1234
            if length >= 4:
                return f"****-****-****-{value[-4:]}"

        elif category == PPICategory.EMAIL:
            # Show first char and domain: j***@example.com
            parts = value.split("@")
            if len(parts) == 2:
                local = parts[0]
                domain = parts[1]
                masked_local = local[0] + "*" * (len(local) - 1) if local else "***"
                return f"{masked_local}@{domain}"

        elif category == PPICategory.PHONE:
            # Show last 4: ***-***-1234
            digits = re.sub(r"\D", "", value)
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"

        elif category == PPICategory.IP_ADDRESS:
            # Show first octet: 192.xxx.xxx.xxx
            parts = value.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.xxx.xxx.xxx"

        # Default: show first and last, mask middle
        if length > 4:
            visible = max(1, length // 4)
            return value[:visible] + "*" * (length - 2 * visible) + value[-visible:]

        return "*" * length

    def _tokenize(self, value: str) -> str:
        """Create reversible token for value."""
        # Check if already tokenized
        if value in self._token_store:
            return self._token_store[value]

        token = f"TOK-{uuid.uuid4().hex[:12].upper()}"
        self._token_store[value] = token
        self._reverse_token_store[token] = value
        return token

    def detokenize(self, token: str) -> str | None:
        """Reverse tokenization (requires authorization)."""
        return self._reverse_token_store.get(token)

    def _encrypt(self, value: str) -> str:
        """Encrypt value (placeholder - implement with actual encryption)."""
        if not self._encryption_key:
            return f"[ENCRYPTED:{hashlib.sha256(value.encode()).hexdigest()[:8]}]"

        # Placeholder for actual AES encryption
        # In production, use cryptography library
        return f"[ENCRYPTED:{hashlib.sha256(value.encode()).hexdigest()[:16]}]"

    def _calculate_confidence(self, category: PPICategory, value: str) -> float:
        """Calculate detection confidence based on pattern and context."""
        # Base confidence by category
        base_confidence = {
            PPICategory.SSN: 0.95,
            PPICategory.CREDIT_CARD: 0.90,
            PPICategory.EMAIL: 0.99,
            PPICategory.PHONE: 0.85,
            PPICategory.API_KEY: 0.70,
            PPICategory.PASSWORD: 0.75,
            PPICategory.IP_ADDRESS: 0.80,
            PPICategory.ACCOUNT_NUMBER: 0.60,
        }

        confidence = base_confidence.get(category, 0.50)

        # Adjust based on value characteristics
        if category == PPICategory.CREDIT_CARD:
            # Luhn check would increase confidence
            if self._luhn_check(re.sub(r"\D", "", value)):
                confidence = min(1.0, confidence + 0.05)

        return confidence

    def _luhn_check(self, number: str) -> bool:
        """Luhn algorithm check for credit card validation."""
        if not number.isdigit():
            return False

        digits = [int(d) for d in number]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]

        total = sum(odd_digits)
        for d in even_digits:
            doubled = d * 2
            total += doubled if doubled < 10 else doubled - 9

        return total % 10 == 0

    def _get_context(
        self,
        text: str,
        match: re.Match,
        context_chars: int = 20,
    ) -> str:
        """Extract surrounding context for a match."""
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)

        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    def _generate_detection_id(self) -> str:
        """Generate unique detection ID."""
        return f"PPI-{uuid.uuid4().hex[:12].upper()}"

    def _trigger_alerts(self, detection: PPIDetection) -> None:
        """Trigger registered alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(detection)
            except Exception:  # noqa: S110
                pass  # Isolate callback errors

    def get_detection_summary(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Get summary of PPI detections."""
        detections = self._detections

        if start:
            detections = [d for d in detections if d.timestamp >= start]
        if end:
            detections = [d for d in detections if d.timestamp <= end]

        category_counts: dict[str, int] = {}
        for detection in detections:
            cat = detection.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_detections": len(detections),
            "by_category": category_counts,
            "high_risk_count": sum(
                1
                for d in detections
                if d.category in [PPICategory.SSN, PPICategory.CREDIT_CARD, PPICategory.PASSWORD]
            ),
            "unique_fields": len({d.field_path for d in detections}),
        }

    def should_block_transmission(
        self,
        detections: list[PPIDetection],
    ) -> tuple[bool, list[PPICategory]]:
        """
        Check if transmission should be blocked based on detections.

        Returns:
            Tuple of (should_block, blocking_categories)
        """
        blocking_categories = []

        for detection in detections:
            policy = self._policies.get(detection.category)
            if policy and policy.block_transmission:
                blocking_categories.append(detection.category)

        return len(blocking_categories) > 0, blocking_categories
