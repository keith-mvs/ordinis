"""
Ethics Engine - OECD AI Principles Implementation.

Implements all five OECD AI Principles (2024 Update):
1. Inclusive growth, sustainable development, and human well-being
2. Respect for rule of law, human rights, and democratic values
3. Transparency and explainability
4. Robustness, security, and safety
5. Accountability (traceability and risk management)

Reference: https://oecd.ai/en/ai-principles
Additional: https://www.oecd.org/en/topics/ai-principles.html
"""

from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class OECDPrinciple(Enum):
    """
    OECD AI Principles (2024 Update).

    Reference: https://oecd.ai/en/ai-principles
    """

    # Principle 1: Inclusive growth, sustainable development, and well-being
    INCLUSIVE_GROWTH = "inclusive_growth"
    SUSTAINABLE_DEVELOPMENT = "sustainable_development"
    HUMAN_WELLBEING = "human_wellbeing"

    # Principle 2: Human rights and democratic values
    HUMAN_RIGHTS = "human_rights"
    FAIRNESS = "fairness"
    PRIVACY = "privacy"
    NON_DISCRIMINATION = "non_discrimination"

    # Principle 3: Transparency and explainability
    TRANSPARENCY = "transparency"
    EXPLAINABILITY = "explainability"
    DISCLOSURE = "disclosure"

    # Principle 4: Robustness, security, and safety
    ROBUSTNESS = "robustness"
    SECURITY = "security"
    SAFETY = "safety"

    # Principle 5: Accountability
    ACCOUNTABILITY = "accountability"
    TRACEABILITY = "traceability"
    RISK_MANAGEMENT = "risk_management"
    HUMAN_OVERSIGHT = "human_oversight"


class ESGCategory(Enum):
    """ESG (Environmental, Social, Governance) categories."""

    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"


class ManipulationType(Enum):
    """Types of market manipulation to detect."""

    SPOOFING = "spoofing"  # Fake orders to move price
    LAYERING = "layering"  # Multiple fake orders
    WASH_TRADING = "wash_trading"  # Trading with self
    PUMP_AND_DUMP = "pump_and_dump"
    FRONT_RUNNING = "front_running"
    MOMENTUM_IGNITION = "momentum_ignition"


@dataclass
class EthicsCheckResult:
    """Result of an ethics/principle check."""

    check_id: str
    timestamp: datetime
    principle: OECDPrinciple
    passed: bool
    score: float  # 0.0 to 1.0 (1.0 = fully compliant)
    threshold: float  # Minimum required score
    details: dict[str, Any]
    recommendations: list[str]
    requires_human_review: bool = False
    blocking: bool = False  # If True, action should be prevented


@dataclass
class ESGScore:
    """ESG score for a security or action."""

    symbol: str
    environmental: float  # 0-100
    social: float  # 0-100
    governance: float  # 0-100
    overall: float  # Weighted average
    data_source: str
    last_updated: datetime
    controversies: list[str] = field(default_factory=list)


@dataclass
class EthicsPolicy:
    """Policy for ethics enforcement."""

    principle: OECDPrinciple
    enabled: bool = True
    threshold: float = 0.7  # Minimum compliance score
    blocking: bool = False  # Block action on violation
    human_review_required: bool = False
    description: str = ""


class EthicsEngine:
    """
    Ethics Engine implementing OECD AI Principles.

    Ensures automated trading decisions comply with:
    - OECD AI Principles (2024)
    - ESG investment criteria
    - Market manipulation prevention
    - Fair and transparent operation

    Reference: https://oecd.ai/en/ai-principles
    """

    # Excluded sectors (default)
    DEFAULT_EXCLUDED_SECTORS = [
        "tobacco",
        "controversial_weapons",
        "thermal_coal",
        "private_prisons",
    ]

    # Default ESG thresholds
    DEFAULT_ESG_THRESHOLDS = {
        "minimum_overall": 40.0,
        "minimum_environmental": 30.0,
        "minimum_social": 30.0,
        "minimum_governance": 40.0,
        "controversy_limit": 3,
    }

    def __init__(
        self,
        policies: dict[OECDPrinciple, EthicsPolicy] | None = None,
        excluded_sectors: list[str] | None = None,
        esg_thresholds: dict[str, float] | None = None,
        esg_data_provider: Any = None,
    ) -> None:
        """
        Initialize ethics engine.

        Args:
            policies: Custom policies for each principle
            excluded_sectors: Sectors to exclude from trading
            esg_thresholds: ESG score thresholds
            esg_data_provider: Provider for ESG data
        """
        self._policies = self._initialize_policies(policies)
        self._excluded_sectors = excluded_sectors or self.DEFAULT_EXCLUDED_SECTORS
        self._esg_thresholds = esg_thresholds or self.DEFAULT_ESG_THRESHOLDS
        self._esg_provider = esg_data_provider

        # ESG cache
        self._esg_cache: dict[str, ESGScore] = {}

        # Manipulation detection state
        self._recent_orders: list[dict] = []
        self._order_patterns: dict[str, list[dict]] = {}

        # Human review queue
        self._review_queue: list[dict] = []

        # Check history
        self._check_history: list[EthicsCheckResult] = []

        # Callbacks
        self._violation_callbacks: list[Callable[[EthicsCheckResult], None]] = []

    def _initialize_policies(
        self,
        custom_policies: dict[OECDPrinciple, EthicsPolicy] | None,
    ) -> dict[OECDPrinciple, EthicsPolicy]:
        """Initialize default policies for all principles."""
        default_policies = {
            # Principle 1
            OECDPrinciple.INCLUSIVE_GROWTH: EthicsPolicy(
                principle=OECDPrinciple.INCLUSIVE_GROWTH,
                threshold=0.5,
                description="AI should benefit people and planet",
            ),
            OECDPrinciple.SUSTAINABLE_DEVELOPMENT: EthicsPolicy(
                principle=OECDPrinciple.SUSTAINABLE_DEVELOPMENT,
                threshold=0.6,
                description="Support sustainable development goals",
            ),
            # Principle 2
            OECDPrinciple.FAIRNESS: EthicsPolicy(
                principle=OECDPrinciple.FAIRNESS,
                threshold=0.8,
                blocking=True,
                description="Ensure fair and non-discriminatory outcomes",
            ),
            OECDPrinciple.PRIVACY: EthicsPolicy(
                principle=OECDPrinciple.PRIVACY,
                threshold=0.9,
                blocking=True,
                description="Protect privacy and personal data",
            ),
            # Principle 3
            OECDPrinciple.TRANSPARENCY: EthicsPolicy(
                principle=OECDPrinciple.TRANSPARENCY,
                threshold=0.7,
                description="Maintain transparency in AI operations",
            ),
            OECDPrinciple.EXPLAINABILITY: EthicsPolicy(
                principle=OECDPrinciple.EXPLAINABILITY,
                threshold=0.7,
                description="Provide explanations for AI decisions",
            ),
            # Principle 4
            OECDPrinciple.ROBUSTNESS: EthicsPolicy(
                principle=OECDPrinciple.ROBUSTNESS,
                threshold=0.8,
                blocking=True,
                description="Ensure robust and reliable operation",
            ),
            OECDPrinciple.SECURITY: EthicsPolicy(
                principle=OECDPrinciple.SECURITY,
                threshold=0.9,
                blocking=True,
                description="Maintain security throughout lifecycle",
            ),
            OECDPrinciple.SAFETY: EthicsPolicy(
                principle=OECDPrinciple.SAFETY,
                threshold=0.85,
                blocking=True,
                description="Ensure safe operation and outcomes",
            ),
            # Principle 5
            OECDPrinciple.ACCOUNTABILITY: EthicsPolicy(
                principle=OECDPrinciple.ACCOUNTABILITY,
                threshold=0.8,
                description="Maintain accountability for AI outcomes",
            ),
            OECDPrinciple.TRACEABILITY: EthicsPolicy(
                principle=OECDPrinciple.TRACEABILITY,
                threshold=0.9,
                description="Enable traceability of decisions",
            ),
            OECDPrinciple.HUMAN_OVERSIGHT: EthicsPolicy(
                principle=OECDPrinciple.HUMAN_OVERSIGHT,
                threshold=0.8,
                human_review_required=True,
                description="Maintain meaningful human oversight",
            ),
        }

        if custom_policies:
            default_policies.update(custom_policies)

        return default_policies

    def check_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        strategy: str,
        signal_explanation: str | None = None,
    ) -> tuple[bool, list[EthicsCheckResult]]:
        """
        Comprehensive ethics check for a proposed trade.

        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            quantity: Number of shares
            price: Trade price
            strategy: Strategy name
            signal_explanation: Explanation of signal generation

        Returns:
            Tuple of (approved, list of check results)
        """
        results = []
        approved = True

        # Check 1: ESG compliance
        esg_result = self.check_esg_compliance(symbol)
        results.append(esg_result)
        if not esg_result.passed and esg_result.blocking:
            approved = False

        # Check 2: Sector exclusions
        sector_result = self.check_sector_exclusion(symbol)
        results.append(sector_result)
        if not sector_result.passed and sector_result.blocking:
            approved = False

        # Check 3: Market manipulation risk
        manipulation_result = self.check_manipulation_risk(symbol, action, quantity, price)
        results.append(manipulation_result)
        if not manipulation_result.passed and manipulation_result.blocking:
            approved = False

        # Check 4: Transparency/Explainability
        if signal_explanation:
            transparency_result = self.check_explainability(strategy, signal_explanation)
            results.append(transparency_result)

        # Check 5: Human oversight requirements
        oversight_result = self.check_human_oversight_required(symbol, action, quantity * price)
        results.append(oversight_result)
        if oversight_result.requires_human_review:
            self._add_to_review_queue(symbol, action, quantity, price, results)

        # Store results
        self._check_history.extend(results)

        # Trigger violation callbacks
        for result in results:
            if not result.passed:
                self._trigger_violation_callbacks(result)

        return approved, results

    def check_esg_compliance(self, symbol: str) -> EthicsCheckResult:
        """
        Check ESG compliance for a symbol.

        Implements OECD Principle 1: Sustainable development.
        """
        esg_score = self._get_esg_score(symbol)

        if not esg_score:
            # No ESG data available - neutral assessment
            return EthicsCheckResult(
                check_id=self._generate_check_id(),
                timestamp=datetime.utcnow(),
                principle=OECDPrinciple.SUSTAINABLE_DEVELOPMENT,
                passed=True,
                score=0.5,
                threshold=self._policies[OECDPrinciple.SUSTAINABLE_DEVELOPMENT].threshold,
                details={"reason": "No ESG data available", "symbol": symbol},
                recommendations=["Obtain ESG data for comprehensive assessment"],
            )

        # Evaluate against thresholds
        thresholds = self._esg_thresholds
        issues = []

        if esg_score.overall < thresholds["minimum_overall"]:
            issues.append(
                f"Overall ESG score {esg_score.overall:.1f} below minimum {thresholds['minimum_overall']}"
            )

        if esg_score.environmental < thresholds["minimum_environmental"]:
            issues.append(f"Environmental score {esg_score.environmental:.1f} below minimum")

        if esg_score.social < thresholds["minimum_social"]:
            issues.append(f"Social score {esg_score.social:.1f} below minimum")

        if esg_score.governance < thresholds["minimum_governance"]:
            issues.append(f"Governance score {esg_score.governance:.1f} below minimum")

        if len(esg_score.controversies) > thresholds["controversy_limit"]:
            issues.append(f"{len(esg_score.controversies)} controversies exceed limit")

        passed = len(issues) == 0
        score = esg_score.overall / 100.0

        return EthicsCheckResult(
            check_id=self._generate_check_id(),
            timestamp=datetime.utcnow(),
            principle=OECDPrinciple.SUSTAINABLE_DEVELOPMENT,
            passed=passed,
            score=score,
            threshold=self._policies[OECDPrinciple.SUSTAINABLE_DEVELOPMENT].threshold,
            details={
                "symbol": symbol,
                "esg_overall": esg_score.overall,
                "environmental": esg_score.environmental,
                "social": esg_score.social,
                "governance": esg_score.governance,
                "controversies": esg_score.controversies,
                "issues": issues,
            },
            recommendations=[
                "Consider ESG-positive alternatives" if not passed else "ESG criteria met"
            ],
            blocking=not passed and self._policies[OECDPrinciple.SUSTAINABLE_DEVELOPMENT].blocking,
        )

    def check_sector_exclusion(self, symbol: str) -> EthicsCheckResult:
        """
        Check if symbol is in an excluded sector.

        Implements OECD Principle 1: Human well-being.
        """
        # In production, fetch sector from market data
        sector = self._get_sector(symbol)

        is_excluded = sector.lower() in [s.lower() for s in self._excluded_sectors]

        return EthicsCheckResult(
            check_id=self._generate_check_id(),
            timestamp=datetime.utcnow(),
            principle=OECDPrinciple.HUMAN_WELLBEING,
            passed=not is_excluded,
            score=0.0 if is_excluded else 1.0,
            threshold=1.0,
            details={
                "symbol": symbol,
                "sector": sector,
                "excluded_sectors": self._excluded_sectors,
            },
            recommendations=[
                f"Symbol {symbol} is in excluded sector: {sector}"
                if is_excluded
                else "Sector check passed"
            ],
            blocking=is_excluded,
        )

    def check_manipulation_risk(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
    ) -> EthicsCheckResult:
        """
        Check for potential market manipulation patterns.

        Implements OECD Principle 2: Rule of law and fairness.
        """
        manipulation_risks = []
        risk_score = 0.0

        # Track order for pattern detection
        order = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.utcnow(),
        }

        # Check for wash trading patterns
        wash_risk = self._check_wash_trading(order)
        if wash_risk > 0.5:
            manipulation_risks.append(
                {
                    "type": ManipulationType.WASH_TRADING.value,
                    "risk": wash_risk,
                }
            )
            risk_score = max(risk_score, wash_risk)

        # Check for layering patterns
        layering_risk = self._check_layering(order)
        if layering_risk > 0.5:
            manipulation_risks.append(
                {
                    "type": ManipulationType.LAYERING.value,
                    "risk": layering_risk,
                }
            )
            risk_score = max(risk_score, layering_risk)

        # Check for momentum ignition
        momentum_risk = self._check_momentum_ignition(order)
        if momentum_risk > 0.5:
            manipulation_risks.append(
                {
                    "type": ManipulationType.MOMENTUM_IGNITION.value,
                    "risk": momentum_risk,
                }
            )
            risk_score = max(risk_score, momentum_risk)

        # Store order for pattern tracking
        self._recent_orders.append(order)
        if len(self._recent_orders) > 1000:
            self._recent_orders = self._recent_orders[-500:]

        passed = risk_score < 0.7  # Threshold for concern

        return EthicsCheckResult(
            check_id=self._generate_check_id(),
            timestamp=datetime.utcnow(),
            principle=OECDPrinciple.FAIRNESS,
            passed=passed,
            score=1.0 - risk_score,
            threshold=self._policies[OECDPrinciple.FAIRNESS].threshold,
            details={
                "symbol": symbol,
                "action": action,
                "manipulation_risks": manipulation_risks,
                "risk_score": risk_score,
            },
            recommendations=[
                f"High manipulation risk detected: {[r['type'] for r in manipulation_risks]}"
                if not passed
                else "No manipulation patterns detected"
            ],
            blocking=not passed,
            requires_human_review=risk_score > 0.5,
        )

    def check_explainability(
        self,
        strategy: str,
        explanation: str,
    ) -> EthicsCheckResult:
        """
        Check if decision has adequate explainability.

        Implements OECD Principle 3: Transparency and explainability.
        """
        # Assess explanation quality
        quality_factors = {
            "has_explanation": len(explanation) > 0,
            "reasonable_length": 20 < len(explanation) < 5000,
            "mentions_factors": any(
                word in explanation.lower()
                for word in ["signal", "indicator", "price", "volume", "trend", "momentum"]
            ),
            "mentions_risk": any(
                word in explanation.lower() for word in ["risk", "stop", "limit", "exposure"]
            ),
        }

        score = sum(quality_factors.values()) / len(quality_factors)
        passed = score >= self._policies[OECDPrinciple.EXPLAINABILITY].threshold

        return EthicsCheckResult(
            check_id=self._generate_check_id(),
            timestamp=datetime.utcnow(),
            principle=OECDPrinciple.EXPLAINABILITY,
            passed=passed,
            score=score,
            threshold=self._policies[OECDPrinciple.EXPLAINABILITY].threshold,
            details={
                "strategy": strategy,
                "explanation_length": len(explanation),
                "quality_factors": quality_factors,
            },
            recommendations=[
                "Improve decision explanation" if not passed else "Explanation quality adequate"
            ],
        )

    def check_human_oversight_required(
        self,
        symbol: str,
        action: str,
        notional_value: float,
    ) -> EthicsCheckResult:
        """
        Determine if human oversight is required.

        Implements OECD Principle 5: Human oversight.
        """
        # Thresholds for human review
        review_required = False
        reasons = []

        # Large trades require review
        if notional_value > 100000:  # $100k threshold
            review_required = True
            reasons.append(f"Trade value ${notional_value:,.0f} exceeds threshold")

        # First trade in symbol requires review
        if symbol not in self._order_patterns:
            review_required = True
            reasons.append("First trade in new symbol")

        return EthicsCheckResult(
            check_id=self._generate_check_id(),
            timestamp=datetime.utcnow(),
            principle=OECDPrinciple.HUMAN_OVERSIGHT,
            passed=True,  # Not blocking, just advisory
            score=1.0 if not review_required else 0.5,
            threshold=self._policies[OECDPrinciple.HUMAN_OVERSIGHT].threshold,
            details={
                "symbol": symbol,
                "action": action,
                "notional_value": notional_value,
                "review_reasons": reasons,
            },
            recommendations=reasons if review_required else ["Automated approval permitted"],
            requires_human_review=review_required,
        )

    def _check_wash_trading(self, order: dict) -> float:
        """Detect potential wash trading patterns."""
        symbol = order["symbol"]
        action = order["action"]

        # Look for recent opposite trades in same symbol
        recent_same_symbol = [o for o in self._recent_orders[-50:] if o["symbol"] == symbol]

        if len(recent_same_symbol) < 2:
            return 0.0

        # Check for buy-sell pairs within short time
        opposite_action = "sell" if action == "buy" else "buy"
        opposite_trades = [
            o
            for o in recent_same_symbol
            if o["action"] == opposite_action
            and (order["timestamp"] - o["timestamp"]).total_seconds() < 300  # 5 minutes
        ]

        if opposite_trades:
            return min(0.9, 0.3 * len(opposite_trades))

        return 0.0

    def _check_layering(self, order: dict) -> float:
        """Detect potential layering patterns."""
        symbol = order["symbol"]

        # Look for multiple orders at different prices
        recent_same_symbol = [
            o
            for o in self._recent_orders[-20:]
            if o["symbol"] == symbol and o["action"] == order["action"]
        ]

        if len(recent_same_symbol) < 3:
            return 0.0

        # Check for price ladder pattern
        prices = sorted([o["price"] for o in recent_same_symbol])
        if len(set(prices)) >= 3:
            # Multiple distinct prices could indicate layering
            return min(0.8, 0.2 * len(set(prices)))

        return 0.0

    def _check_momentum_ignition(self, order: dict) -> float:
        """Detect potential momentum ignition patterns."""
        symbol = order["symbol"]
        quantity = order["quantity"]

        # Look for large orders followed by small opposite orders
        recent_same_symbol = [o for o in self._recent_orders[-10:] if o["symbol"] == symbol]

        if len(recent_same_symbol) < 2:
            return 0.0

        # Check if this is an unusually large order
        avg_quantity = sum(o["quantity"] for o in recent_same_symbol) / len(recent_same_symbol)

        if quantity > avg_quantity * 5:
            return 0.6

        return 0.0

    def _get_esg_score(self, symbol: str) -> ESGScore | None:
        """Get ESG score for symbol from cache or provider."""
        if symbol in self._esg_cache:
            return self._esg_cache[symbol]

        if self._esg_provider:
            try:
                score = self._esg_provider.get_score(symbol)
                self._esg_cache[symbol] = score
                return score
            except Exception:
                pass  # Provider error - fall through to return None

        return None

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        # Placeholder - in production, fetch from market data
        return "unknown"

    def _add_to_review_queue(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        results: list[EthicsCheckResult],
    ) -> None:
        """Add trade to human review queue."""
        self._review_queue.append(
            {
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "check_results": [r.check_id for r in results],
                "status": "pending",
            }
        )

    def get_review_queue(self) -> list[dict]:
        """Get pending human review items."""
        return [r for r in self._review_queue if r["status"] == "pending"]

    def approve_review(self, review_id: int, reviewer: str, notes: str) -> bool:
        """Approve a pending review."""
        if review_id < len(self._review_queue):
            self._review_queue[review_id]["status"] = "approved"
            self._review_queue[review_id]["reviewer"] = reviewer
            self._review_queue[review_id]["notes"] = notes
            self._review_queue[review_id]["review_time"] = datetime.utcnow()
            return True
        return False

    def reject_review(self, review_id: int, reviewer: str, reason: str) -> bool:
        """Reject a pending review."""
        if review_id < len(self._review_queue):
            self._review_queue[review_id]["status"] = "rejected"
            self._review_queue[review_id]["reviewer"] = reviewer
            self._review_queue[review_id]["reason"] = reason
            self._review_queue[review_id]["review_time"] = datetime.utcnow()
            return True
        return False

    def set_excluded_sectors(self, sectors: list[str]) -> None:
        """Update excluded sectors list."""
        self._excluded_sectors = sectors

    def add_excluded_sector(self, sector: str) -> None:
        """Add sector to exclusion list."""
        if sector.lower() not in [s.lower() for s in self._excluded_sectors]:
            self._excluded_sectors.append(sector)

    def set_esg_score(self, symbol: str, score: ESGScore) -> None:
        """Manually set ESG score (for testing or override)."""
        self._esg_cache[symbol] = score

    def register_violation_callback(
        self,
        callback: Callable[[EthicsCheckResult], None],
    ) -> None:
        """Register callback for ethics violations."""
        self._violation_callbacks.append(callback)

    def _trigger_violation_callbacks(self, result: EthicsCheckResult) -> None:
        """Trigger callbacks for violations."""
        for callback in self._violation_callbacks:
            with contextlib.suppress(Exception):
                callback(result)

    def _generate_check_id(self) -> str:
        """Generate unique check ID."""
        return f"ETH-{uuid.uuid4().hex[:12].upper()}"

    def get_compliance_summary(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Get summary of ethics compliance."""
        checks = self._check_history

        if start:
            checks = [c for c in checks if c.timestamp >= start]
        if end:
            checks = [c for c in checks if c.timestamp <= end]

        passed_count = sum(1 for c in checks if c.passed)
        failed_count = len(checks) - passed_count

        by_principle = {}
        for check in checks:
            principle = check.principle.value
            if principle not in by_principle:
                by_principle[principle] = {"passed": 0, "failed": 0}
            if check.passed:
                by_principle[principle]["passed"] += 1
            else:
                by_principle[principle]["failed"] += 1

        return {
            "total_checks": len(checks),
            "passed": passed_count,
            "failed": failed_count,
            "compliance_rate": passed_count / len(checks) if checks else 1.0,
            "by_principle": by_principle,
            "pending_reviews": len(self.get_review_queue()),
        }
