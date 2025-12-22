"""
Tests for CortexRepository persistence layer.

Tests CRUD operations for CortexOutput and StrategyHypothesis.
"""

import json
import pytest
from datetime import datetime

from ordinis.adapters.storage.database import DatabaseManager
from ordinis.adapters.storage.repositories.cortex import (
    CortexOutputRow,
    CortexRepository,
    StrategyHypothesisRow,
)


@pytest.fixture
async def db_manager(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_cortex.db"
    db = DatabaseManager(db_path=db_path, auto_backup=False)
    await db.initialize()
    yield db
    await db.shutdown()


@pytest.fixture
async def cortex_repo(db_manager):
    """Create a CortexRepository with initialized schema."""
    repo = CortexRepository(db=db_manager)
    await repo.initialize_schema()
    return repo


# --- Test CortexOutputRow ---


class TestCortexOutputRow:
    """Test CortexOutputRow model."""

    def test_create_output_row(self):
        """Test creating a CortexOutputRow."""
        row = CortexOutputRow(
            output_id="out-123",
            output_type="hypothesis",
            content='{"key": "value"}',
            confidence=0.85,
            reasoning="Test reasoning",
            model_used="deepseek-r1",
            created_at=datetime.now().isoformat(),
        )
        assert row.output_id == "out-123"
        assert row.confidence == 0.85

    def test_get_content_dict(self):
        """Test parsing content JSON."""
        row = CortexOutputRow(
            output_id="out-123",
            output_type="research",
            content='{"summary": "Test summary", "sources": [1, 2, 3]}',
            confidence=0.9,
            reasoning="Test",
            created_at=datetime.now().isoformat(),
        )
        content = row.get_content_dict()
        assert content["summary"] == "Test summary"
        assert len(content["sources"]) == 3

    def test_to_insert_tuple(self):
        """Test conversion to insert tuple."""
        row = CortexOutputRow(
            output_id="out-456",
            output_type="code_analysis",
            content="{}",
            confidence=0.75,
            reasoning="Analysis",
            prompt_tokens=100,
            completion_tokens=50,
            created_at="2024-12-21T10:00:00",
        )
        tup = row.to_insert_tuple()
        assert tup[0] == "out-456"
        assert tup[1] == "code_analysis"
        assert tup[8] == 100  # prompt_tokens
        assert tup[9] == 50  # completion_tokens


# --- Test StrategyHypothesisRow ---


class TestStrategyHypothesisRow:
    """Test StrategyHypothesisRow model."""

    def test_create_hypothesis_row(self):
        """Test creating a StrategyHypothesisRow."""
        row = StrategyHypothesisRow(
            hypothesis_id="hyp-123",
            name="Test Strategy",
            description="A test strategy",
            rationale="Testing",
            instrument_class="equity",
            time_horizon="swing",
            strategy_type="trend_following",
            parameters='{"fast_period": 50, "slow_period": 200}',
            entry_conditions='["Price above SMA"]',
            exit_conditions='["Stop loss hit"]',
            max_position_size_pct=0.10,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            expected_sharpe=1.5,
            expected_win_rate=0.55,
            confidence=0.80,
            created_at=datetime.now().isoformat(),
        )
        assert row.hypothesis_id == "hyp-123"
        assert row.strategy_type == "trend_following"

    def test_get_parameters_dict(self):
        """Test parsing parameters JSON."""
        row = StrategyHypothesisRow(
            hypothesis_id="hyp-456",
            name="RSI Strategy",
            description="Mean reversion",
            rationale="High volatility",
            instrument_class="equity",
            time_horizon="swing",
            strategy_type="mean_reversion",
            parameters='{"rsi_period": 14, "oversold": 30}',
            entry_conditions="[]",
            exit_conditions="[]",
            max_position_size_pct=0.10,
            stop_loss_pct=0.03,
            take_profit_pct=0.08,
            expected_sharpe=1.2,
            expected_win_rate=0.55,
            confidence=0.70,
            created_at=datetime.now().isoformat(),
        )
        params = row.get_parameters_dict()
        assert params["rsi_period"] == 14
        assert params["oversold"] == 30


# --- Test CortexRepository ---


class TestCortexRepository:
    """Test CortexRepository database operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_output(self, cortex_repo):
        """Test saving and retrieving an output."""
        output = CortexOutputRow(
            output_id="out-test-001",
            output_type="research",
            content='{"query": "test", "results": []}',
            confidence=0.85,
            reasoning="Test research output",
            model_used="deepseek-r1",
            prompt_tokens=150,
            completion_tokens=75,
            trace_id="trace-001",
            created_at=datetime.now().isoformat(),
        )

        row_id = await cortex_repo.save_output(output)
        assert row_id is not None

        retrieved = await cortex_repo.get_output_by_id("out-test-001")
        assert retrieved is not None
        assert retrieved.output_type == "research"
        assert retrieved.confidence == 0.85
        assert retrieved.prompt_tokens == 150

    @pytest.mark.asyncio
    async def test_get_outputs_by_type(self, cortex_repo):
        """Test retrieving outputs by type."""
        # Save multiple outputs of different types
        for i, output_type in enumerate(["research", "research", "code_analysis"]):
            output = CortexOutputRow(
                output_id=f"out-type-{i}",
                output_type=output_type,
                content="{}",
                confidence=0.8,
                reasoning="Test",
                created_at=datetime.now().isoformat(),
            )
            await cortex_repo.save_output(output)

        research_outputs = await cortex_repo.get_outputs_by_type("research")
        assert len(research_outputs) == 2

        code_outputs = await cortex_repo.get_outputs_by_type("code_analysis")
        assert len(code_outputs) == 1

    @pytest.mark.asyncio
    async def test_save_and_get_hypothesis(self, cortex_repo):
        """Test saving and retrieving a hypothesis."""
        hypothesis = StrategyHypothesisRow(
            hypothesis_id="hyp-test-001",
            name="SMA Crossover",
            description="Moving average crossover strategy",
            rationale="Trending market conditions",
            instrument_class="equity",
            time_horizon="swing",
            strategy_type="trend_following",
            parameters=json.dumps({"fast": 50, "slow": 200}),
            entry_conditions=json.dumps(["Fast > Slow", "Volume confirmed"]),
            exit_conditions=json.dumps(["Fast < Slow", "Stop loss"]),
            max_position_size_pct=0.10,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            expected_sharpe=1.5,
            expected_win_rate=0.45,
            confidence=0.75,
            created_at=datetime.now().isoformat(),
        )

        row_id = await cortex_repo.save_hypothesis(hypothesis)
        assert row_id is not None

        retrieved = await cortex_repo.get_hypothesis_by_id("hyp-test-001")
        assert retrieved is not None
        assert retrieved.name == "SMA Crossover"
        assert retrieved.strategy_type == "trend_following"
        assert retrieved.validation_status == "pending"

    @pytest.mark.asyncio
    async def test_get_hypotheses_above_confidence(self, cortex_repo):
        """Test retrieving hypotheses above confidence threshold."""
        # Save hypotheses with different confidence levels
        for i, confidence in enumerate([0.5, 0.7, 0.9]):
            hypothesis = StrategyHypothesisRow(
                hypothesis_id=f"hyp-conf-{i}",
                name=f"Strategy {i}",
                description="Test",
                rationale="Test",
                instrument_class="equity",
                time_horizon="swing",
                strategy_type="mean_reversion",
                parameters="{}",
                entry_conditions="[]",
                exit_conditions="[]",
                max_position_size_pct=0.10,
                stop_loss_pct=0.05,
                take_profit_pct=0.10,
                expected_sharpe=1.0,
                expected_win_rate=0.50,
                confidence=confidence,
                created_at=datetime.now().isoformat(),
            )
            await cortex_repo.save_hypothesis(hypothesis)

        high_confidence = await cortex_repo.get_hypotheses_above_confidence(0.6)
        assert len(high_confidence) == 2  # 0.7 and 0.9

    @pytest.mark.asyncio
    async def test_update_validation_status(self, cortex_repo):
        """Test updating hypothesis validation status."""
        hypothesis = StrategyHypothesisRow(
            hypothesis_id="hyp-validate-001",
            name="Validatable Strategy",
            description="Test",
            rationale="Test",
            instrument_class="equity",
            time_horizon="day",
            strategy_type="adaptive",
            parameters="{}",
            entry_conditions="[]",
            exit_conditions="[]",
            max_position_size_pct=0.10,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            expected_sharpe=1.0,
            expected_win_rate=0.50,
            confidence=0.80,
            created_at=datetime.now().isoformat(),
        )
        await cortex_repo.save_hypothesis(hypothesis)

        # Update status
        updated = await cortex_repo.update_validation_status(
            "hyp-validate-001",
            status="validated",
            validated_by="ProofBench",
        )
        assert updated is True

        # Verify update
        retrieved = await cortex_repo.get_hypothesis_by_id("hyp-validate-001")
        assert retrieved.validation_status == "validated"
        assert retrieved.validated_by == "ProofBench"
        assert retrieved.validated_at is not None

    @pytest.mark.asyncio
    async def test_get_output_stats(self, cortex_repo):
        """Test output statistics."""
        # Save some outputs
        for i, (output_type, tokens) in enumerate([
            ("research", (100, 50)),
            ("research", (150, 75)),
            ("code_analysis", (200, 100)),
        ]):
            output = CortexOutputRow(
                output_id=f"out-stats-{i}",
                output_type=output_type,
                content="{}",
                confidence=0.8,
                reasoning="Test",
                prompt_tokens=tokens[0],
                completion_tokens=tokens[1],
                created_at=datetime.now().isoformat(),
            )
            await cortex_repo.save_output(output)

        stats = await cortex_repo.get_output_stats()
        assert stats["by_type"]["research"] == 2
        assert stats["by_type"]["code_analysis"] == 1
        assert stats["total_prompt_tokens"] == 450
        assert stats["total_completion_tokens"] == 225

    @pytest.mark.asyncio
    async def test_get_hypothesis_stats(self, cortex_repo):
        """Test hypothesis statistics."""
        # Save hypotheses with different types and statuses
        for i, (strategy_type, confidence) in enumerate([
            ("trend_following", 0.75),
            ("mean_reversion", 0.80),
            ("trend_following", 0.65),
        ]):
            hypothesis = StrategyHypothesisRow(
                hypothesis_id=f"hyp-stats-{i}",
                name=f"Strategy {i}",
                description="Test",
                rationale="Test",
                instrument_class="equity",
                time_horizon="swing",
                strategy_type=strategy_type,
                parameters="{}",
                entry_conditions="[]",
                exit_conditions="[]",
                max_position_size_pct=0.10,
                stop_loss_pct=0.05,
                take_profit_pct=0.10,
                expected_sharpe=1.0,
                expected_win_rate=0.50,
                confidence=confidence,
                created_at=datetime.now().isoformat(),
            )
            await cortex_repo.save_hypothesis(hypothesis)

        stats = await cortex_repo.get_hypothesis_stats()
        assert stats["by_type"]["trend_following"] == 2
        assert stats["by_type"]["mean_reversion"] == 1
        assert stats["by_status"]["pending"] == 3
        assert 0.7 <= stats["avg_confidence"] <= 0.75  # Average of 0.75, 0.80, 0.65


# --- Test Engine Persistence Integration ---
# NOTE: These tests are skipped due to environment issues with tensorflow/h5py DLL loading.
# The CortexOutput and StrategyHypothesis imports trigger the RAG module which uses
# sentence-transformers, which in turn loads tensorflow and causes DLL errors.
# Run these tests when the environment is fixed or in CI without tensorflow issues.


class TestEnginePersistenceIntegration:
    """Test CortexEngine persistence integration.

    Note: These tests verify the repository can store outputs in the same format
    the engine would produce. The actual engine integration is tested separately.
    """

    @pytest.mark.asyncio
    async def test_persist_output_row_directly(self, cortex_repo):
        """Test persisting a CortexOutputRow directly."""
        import uuid

        output_id = f"out-{uuid.uuid4().hex[:12]}"
        row = CortexOutputRow(
            output_id=output_id,
            output_type="hypothesis",
            content='{"test": "value"}',
            confidence=0.85,
            reasoning="Test reasoning",
            model_used="deepseek-r1",
            prompt_tokens=100,
            completion_tokens=50,
            metadata='{"trace_id": "test-123"}',
            created_at=datetime.now().isoformat(),
        )

        # Persist via repository
        await cortex_repo.save_output(row)

        # Retrieve and verify
        retrieved = await cortex_repo.get_output_by_id(output_id)
        assert retrieved is not None
        assert retrieved.output_type == "hypothesis"
        assert abs(retrieved.confidence - 0.85) < 0.001
        assert retrieved.model_used == "deepseek-r1"
        assert retrieved.prompt_tokens == 100
        assert retrieved.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_persist_hypothesis_row_directly(self, cortex_repo):
        """Test persisting a StrategyHypothesisRow directly."""
        row = StrategyHypothesisRow(
            hypothesis_id="hyp-test-persist",
            name="Test Strategy",
            description="A test strategy",
            rationale="Testing persistence",
            strategy_type="trend_following",
            instrument_class="equity",
            time_horizon="swing",
            parameters='{"fast_period": 50, "slow_period": 200}',
            entry_conditions='["condition1", "condition2"]',
            exit_conditions='["exit1"]',
            max_position_size_pct=0.10,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            expected_sharpe=1.5,
            expected_win_rate=0.55,
            confidence=0.80,
            created_at=datetime.now().isoformat(),
        )

        # Persist via repository
        await cortex_repo.save_hypothesis(row)

        # Retrieve and verify
        retrieved = await cortex_repo.get_hypothesis_by_id(row.hypothesis_id)
        assert retrieved is not None
        assert retrieved.name == "Test Strategy"
        assert retrieved.strategy_type == "trend_following"
        assert abs(retrieved.confidence - 0.80) < 0.001
        assert retrieved.get_parameters_dict() == {"fast_period": 50, "slow_period": 200}
