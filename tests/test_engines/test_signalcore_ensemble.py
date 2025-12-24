import pytest
from datetime import datetime, timezone
from ordinis.engines.signalcore.core.signal import Signal, SignalType, Direction
from ordinis.engines.signalcore.core.ensemble import SignalEnsemble, EnsembleStrategy

@pytest.fixture
def sample_signals():
    dt = datetime.now(timezone.utc)
    s1 = Signal(
        symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG,
        probability=0.8, score=0.8, model_id="m1"
    )
    s2 = Signal(
        symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG,
        probability=0.7, score=0.7, model_id="m2"
    )
    s3 = Signal(
        symbol="AAPL", timestamp=dt, signal_type=SignalType.EXIT, direction=Direction.SHORT,
        probability=0.6, score=-0.6, model_id="m3"
    )
    return [s1, s2, s3]

def test_combine_empty():
    assert SignalEnsemble.combine([]) is None

def test_combine_single(sample_signals):
    s = sample_signals[0]
    result = SignalEnsemble.combine([s])
    assert result == s

def test_combine_voting(sample_signals):
    # s1=LONG, s2=LONG, s3=SHORT -> Voting should result in LONG
    result = SignalEnsemble.combine(sample_signals, strategy=EnsembleStrategy.VOTING)
    assert result.direction == Direction.LONG
    assert result.model_id == "ensemble_voting"
    # Average probability of winning side (0.8 + 0.7) / 2 = 0.75
    assert result.probability == 0.75

def test_combine_voting_tie():
    dt = datetime.now(timezone.utc)
    s1 = Signal(symbol="A", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.8, score=0.8)
    s2 = Signal(symbol="A", timestamp=dt, signal_type=SignalType.EXIT, direction=Direction.SHORT, probability=0.8, score=-0.8)
    
    result = SignalEnsemble.combine([s1, s2], strategy=EnsembleStrategy.VOTING)
    assert result.direction == Direction.NEUTRAL
    assert result.signal_type == SignalType.HOLD

def test_combine_weighted_average(sample_signals):
    # s1: score 0.8, weight 0.8 -> 0.64
    # s2: score 0.7, weight 0.7 -> 0.49
    # s3: score -0.6, weight 0.6 -> -0.36
    # total score: 0.77, total weight: 2.1
    # avg score: 0.77 / 2.1 = 0.366...
    # > 0.1 -> LONG
    
    result = SignalEnsemble.combine(sample_signals, strategy=EnsembleStrategy.WEIGHTED_AVERAGE)
    assert result.direction == Direction.LONG
    assert result.model_id == "ensemble_weighted"
    assert 0.36 < result.score < 0.37

def test_combine_highest_confidence(sample_signals):
    result = SignalEnsemble.combine(sample_signals, strategy=EnsembleStrategy.HIGHEST_CONFIDENCE)
    assert result.model_id == "m1"  # Highest prob 0.8

def test_combine_ic_weighted(sample_signals):
    result = SignalEnsemble.combine(sample_signals, strategy=EnsembleStrategy.IC_WEIGHTED)
    assert result.model_id == "ensemble_ic_weighted"
    # Logic is similar to weighted average but with uniform weights for now
    assert result.direction == Direction.LONG

def test_combine_vol_adjusted(sample_signals):
    result = SignalEnsemble.combine(sample_signals, strategy=EnsembleStrategy.VOL_ADJUSTED)
    assert result.model_id == "ensemble_vol_adjusted"
    assert result.direction == Direction.LONG

def test_combine_regression(sample_signals):
    result = SignalEnsemble.combine(sample_signals, strategy=EnsembleStrategy.REGRESSION)
    assert result.model_id == "ensemble_regression"
    # (0.8 + 0.7 - 0.6) / 3 = 0.3 -> LONG
    assert result.direction == Direction.LONG
    assert pytest.approx(result.score) == 0.3

def test_combine_all_neutral():
    dt = datetime.now(timezone.utc)
    s1 = Signal(symbol="A", timestamp=dt, signal_type=SignalType.HOLD, direction=Direction.NEUTRAL, probability=0.5, score=0.0)
    s2 = Signal(symbol="A", timestamp=dt, signal_type=SignalType.HOLD, direction=Direction.NEUTRAL, probability=0.5, score=0.0)
    
    result = SignalEnsemble.combine([s1, s2])
    assert result == s1
