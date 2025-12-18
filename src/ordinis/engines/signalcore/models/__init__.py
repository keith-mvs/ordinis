"""SignalCore trading models."""

from .adx_trend import ADXTrendModel
from .atr_breakout import ATRBreakoutModel
from .atr_optimized_rsi import ATROptimizedRSIModel
from .bollinger_bands import BollingerBandsModel
from .evt_risk_gate import EVTGatedStrategy, EVTRiskGate
from .fibonacci_retracement import FibonacciRetracementModel
from .fundamental_value import FundamentalValueModel
from .garch_breakout import GARCHBreakoutModel
from .hmm_regime import HMMRegimeModel
from .kalman_hybrid import KalmanHybridModel
from .llm_enhanced import LLMEnhancedModel, LLMFeatureEngineer
from .lstm_model import LSTMModel
from .macd import MACDModel
from .mi_ensemble import MIEnsembleModel
from .momentum_breakout import MomentumBreakoutModel
from .mtf_momentum import MTFMomentumModel
from .network_parity import NetworkRiskParityModel
from .ou_pairs import OUPairsModel
from .parabolic_sar import ParabolicSARModel
from .rsi_mean_reversion import RSIMeanReversionModel
from .rsi_volume_reversion import RSIVolumeReversionModel
from .sentiment_momentum import SentimentMomentumModel
from .sma_crossover import SMACrossoverModel
from .statistical_reversion import StatisticalReversionModel
from .trend_following import TrendFollowingModel
from .volume_trend import VolumeTrendModel

__all__ = [
    # Classic Technical
    "ADXTrendModel",
    "ATRBreakoutModel",
    "ATROptimizedRSIModel",
    "BollingerBandsModel",
    "FibonacciRetracementModel",
    "MACDModel",
    "MomentumBreakoutModel",
    "ParabolicSARModel",
    "RSIMeanReversionModel",
    "RSIVolumeReversionModel",
    "SMACrossoverModel",
    "TrendFollowingModel",
    "VolumeTrendModel",
    # Quantitative
    "EVTRiskGate",
    "EVTGatedStrategy",
    "GARCHBreakoutModel",
    "HMMRegimeModel",
    "KalmanHybridModel",
    "MIEnsembleModel",
    "MTFMomentumModel",
    "NetworkRiskParityModel",
    "OUPairsModel",
    # ML/AI
    "FundamentalValueModel",
    "LLMEnhancedModel",
    "LLMFeatureEngineer",
    "LSTMModel",
    "SentimentMomentumModel",
    "StatisticalReversionModel",
]
