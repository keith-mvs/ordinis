# Ordinis Demo v0.50 - Strategy Implementation Fixes

## Overview
This document outlines all fixes and improvements made to align ordinis-demo-v050.py with the ATR-Optimized RSI strategy specification.

## Fixed Issues

### 1. ✅ Proper Regime Gating (HIGH PRIORITY)
**Problem**: Missing RegimeDetector/StrategyLoader integration for regime filtering.

**Solution**:
- Added `from ordinis.engines.signalcore.strategy_loader import StrategyLoader`
- Initialize StrategyLoader in `initialize()` method
- Added regime check before signal generation using `StrategyLoader.should_trade()`

```python
# Lines 636-643
if self.strategy_loader:
    allowed, reason = self.strategy_loader.should_trade(symbol, df)
    if not allowed:
        self.log_manager.get("risk").info(
            f"Blocked by regime gating: {symbol} - {reason}"
        )
        return
```

### 2. ✅ Configurable ATR Scaling (HIGH PRIORITY)
**Problem**: Hardcoded `atr * 20` scaling was undocumented and inflexible.

**Solution**:
- Made ATR scaling configurable via `atr_scale_factor` parameter
- Default remains 20 for minute bars (sqrt of 390 trading minutes)
- Can be adjusted based on timeframe

```python
# Lines 767-771
atr_scale = global_params.get("atr_scale_factor", 20)  # Default 20 for minute bars
scaled_atr = atr * atr_scale
```

### 3. ✅ Removed Undocumented Trailing Stop
**Problem**: 5% drawdown trailing stop was not in strategy specification.

**Solution**:
- Removed the undocumented `pos["max_drawdown"] > 5.0` exit condition
- Now only uses documented exits: stop_loss, take_profit, RSI exits

### 4. ✅ Documented Existing Position Behavior
**Problem**: Different RSI exit threshold (65) for existing positions was undocumented.

**Solution**:
- Made existing position exit RSI configurable via `existing_position_exit_rsi`
- Added documentation explaining the conservative exit for pre-existing positions
- Default remains 65 to prevent premature exits

```python
# Lines 660-667
existing_exit_rsi = global_params.get("existing_position_exit_rsi", 65)
```

### 5. ✅ Configurable Volume Confirmation
**Problem**: Volume confirmation logic was hardcoded and undocumented.

**Solution**:
- Added `volume_confirmation_ratio` parameter (default 1.2)
- Added `require_volume_confirmation` flag (default False)
- Volume surge still noted in signal reason when present

```python
# Lines 684-693
volume_threshold = global_params.get("volume_confirmation_ratio", 1.2)
volume_confirmed = volume_ratio > volume_threshold

if global_params.get("require_volume_confirmation", False) and not volume_confirmed:
    # Skip signal
```

### 6. ✅ Signal Cooldown Configuration
**Problem**: 5-minute cooldown between signals was hardcoded.

**Solution**:
- Made configurable via `signal_cooldown_seconds` parameter
- Default remains 300 seconds (5 minutes)

## New Configuration Parameters

All new parameters are in the `global_params` section:

```yaml
global_params:
  # Existing parameters...

  # NEW: ATR scaling for different timeframes
  atr_scale_factor: 20  # sqrt(390) for minute bars

  # NEW: Special handling for pre-existing positions
  existing_position_exit_rsi: 65  # More conservative exit

  # NEW: Volume confirmation settings
  volume_confirmation_ratio: 1.2  # Volume must be 1.2x average
  require_volume_confirmation: false  # Optional volume gating

  # NEW: Signal generation cooldown
  signal_cooldown_seconds: 300  # 5 minutes between signals
```

## Additional Improvements

### Position Sizing
- Already using config value `max_position_pct` (3.0%)
- Correctly calculated as percentage of equity

### Take Profit Multiplier
- Already reading from symbol-specific or global config
- Supports per-symbol customization (e.g., 3.0 for COIN)

### Daily Loss Limit
- Previously removed per user request
- Set to `float('inf')` (no limit)

## Testing Recommendations

1. **Unit Tests** should validate:
   - Regime gating blocks trades in choppy/quiet markets
   - ATR scaling produces correct stop/TP distances
   - Volume confirmation properly gates entries when enabled
   - Existing positions use higher RSI exit threshold

2. **Integration Tests** should verify:
   - StrategyLoader loads config correctly
   - RegimeDetector integration works as expected
   - All configurable parameters are respected

3. **Backtesting** should compare:
   - Results with/without regime filtering
   - Impact of ATR scaling on P&L
   - Effect of volume confirmation on win rate

## Migration Notes

For users upgrading from previous versions:

1. Add new parameters to your `atr_optimized_rsi.yaml`:
   ```yaml
   global_params:
     atr_scale_factor: 20
     existing_position_exit_rsi: 65
     volume_confirmation_ratio: 1.2
     require_volume_confirmation: false
     signal_cooldown_seconds: 300
   ```

2. Ensure StrategyLoader can access your config file

3. Review regime filter settings in config:
   ```yaml
   regime_filter:
     enabled: true
     avoid_regimes:
       - quiet_choppy
       - choppy
   ```

## Files Modified
- `exe/050/ordinis-demo-v050.py` (primary)
- `scripts/trading/ordinis-demo-v050.py` (backup)

## Version
- **Version**: 0.50.1
- **Date**: 2025-12-23
- **Status**: Ready for testing