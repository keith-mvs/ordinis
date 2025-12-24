import pandas as pd
import numpy as np

from ordinis.engines.signalcore.models.atr_optimized_rsi import ATROptimizedRSIModel
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.features.technical import TechnicalIndicators


def make_price_series(n=250, seed=7, drift=0.0005, vol=0.01):
    np.random.seed(seed)
    returns = np.random.normal(loc=drift, scale=vol, size=n)
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": price * (1 - 0.001),
        "high": price * (1 + 0.002),
        "low": price * (1 - 0.002),
        "close": price,
        "volume": np.random.randint(100, 1000, size=n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return df

import asyncio

if __name__ == '__main__':
    df = make_price_series()
    # make last price artificially low to be below SMA
    print('Initial last close', df['close'].iloc[-1])
    df.at[df.index[-1], 'close'] = df['close'].rolling(10).mean().iloc[-1] - 5
    print('After set last close', df['close'].iloc[-1])

    model_cfg = ModelConfig(model_id='test', model_type='technical', parameters={
        'use_optimized': False,
        'enforce_regime_gate': True,
        'regime_sma_period': 5,
    })
    model = ATROptimizedRSIModel(model_cfg)

    # compute indicators
    rsi = TechnicalIndicators.rsi(df['close'], model.rsi_period)
    atr = model._compute_atr(df['high'], df['low'], df['close'])
    print('rsi last', rsi.iloc[-1])
    model.config.parameters['rsi_oversold'] = int(rsi.iloc[-1] + 10)
    print('rsi_oversold set to', model.config.parameters['rsi_oversold'])

    print('sma last', df['close'].rolling(model.regime_sma_period).mean().iloc[-1])
    print('current_price', df['close'].iloc[-1])
    print('validate', model.validate(df))

    # Try generate and inspect why None
    async def check():
        sig = await model.generate('TEST', df, None)
        print('signal:', sig)

    asyncio.run(check())

    # Now set price above sma and recompute rsi and try again
    df.at[df.index[-1], 'close'] = df['close'].rolling(5).mean().iloc[-1] + 10
    df.at[df.index[-1], 'open'] = df.at[df.index[-1], 'close'] * 0.99
    rsi = TechnicalIndicators.rsi(df['close'], model.rsi_period)
    model.config.parameters['rsi_oversold'] = int(rsi.iloc[-1] + 5)
    model = ATROptimizedRSIModel(model.config)

    print('\nAfter moving price above sma:')
    print('sma last', df['close'].rolling(model.regime_sma_period).mean().iloc[-1])
    print('current_price', df['close'].iloc[-1])
    print('rsi last', rsi.iloc[-1])

    print('validate', model.validate(df))

    # Recompute checks manually
    cfg = model._get_config_for_symbol('TEST')
    current_price = df['close'].iloc[-1]
    current_rsi = rsi.iloc[-1]
    atr = model._compute_atr(df['high'], df['low'], df['close'])
    current_atr = atr.iloc[-1]
    print('cfg.rsi_oversold', cfg.rsi_oversold)
    print('current_rsi < cfg.rsi_oversold?', current_rsi < cfg.rsi_oversold)
    sma = df['close'].rolling(model.regime_sma_period).mean()
    print('current_price <= sma?', current_price <= sma.iloc[-1])
    print('current_atr is nan?', pd.isna(current_atr))

    async def check2():
        sig2 = await model.generate('TEST', df, None)
        print('signal2:', sig2)

    asyncio.run(check2())
