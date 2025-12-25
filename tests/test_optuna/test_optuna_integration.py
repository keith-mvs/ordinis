from scripts.strategy_optimizer import StrategyOptimizer, OPTUNA_AVAILABLE


def test_optuna_fallback():
    # Ensure that when Optuna is not available optimizer falls back gracefully
    import scripts.strategy_optimizer as mod
    mod.OPTUNA_AVAILABLE = False

    opt = StrategyOptimizer(
        strategy_name='fibonacci_adx',
        n_cycles=2,
        use_gpu=False,
        metric='cagr',
        n_workers=1,
        bootstrap_n=0,
        skip_bootstrap=True,
        use_optuna=True,
        optuna_trials=2,
    )

    # Monkeypatch load_data to provide synthetic dataset
    def fake_load(self):
        import pandas as pd
        import numpy as np
        idx = pd.date_range(end=pd.Timestamp.now(), periods=600, freq='D')
        prices = np.linspace(100.0, 150.0, num=len(idx))
        df = pd.DataFrame({'open':prices*0.99,'high':prices*1.02,'low':prices*0.98,'close':prices,'volume':np.random.randint(100,1000,len(idx))}, index=idx)
        self.data_cache = {'S': df}
        return True

    StrategyOptimizer.load_data = fake_load

    # Should run and not crash even though OPTUNA_AVAILABLE==False
    res = opt.run_optimization()
    assert hasattr(res, 'best_params')