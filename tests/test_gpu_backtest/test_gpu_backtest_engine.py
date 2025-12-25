import numpy as np
from scripts.strategy_optimizer import GPUBacktestEngine, HAS_TORCH, torch


def test_compute_returns_cpu():
    engine = GPUBacktestEngine(use_gpu=False)
    prices = np.array([100.0, 101.0, 103.0, 107.0])
    r = engine.compute_returns(prices)
    expected = np.diff(np.log(prices))
    assert np.allclose(r, expected)


def test_torch_fallback():
    # Create a fake torch with minimal API to exercise fallback
    class FakeTensor:
        def __init__(self, arr):
            import numpy as _np
            self.arr = _np.array(arr, dtype=float)

        def log(self):
            import numpy as _np
            return FakeTensor(_np.log(self.arr))

        def diff(self):
            import numpy as _np
            return FakeTensor(_np.diff(self.arr))

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

    class FakeCuda:
        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def tensor(arr, device=None, dtype=None):
            return FakeTensor(arr)

        @staticmethod
        def device(name):
            return name

    # Monkeypatch by assignment
    import scripts.strategy_optimizer as mod
    mod.HAS_TORCH = True
    mod.torch = FakeTorch

    engine = GPUBacktestEngine(use_gpu=True)
    prices = np.array([100.0, 101.0, 103.0, 107.0])
    r = engine.compute_returns(prices)
    expected = np.diff(np.log(prices))
    assert np.allclose(r, expected)
