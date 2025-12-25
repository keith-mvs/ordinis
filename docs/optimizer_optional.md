Optional Optimizer Dependencies

The optimizer supports a few optional features that require extra packages. Install them with:

```bash
pip install -r requirements-optional.txt
```

- optuna: Bayesian hyperparameter search (TPE sampler) with `--use-optuna` and `--optuna-trials` CLI flags
- polygon-api-client / polygon: Polygon.io client for fetching historical OHLC data when CSVs are not available
- finnhub-python: Finnhub client as an alternative data source
- requests: used by internal, lightweight API fetch helpers

If you want me to attempt installing these into your current environment, say so and I'll proceed (I will check current env and notify you of any package conflicts).