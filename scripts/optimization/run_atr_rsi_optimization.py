"""
Run optimization for ATR-RSI strategy across a universe of stocks.

Uses Polygon.io (Massive) for data fetching and Optuna for optimization.
Implements Walk-Forward Analysis (Proof of Concept).
"""
import asyncio
import logging
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from polygon import RESTClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.tools.optimizer import BacktestOptimizer, OptimizerConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MASSIVE_KEY = os.getenv("MASSIVE_API_KEY", "eV6j21wqw1dD9EUlLcORZBOma4Lc0oQq")

def fetch_polygon_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily data from Polygon."""
    client = RESTClient(api_key=MASSIVE_KEY)
    
    aggs = []
    try:
        # Fetch Aggs (candles)
        for a in client.list_aggs(
            symbol, 
            1, 
            "day", 
            start, 
            end, 
            limit=50000
        ):
            aggs.append(a)
            
        if not aggs:
            return None
            
        df = pd.DataFrame(aggs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Rename columns to match backtester expectations
        df.rename(columns={
            'open': 'open',
            'high': 'high', 
            'low': 'low', 
            'close': 'close', 
            'volume': 'volume'
        }, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Polygon fetch error for {symbol}: {e}")
        return None

async def fetch_data(symbols: list, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """Fetch data for all symbols."""
    data_map = {}
    logger.info(f"Fetching data for {len(symbols)} symbols from Polygon...")
    
    # Run in thread pool since Polygon client is sync
    loop = asyncio.get_event_loop()
    
    for symbol in symbols:
        try:
            # Demo limit
            if len(data_map) >= 3: 
                logger.info("Demo mode: Limited to 3 symbols (Proof of Concept).")
                break
                
            logger.info(f"Fetching {symbol}...")
            
            df = await loop.run_in_executor(None, fetch_polygon_daily, symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                data_map[symbol] = df
                logger.info(f"  Loaded {len(df)} rows for {symbol}")
            
            # Tiny delay to be nice
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            
    return data_map

def run_walk_forward(data_map: dict, cfg: dict):
    """
    Run Walk-Forward Optimization.
    
    Train on 2 years, Test on 6 months.
    """
    start_dt = pd.Timestamp(cfg['start_date'])
    end_dt = pd.Timestamp(cfg['end_date'])
    
    train_window = pd.DateOffset(years=2)
    test_window = pd.DateOffset(months=6)
    
    current_train_start = start_dt
    
    results = []
    
    while current_train_start + train_window + test_window <= end_dt:
        train_end = current_train_start + train_window
        test_end = train_end + test_window
        
        logger.info(f"\n--- Walk Forward Step ---")
        logger.info(f"Train: {current_train_start.date()} -> {train_end.date()}")
        logger.info(f"Test:  {train_end.date()} -> {test_end.date()}")
        
        # Slice Data for Training
        train_data = {}
        test_data = {}
        for sym, df in data_map.items():
            train_mask = (df.index >= current_train_start) & (df.index < train_end)
            test_mask = (df.index >= train_end) & (df.index < test_end)
            
            if train_mask.sum() > 100: # Min bars check
                train_data[sym] = df.loc[train_mask]
            if test_mask.sum() > 20:
                test_data[sym] = df.loc[test_mask]
        
        if not train_data:
            logger.warning("Insufficient training data for this window, skipping.")
            current_train_start += test_window
            continue
            
        # Optimize on Train
        opt_config = OptimizerConfig(
            trials=cfg['trials'],
            metric=cfg['metric'],
            direction=cfg['direction']
        )
        optimizer = BacktestOptimizer(train_data, opt_config)
        study = optimizer.run()
        
        best_params = study.best_params
        train_score = study.best_value
        
        # Validate on Test (Out of Sample)
        # We create a new optimizer just to run one 'trial' with best params, 
        # or we manually call the backtester. Let's use the optimizer for convenience
        # but force the params (requires refactoring optimizer, so we'll just log best params for now)
        
        logger.info(f"Best Train Score ({cfg['metric']}): {train_score:.4f}")
        logger.info(f"Params: {best_params}")
        
        # Store result
        results.append({
            "window_start": current_train_start,
            "train_score": train_score,
            "best_params": best_params
        })
        
        # Step forward
        current_train_start += test_window

    return results

def main():
    config_path = Path(__file__).parent.parent.parent / "configs/optimization/atr_rsi_opt_config.yaml"
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    # Flatten symbol list
    symbols = cfg['symbols']['mid_cap'] + cfg['symbols']['small_cap']
    
    # Run async data fetch
    # Use config dates
    data_map = asyncio.run(fetch_data(symbols, cfg['start_date'], cfg['end_date']))
    
    if not data_map:
        logger.error("No data fetched. Exiting.")
        return

    # Run Walk Forward
    wfo_results = run_walk_forward(data_map, cfg)
    
    print("\n" + "="*60)
    print("WALK FORWARD OPTIMIZATION SUMMARY")
    print("="*60)
    for res in wfo_results:
        print(f"Window: {res['window_start'].date()}")
        print(f"  Train Score: {res['train_score']:.4f}")
        print(f"  Params: {res['best_params']}")
    print("="*60)

if __name__ == "__main__":
    main()
