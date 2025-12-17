# Queueing Orderbook Pressure Signal

## Purpose

Estimate short-term orderbook pressure using queue position and arrival/departure dynamics to inform entry timing and size.

## Inputs

- L2 orderbook snapshots or events (price, size, side, timestamp).
- Queue position estimates per level; venue-specific rules for priority.

## Method (outline)

1) For top-of-book levels, track arrivals/cancels/trades to estimate queue velocity.
2) Compute pressure metrics:
   - bid_pressure = arrivals_bid - departures_bid
   - ask_pressure = arrivals_ask - departures_ask
   - imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + ε)
3) Signal:
   - pressure_score = imbalance (scaled)
   - optional fill_prob estimate from queue position and velocity.

## Outputs

- pressure_score (-1 to 1), fill_prob, queue_velocity metrics.

## Usage Notes

- Align timestamps; handle bursty updates; smooth over short windows (e.g., 100-500ms).
- Venue rules matter (price-time vs pro-rata); adjust queue estimates accordingly.
- High noise: gate by spread/volume filters to avoid microstructure churn.

## Python Example (event-based tally, simplified)

```python
import pandas as pd
import numpy as np

events = pd.read_csv("lob_events.csv")  # ts, side, type (add/cancel/trade), size
events["ts"] = pd.to_datetime(events["ts"])
events = events.set_index("ts").sort_index()

def pressure(events, window="500ms"):
    grouped = events.groupby(["side", pd.Grouper(freq=window)])
    adds = grouped.apply(lambda df: df.loc[df["type"] == "add", "size"].sum())
    removes = grouped.apply(lambda df: df.loc[df["type"] != "add", "size"].sum())
    df = pd.concat([adds.rename("adds"), removes.rename("removes")], axis=1).fillna(0)
    df["pressure"] = (df["adds"] - df["removes"]) / (df["adds"] + df["removes"] + 1e-6)
    return df

pressure_scores = pressure(events, window="500ms")

# Aggregate to a single series (bid positive, ask negative)
pressure_scores = pressure_scores.unstack(0).fillna(0)
pressure_scores["pressure_score"] = pressure_scores["adds"]["bid"] - pressure_scores["adds"]["ask"]
pressure_scores["pressure_score"] -= (pressure_scores["removes"]["bid"] - pressure_scores["removes"]["ask"])
pressure_series = pressure_scores["pressure_score"].rolling("1s").mean()

# Optional fill probability proxy: ratio of adds to total events
pressure_scores["fill_prob"] = (
    pressure_scores["adds"]["bid"] + pressure_scores["adds"]["ask"]
) / (
    pressure_scores["adds"]["bid"] + pressure_scores["adds"]["ask"] +
    pressure_scores["removes"]["bid"] + pressure_scores["removes"]["ask"] + 1e-6
)

# Gate noisy periods: require spread tight and volume above threshold
top_of_book = pd.read_csv("top_of_book.csv", parse_dates=["ts"], index_col="ts")  # ts, bid, ask, bid_size, ask_size
top_of_book = top_of_book.reindex(pressure_series.index).ffill()
spread = (top_of_book["ask"] - top_of_book["bid"]) / top_of_book["bid"]
volume_filter = (top_of_book["bid_size"] + top_of_book["ask_size"]) > top_of_book["bid_size"].quantile(0.2)
clean_pressure = pressure_series.where((spread < 0.001) & volume_filter)
```

## Ensemble Hook

Use pressure_score to time entries/exits or size increments; treat fill_prob as execution confidence to modulate aggressiveness.
