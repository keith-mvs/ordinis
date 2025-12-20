#!/usr/bin/env python
"""Quick position check."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.adapters.broker import AlpacaBroker


async def check():
    broker = AlpacaBroker(paper=True)
    await broker.connect()

    account = await broker.get_account()
    print(f"Equity: ${account.equity:,.2f}")
    print(f"Cash: ${account.cash:,.2f}")
    print()

    positions = await broker.get_positions()
    if positions:
        print("OPEN POSITIONS:")
        print("-" * 60)
        total_value = 0
        for p in sorted(positions, key=lambda x: float(x.market_value), reverse=True):
            mv = float(p.market_value)
            pct = (mv / float(account.equity)) * 100
            total_value += mv
            print(
                f"{p.symbol:6} | {p.quantity:4} shares | " f"${mv:>8,.2f} | {pct:5.1f}% of equity"
            )
        print("-" * 60)
        print(f"Total deployed: ${total_value:,.2f} ({total_value/float(account.equity)*100:.1f}%)")
    else:
        print("No open positions")


if __name__ == "__main__":
    asyncio.run(check())
