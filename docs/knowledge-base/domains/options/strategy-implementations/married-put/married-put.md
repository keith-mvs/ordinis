# Married Put Strategy

## Purpose

Hedge downside risk on a long stock position by buying a put (1:1), capping losses while retaining upside.

## Setup

- Long 100 shares of underlying.
- Buy 1 put option (same underlying).
- Strike: at/near-the-money; expiry aligned with hedge horizon.

## Payoff Profile

- Max loss ≈ put premium + (purchase price - strike) if strike < purchase price.
- Upside: stock gains minus put premium; protected below strike.

## Greeks (approx)

- Delta: ~ stock delta (near +1) minus put delta (negative), net < 1.
- Vega: positive (benefits from vol increase).
- Theta: negative from put decay.

## Implementation Notes

- Choose strike/tenor based on risk budget and cost tolerance.
- Roll hedge as expiry approaches; reassess after large moves.
- Be aware of early exercise risk if put is deep ITM near expiry (American style).

## Management

- If underlying rallies: consider selling/rolling put to lock gains, or let decay fund future hedges.
- If underlying drops: decide to keep hedge, roll down, or exit stock+put.
- Monitor IV: high IV inflates hedge cost; low IV reduces protection quality if vol spikes.

## References

- See `married-put/` subfolder for detailed guides, examples, and calculators.
