# Protective Collar Strategy

## Purpose

Cap downside risk on a long stock position by pairing a protective put with a covered call to offset hedge cost.

## Setup

- Long 100 shares of underlying.
- Buy 1 put (protection) at/below ATM.
- Sell 1 call (income) at/above ATM.
- Align expiries; strikes set by risk/return and cost targets.

## Payoff Profile

- Downside limited below put strike (minus net premium).
- Upside capped at call strike (plus net premium).
- Net cost can be low/zero if call premium offsets put cost.

## Greeks (approx)

- Delta: stock (+), put (-), call (+); net less than 1 and capped on upside.
- Vega: put positive, call negative; near neutral depending on strikes.
- Theta: call income offsets put decay; can be near neutral.

## Implementation Notes

- Select call strike to balance premium vs. upside cap; avoid too tight caps unless protection cost dominates.
- Avoid short calls into known dividend dates to reduce early assignment risk; prefer expiries after ex-div.
- Roll legs independently: roll call if close to assignment; roll put as protection horizon shifts.

## Management

- If stock rallies to call strike: decide to let assign, roll up/out, or close collar.
- If stock drops: keep/roll put; consider rolling call lower to collect more premium (increases cap risk).
- Monitor IV skew: collars cheaper when put-call skew is steep in your favor.

## References

- See `protective-collar/` subfolder for detailed guides, examples, and calculators.
