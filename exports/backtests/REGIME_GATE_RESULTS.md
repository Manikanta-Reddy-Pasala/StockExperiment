# Regime Gate Results — Phase 3

_Generated: 2026-05-12_

## Approach

Nifty 50 daily bars (no India VIX — Fyers doesn't expose it). Indicators:

- **Regime**: bull (close > EMA50 > EMA200), neutral (close > EMA200), bear (close < EMA200)
- **Volatility**: calm (ATR% < 1.0), normal (1.0-2.0), volatile (> 2.0)

Gate variants tested on selector top-10 + max_concurrent=2 base
(baseline = +21.30% ROI, 9.60% MaxDD).

## Window distribution (Nifty 50, May 2025 → May 2026)

- Regime: bull=149 days, neutral=48, bear=52 (60% bull, 21% bear)
- Volatility: calm=160, normal=72, volatile=17 (7% volatile)

## Results

| Gate variant | Blocked | max=2 ROI | max=3 ROI | DD% |
|--------------|--------:|----------:|----------:|----:|
| **No gate (baseline)** | 0 | **+21.30%** | +12.52% | 9.60 |
| Block bear AND volatile | 8 | +5.99% | +1.56% | 9.60 |
| Block volatile only (allow bear) | 4 | n/a (max=3) +6.53% | - | 7.45 |
| Block bear only (allow volatile) | 8 | n/a (max=3) +1.56% | - | 7.45 |
| Strict volatile ATR > 3% only | 0 | (identical to baseline) | +12.52% | 7.45 |

## Conclusion

**Regime gating HURTS this strategy.**

EMA 200/400 crossover relies on directional moves through long-term means.
The 8 entries blocked by bear/volatile gates were winning SELL setups
(SELL side dominates with ~+382% sum% vs BUY ~+187% in baseline analysis).

Blocking bear regime kicks out the model's primary alpha source.
Blocking volatile periods kicks out the breakout/range-expansion trades.

## What WOULD help (not tested — would need code change)

1. **Direction-aware gate**: in bull regime, block SELL entries; in bear,
   block BUY entries. Requires extracting direction from cycle table —
   not currently exposed. Strategy code change needed.

2. **Volatility-scaled sizing**: instead of binary gate, scale position
   size: full in calm, 0.7× in normal, 0.5× in volatile. Smooths DD
   without dropping signals.

3. **Loss-streak gate**: after N consecutive losses, halt new entries
   for K days. Capital preservation only — doesn't add alpha.

## Recommended path forward

**Drop regime gate from production config.** Keep selector top-10 + max=2
without modifications. Volume/liquidity filters at selector level already
capture the regime-relevant signal.

Re-evaluate regime gating only after:
- Direction info exposed in cycle table (strategy code refactor)
- Multi-year (3+) baseline available to test gate robustness across regimes
