# Polycab India Ltd. (POLYCAB)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 9083.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 10.88% / 6.11%
- **Sum % (uncompounded):** 43.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 10.88% | 43.5% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 10.88% | 43.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 10.88% | 43.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 05:30:00 | 3614.05 | 3012.57 | 3498.48 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=84.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 05:30:00 | 3783.83 | 3055.19 | 3541.96 | T1 booked 50% @ 3783.83 |
| Target hit | 2023-09-12 05:30:00 | 4942.40 | 3674.21 | 5030.34 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 05:30:00 | 5360.55 | 3818.20 | 5137.32 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=143.85 |
| Stop hit — per-position SL triggered | 2023-10-09 05:30:00 | 5144.77 | 3917.30 | 5206.96 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-03-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 05:30:00 | 4986.55 | 4581.54 | 4794.15 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=152.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 05:30:00 | 5291.09 | 4622.13 | 4980.00 | T1 booked 50% @ 5291.09 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 05:30:00 | 3614.05 | 2023-07-11 05:30:00 | 3783.83 | PARTIAL | 0.50 | 4.70% |
| BUY | retest1 | 2023-06-28 05:30:00 | 3614.05 | 2023-09-12 05:30:00 | 4942.40 | TARGET_HIT | 0.50 | 36.76% |
| BUY | retest1 | 2023-09-27 05:30:00 | 5360.55 | 2023-10-09 05:30:00 | 5144.77 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest1 | 2024-03-21 05:30:00 | 4986.55 | 2024-04-04 05:30:00 | 5291.09 | PARTIAL | 0.50 | 6.11% |
