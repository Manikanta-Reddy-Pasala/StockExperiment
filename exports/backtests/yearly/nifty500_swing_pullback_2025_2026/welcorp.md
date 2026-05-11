# Welspun Corp Ltd. (WELCORP)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 1291.90
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** -0.32% / 2.71%
- **Sum % (uncompounded):** -1.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.32% | -1.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.32% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | -0.32% | -1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 05:30:00 | 941.60 | 817.79 | 913.81 | Stage2 pullback-breakout RSI=57 vol=3.2x ATR=32.22 |
| Stop hit — per-position SL triggered | 2025-08-01 05:30:00 | 893.26 | 819.70 | 913.73 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2025-09-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 05:30:00 | 923.80 | 832.97 | 875.73 | Stage2 pullback-breakout RSI=63 vol=3.8x ATR=26.06 |
| Stop hit — per-position SL triggered | 2025-09-22 05:30:00 | 884.70 | 836.11 | 884.20 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 873.05 | 839.22 | 850.75 | Stage2 pullback-breakout RSI=56 vol=4.1x ATR=23.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 05:30:00 | 919.82 | 841.22 | 865.55 | T1 booked 50% @ 919.82 |
| Target hit | 2025-11-19 05:30:00 | 896.75 | 850.62 | 901.18 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-30 05:30:00 | 941.60 | 2025-08-01 05:30:00 | 893.26 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest1 | 2025-09-15 05:30:00 | 923.80 | 2025-09-22 05:30:00 | 884.70 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest1 | 2025-10-27 05:30:00 | 873.05 | 2025-10-30 05:30:00 | 919.82 | PARTIAL | 0.50 | 5.36% |
| BUY | retest1 | 2025-10-27 05:30:00 | 873.05 | 2025-11-19 05:30:00 | 896.75 | TARGET_HIT | 0.50 | 2.71% |
