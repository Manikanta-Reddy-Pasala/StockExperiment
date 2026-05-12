# Escorts Kubota Ltd. (ESCORTS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3025.40
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
- **Avg / median % per leg:** 10.43% / 4.92%
- **Sum % (uncompounded):** 41.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 10.43% | 41.7% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 10.43% | 41.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 10.43% | 41.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 00:00:00 | 2285.85 | 2074.81 | 2205.56 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=54.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 00:00:00 | 2394.76 | 2087.48 | 2257.76 | T1 booked 50% @ 2394.76 |
| Target hit | 2023-10-03 00:00:00 | 3124.65 | 2423.70 | 3132.47 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 00:00:00 | 3393.45 | 2463.81 | 3179.86 | Stage2 pullback-breakout RSI=67 vol=3.8x ATR=105.51 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 3235.19 | 2531.36 | 3263.43 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 2858.50 | 2803.21 | 2797.30 | Stage2 pullback-breakout RSI=56 vol=1.5x ATR=70.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 00:00:00 | 2999.11 | 2805.93 | 2823.87 | T1 booked 50% @ 2999.11 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-10 00:00:00 | 2285.85 | 2023-07-17 00:00:00 | 2394.76 | PARTIAL | 0.50 | 4.76% |
| BUY | retest1 | 2023-07-10 00:00:00 | 2285.85 | 2023-10-03 00:00:00 | 3124.65 | TARGET_HIT | 0.50 | 36.70% |
| BUY | retest1 | 2023-10-10 00:00:00 | 3393.45 | 2023-10-20 00:00:00 | 3235.19 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest1 | 2024-04-01 00:00:00 | 2858.50 | 2024-04-03 00:00:00 | 2999.11 | PARTIAL | 0.50 | 4.92% |
