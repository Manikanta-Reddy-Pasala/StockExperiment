# L&T Technology Services Ltd. (LTTS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 3802.60
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 3.00% / 0.00%
- **Sum % (uncompounded):** 21.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 3.00% | 21.0% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 3.00% | 21.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 3.00% | 21.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 4094.95 | 3722.04 | 3921.45 | Stage2 pullback-breakout RSI=63 vol=2.8x ATR=100.72 |
| Stop hit — per-position SL triggered | 2023-07-25 00:00:00 | 3943.87 | 3744.71 | 3985.79 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-08-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 00:00:00 | 4164.50 | 3758.08 | 4002.84 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=102.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 00:00:00 | 4369.76 | 3780.27 | 4087.93 | T1 booked 50% @ 4369.76 |
| Stop hit — per-position SL triggered | 2023-08-14 00:00:00 | 4164.50 | 3798.98 | 4143.46 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2023-10-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 00:00:00 | 4746.95 | 4033.61 | 4620.34 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=100.04 |
| Stop hit — per-position SL triggered | 2023-10-18 00:00:00 | 4596.90 | 4076.08 | 4630.35 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2023-11-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 00:00:00 | 4391.00 | 4105.71 | 4317.19 | Stage2 pullback-breakout RSI=54 vol=1.5x ATR=79.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 00:00:00 | 4550.57 | 4116.54 | 4358.17 | T1 booked 50% @ 4550.57 |
| Target hit | 2024-02-14 00:00:00 | 5402.75 | 4617.20 | 5475.17 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-03-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 00:00:00 | 5419.35 | 4758.12 | 5325.16 | Stage2 pullback-breakout RSI=56 vol=2.3x ATR=134.76 |
| Stop hit — per-position SL triggered | 2024-03-19 00:00:00 | 5217.21 | 4768.62 | 5317.67 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 00:00:00 | 4094.95 | 2023-07-25 00:00:00 | 3943.87 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest1 | 2023-08-01 00:00:00 | 4164.50 | 2023-08-08 00:00:00 | 4369.76 | PARTIAL | 0.50 | 4.93% |
| BUY | retest1 | 2023-08-01 00:00:00 | 4164.50 | 2023-08-14 00:00:00 | 4164.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-09 00:00:00 | 4746.95 | 2023-10-18 00:00:00 | 4596.90 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest1 | 2023-11-15 00:00:00 | 4391.00 | 2023-11-20 00:00:00 | 4550.57 | PARTIAL | 0.50 | 3.63% |
| BUY | retest1 | 2023-11-15 00:00:00 | 4391.00 | 2024-02-14 00:00:00 | 5402.75 | TARGET_HIT | 0.50 | 23.04% |
| BUY | retest1 | 2024-03-15 00:00:00 | 5419.35 | 2024-03-19 00:00:00 | 5217.21 | STOP_HIT | 1.00 | -3.73% |
