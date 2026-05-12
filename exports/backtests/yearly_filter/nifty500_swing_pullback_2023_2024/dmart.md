# Avenue Supermarts Ltd. (DMART)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 4376.50
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 3.78% / 3.73%
- **Sum % (uncompounded):** 18.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.78% | 18.9% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.78% | 18.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.78% | 18.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 3949.75 | 3762.76 | 3813.67 | Stage2 pullback-breakout RSI=69 vol=2.7x ATR=73.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 00:00:00 | 4097.21 | 3773.38 | 3888.23 | T1 booked 50% @ 4097.21 |
| Stop hit — per-position SL triggered | 2023-12-13 00:00:00 | 3949.75 | 3787.97 | 3959.67 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2024-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 00:00:00 | 4103.35 | 3817.86 | 4018.58 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=83.25 |
| Stop hit — per-position SL triggered | 2024-01-03 00:00:00 | 3978.47 | 3818.99 | 4010.37 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2024-03-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 00:00:00 | 3979.65 | 3810.32 | 3848.57 | Stage2 pullback-breakout RSI=68 vol=2.0x ATR=75.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 00:00:00 | 4131.23 | 3826.84 | 3950.83 | T1 booked 50% @ 4131.23 |
| Target hit | 2024-04-26 00:00:00 | 4553.15 | 3983.27 | 4582.85 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-05-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 00:00:00 | 4806.75 | 4026.73 | 4611.78 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=133.41 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-30 00:00:00 | 3949.75 | 2023-12-06 00:00:00 | 4097.21 | PARTIAL | 0.50 | 3.73% |
| BUY | retest1 | 2023-11-30 00:00:00 | 3949.75 | 2023-12-13 00:00:00 | 3949.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-02 00:00:00 | 4103.35 | 2024-01-03 00:00:00 | 3978.47 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest1 | 2024-03-11 00:00:00 | 3979.65 | 2024-03-21 00:00:00 | 4131.23 | PARTIAL | 0.50 | 3.81% |
| BUY | retest1 | 2024-03-11 00:00:00 | 3979.65 | 2024-04-26 00:00:00 | 4553.15 | TARGET_HIT | 0.50 | 14.41% |
