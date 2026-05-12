# Central Depository Services (India) Ltd. (CDSL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1233.70
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
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 1
- **Target hits / Stop hits / Partials:** 4 / 1 / 4
- **Avg / median % per leg:** 7.42% / 6.67%
- **Sum % (uncompounded):** 66.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 8 | 88.9% | 4 | 1 | 4 | 7.42% | 66.8% |
| BUY @ 2nd Alert (retest1) | 9 | 8 | 88.9% | 4 | 1 | 4 | 7.42% | 66.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 8 | 88.9% | 4 | 1 | 4 | 7.42% | 66.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 00:00:00 | 612.55 | 551.59 | 588.18 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=14.62 |
| Stop hit — per-position SL triggered | 2023-08-10 00:00:00 | 599.88 | 557.16 | 601.50 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 612.73 | 560.48 | 582.64 | Stage2 pullback-breakout RSI=65 vol=5.9x ATR=16.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 00:00:00 | 645.28 | 562.17 | 592.01 | T1 booked 50% @ 645.28 |
| Target hit | 2023-10-23 00:00:00 | 653.58 | 588.97 | 666.12 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-10-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 00:00:00 | 728.08 | 593.30 | 671.94 | Stage2 pullback-breakout RSI=67 vol=4.5x ATR=27.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 00:00:00 | 782.35 | 596.52 | 687.18 | T1 booked 50% @ 782.35 |
| Target hit | 2023-12-20 00:00:00 | 903.48 | 684.85 | 926.14 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 00:00:00 | 905.48 | 742.14 | 903.33 | Stage2 pullback-breakout RSI=50 vol=1.9x ATR=26.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 00:00:00 | 957.79 | 749.85 | 915.60 | T1 booked 50% @ 957.79 |
| Target hit | 2024-02-12 00:00:00 | 924.93 | 756.24 | 928.07 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 918.85 | 798.43 | 887.29 | Stage2 pullback-breakout RSI=55 vol=2.1x ATR=35.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 00:00:00 | 989.32 | 809.85 | 921.78 | T1 booked 50% @ 989.32 |
| Target hit | 2024-05-10 00:00:00 | 1008.23 | 846.91 | 1021.72 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-27 00:00:00 | 612.55 | 2023-08-10 00:00:00 | 599.88 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest1 | 2023-09-05 00:00:00 | 612.73 | 2023-09-08 00:00:00 | 645.28 | PARTIAL | 0.50 | 5.31% |
| BUY | retest1 | 2023-09-05 00:00:00 | 612.73 | 2023-10-23 00:00:00 | 653.58 | TARGET_HIT | 0.50 | 6.67% |
| BUY | retest1 | 2023-10-31 00:00:00 | 728.08 | 2023-11-02 00:00:00 | 782.35 | PARTIAL | 0.50 | 7.45% |
| BUY | retest1 | 2023-10-31 00:00:00 | 728.08 | 2023-12-20 00:00:00 | 903.48 | TARGET_HIT | 0.50 | 24.09% |
| BUY | retest1 | 2024-02-01 00:00:00 | 905.48 | 2024-02-07 00:00:00 | 957.79 | PARTIAL | 0.50 | 5.78% |
| BUY | retest1 | 2024-02-01 00:00:00 | 905.48 | 2024-02-12 00:00:00 | 924.93 | TARGET_HIT | 0.50 | 2.15% |
| BUY | retest1 | 2024-04-01 00:00:00 | 918.85 | 2024-04-12 00:00:00 | 989.32 | PARTIAL | 0.50 | 7.67% |
| BUY | retest1 | 2024-04-01 00:00:00 | 918.85 | 2024-05-10 00:00:00 | 1008.23 | TARGET_HIT | 0.50 | 9.73% |
