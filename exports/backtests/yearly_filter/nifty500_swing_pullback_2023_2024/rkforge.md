# Ramkrishna Forgings Ltd. (RKFORGE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 610.25
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 3.89% / 7.96%
- **Sum % (uncompounded):** 23.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.89% | 23.3% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.89% | 23.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 3.89% | 23.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 660.00 | 483.50 | 643.83 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=30.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 00:00:00 | 720.14 | 505.99 | 683.41 | T1 booked 50% @ 720.14 |
| Target hit | 2023-12-08 00:00:00 | 738.95 | 540.94 | 750.58 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 00:00:00 | 772.90 | 575.69 | 734.46 | Stage2 pullback-breakout RSI=63 vol=3.4x ATR=27.43 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 731.76 | 591.24 | 756.89 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2024-02-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 00:00:00 | 761.05 | 621.07 | 741.30 | Stage2 pullback-breakout RSI=56 vol=2.6x ATR=29.25 |
| Stop hit — per-position SL triggered | 2024-03-04 00:00:00 | 758.00 | 634.62 | 756.51 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 00:00:00 | 725.00 | 646.97 | 697.96 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=28.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 00:00:00 | 782.73 | 649.05 | 707.94 | T1 booked 50% @ 782.73 |
| Stop hit — per-position SL triggered | 2024-04-22 00:00:00 | 725.00 | 650.03 | 711.70 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-02 00:00:00 | 660.00 | 2023-11-17 00:00:00 | 720.14 | PARTIAL | 0.50 | 9.11% |
| BUY | retest1 | 2023-11-02 00:00:00 | 660.00 | 2023-12-08 00:00:00 | 738.95 | TARGET_HIT | 0.50 | 11.96% |
| BUY | retest1 | 2024-01-08 00:00:00 | 772.90 | 2024-01-18 00:00:00 | 731.76 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest1 | 2024-02-20 00:00:00 | 761.05 | 2024-03-04 00:00:00 | 758.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-04-16 00:00:00 | 725.00 | 2024-04-19 00:00:00 | 782.73 | PARTIAL | 0.50 | 7.96% |
| BUY | retest1 | 2024-04-16 00:00:00 | 725.00 | 2024-04-22 00:00:00 | 725.00 | STOP_HIT | 0.50 | 0.00% |
