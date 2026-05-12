# Cartrade Tech Ltd. (CARTRADE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1855.50
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
- **Avg / median % per leg:** 2.04% / 5.36%
- **Sum % (uncompounded):** 10.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.04% | 10.2% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.04% | 10.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 2.04% | 10.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 584.80 | 511.97 | 558.52 | Stage2 pullback-breakout RSI=60 vol=1.5x ATR=20.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 00:00:00 | 625.41 | 516.58 | 578.12 | T1 booked 50% @ 625.41 |
| Target hit | 2023-10-23 00:00:00 | 616.15 | 528.66 | 623.72 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-18 00:00:00 | 726.70 | 624.36 | 716.82 | Stage2 pullback-breakout RSI=52 vol=2.6x ATR=22.82 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 692.47 | 627.99 | 716.90 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 760.30 | 643.73 | 706.78 | Stage2 pullback-breakout RSI=69 vol=3.5x ATR=24.39 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 723.72 | 652.72 | 735.13 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-05-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 00:00:00 | 749.95 | 664.61 | 708.32 | Stage2 pullback-breakout RSI=61 vol=3.3x ATR=27.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 00:00:00 | 805.51 | 666.06 | 717.98 | T1 booked 50% @ 805.51 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-03 00:00:00 | 584.80 | 2023-10-10 00:00:00 | 625.41 | PARTIAL | 0.50 | 6.94% |
| BUY | retest1 | 2023-10-03 00:00:00 | 584.80 | 2023-10-23 00:00:00 | 616.15 | TARGET_HIT | 0.50 | 5.36% |
| BUY | retest1 | 2024-01-18 00:00:00 | 726.70 | 2024-01-24 00:00:00 | 692.47 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest1 | 2024-02-26 00:00:00 | 760.30 | 2024-03-06 00:00:00 | 723.72 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest1 | 2024-05-03 00:00:00 | 749.95 | 2024-05-06 00:00:00 | 805.51 | PARTIAL | 0.50 | 7.41% |
