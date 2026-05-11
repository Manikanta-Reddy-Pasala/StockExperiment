# SBIN (SBIN)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1019.30
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
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 3.20% / 2.95%
- **Sum % (uncompounded):** 22.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 2 | 3 | 2 | 3.20% | 22.4% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 2 | 3 | 2 | 3.20% | 22.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 2 | 3 | 2 | 3.20% | 22.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 05:30:00 | 589.25 | 562.61 | 573.80 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=8.49 |
| Stop hit — per-position SL triggered | 2023-07-18 05:30:00 | 592.35 | 565.32 | 584.79 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-07-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 05:30:00 | 610.05 | 566.12 | 588.63 | Stage2 pullback-breakout RSI=68 vol=2.1x ATR=10.06 |
| Stop hit — per-position SL triggered | 2023-08-02 05:30:00 | 594.96 | 570.18 | 602.87 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2023-12-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 05:30:00 | 594.70 | 573.76 | 571.63 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=8.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 05:30:00 | 612.26 | 574.45 | 578.29 | T1 booked 50% @ 612.26 |
| Target hit | 2024-01-08 05:30:00 | 627.00 | 586.60 | 632.09 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 05:30:00 | 675.25 | 595.40 | 637.79 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=16.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 05:30:00 | 707.37 | 596.44 | 643.67 | T1 booked 50% @ 707.37 |
| Target hit | 2024-03-13 05:30:00 | 747.25 | 630.93 | 753.06 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 05:30:00 | 812.70 | 660.73 | 764.60 | Stage2 pullback-breakout RSI=70 vol=2.4x ATR=17.28 |
| Stop hit — per-position SL triggered | 2024-05-10 05:30:00 | 817.35 | 675.63 | 797.63 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-04 05:30:00 | 589.25 | 2023-07-18 05:30:00 | 592.35 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest1 | 2023-07-20 05:30:00 | 610.05 | 2023-08-02 05:30:00 | 594.96 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest1 | 2023-12-04 05:30:00 | 594.70 | 2023-12-06 05:30:00 | 612.26 | PARTIAL | 0.50 | 2.95% |
| BUY | retest1 | 2023-12-04 05:30:00 | 594.70 | 2024-01-08 05:30:00 | 627.00 | TARGET_HIT | 0.50 | 5.43% |
| BUY | retest1 | 2024-02-07 05:30:00 | 675.25 | 2024-02-08 05:30:00 | 707.37 | PARTIAL | 0.50 | 4.76% |
| BUY | retest1 | 2024-02-07 05:30:00 | 675.25 | 2024-03-13 05:30:00 | 747.25 | TARGET_HIT | 0.50 | 10.66% |
| BUY | retest1 | 2024-04-25 05:30:00 | 812.70 | 2024-05-10 05:30:00 | 817.35 | STOP_HIT | 1.00 | 0.57% |
