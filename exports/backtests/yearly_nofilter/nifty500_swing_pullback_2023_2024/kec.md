# Kec International Ltd. (KEC)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 598.05
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 4.04% / 5.39%
- **Sum % (uncompounded):** 28.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 1 | 3 | 3 | 4.04% | 28.3% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 1 | 3 | 3 | 4.04% | 28.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 1 | 3 | 3 | 4.04% | 28.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 00:00:00 | 575.00 | 480.76 | 550.79 | Stage2 pullback-breakout RSI=63 vol=2.3x ATR=17.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 00:00:00 | 610.00 | 483.98 | 561.10 | T1 booked 50% @ 610.00 |
| Target hit | 2023-08-11 00:00:00 | 623.35 | 515.15 | 626.70 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 00:00:00 | 612.25 | 583.12 | 602.29 | Stage2 pullback-breakout RSI=55 vol=2.8x ATR=14.99 |
| Stop hit — per-position SL triggered | 2024-01-20 00:00:00 | 616.95 | 586.24 | 611.62 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 00:00:00 | 657.65 | 588.68 | 619.89 | Stage2 pullback-breakout RSI=70 vol=5.6x ATR=17.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-14 00:00:00 | 693.09 | 595.81 | 648.81 | T1 booked 50% @ 693.09 |
| Stop hit — per-position SL triggered | 2024-02-16 00:00:00 | 668.55 | 597.45 | 654.15 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-02-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 00:00:00 | 697.60 | 602.34 | 662.99 | Stage2 pullback-breakout RSI=66 vol=2.9x ATR=20.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 00:00:00 | 739.45 | 612.60 | 697.79 | T1 booked 50% @ 739.45 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 697.60 | 614.58 | 700.25 | SL hit (bars_held=11) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-04 00:00:00 | 575.00 | 2023-07-07 00:00:00 | 610.00 | PARTIAL | 0.50 | 6.09% |
| BUY | retest1 | 2023-07-04 00:00:00 | 575.00 | 2023-08-11 00:00:00 | 623.35 | TARGET_HIT | 0.50 | 8.41% |
| BUY | retest1 | 2024-01-08 00:00:00 | 612.25 | 2024-01-20 00:00:00 | 616.95 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest1 | 2024-01-31 00:00:00 | 657.65 | 2024-02-14 00:00:00 | 693.09 | PARTIAL | 0.50 | 5.39% |
| BUY | retest1 | 2024-01-31 00:00:00 | 657.65 | 2024-02-16 00:00:00 | 668.55 | STOP_HIT | 0.50 | 1.66% |
| BUY | retest1 | 2024-02-27 00:00:00 | 697.60 | 2024-03-11 00:00:00 | 739.45 | PARTIAL | 0.50 | 6.00% |
| BUY | retest1 | 2024-02-27 00:00:00 | 697.60 | 2024-03-13 00:00:00 | 697.60 | STOP_HIT | 0.50 | 0.00% |
