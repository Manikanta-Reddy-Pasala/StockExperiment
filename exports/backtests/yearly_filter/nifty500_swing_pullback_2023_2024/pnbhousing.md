# PNB Housing Finance Ltd. (PNBHOUSING)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1076.90
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** 0.67% / 0.00%
- **Sum % (uncompounded):** 4.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | 0.67% | 4.0% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | 0.67% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | 0.67% | 4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 00:00:00 | 659.70 | 490.05 | 629.32 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=22.67 |
| Stop hit — per-position SL triggered | 2023-08-28 00:00:00 | 652.05 | 505.17 | 641.57 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 742.30 | 541.87 | 679.91 | Stage2 pullback-breakout RSI=66 vol=8.1x ATR=30.95 |
| Stop hit — per-position SL triggered | 2023-10-17 00:00:00 | 746.60 | 559.30 | 709.68 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 00:00:00 | 790.70 | 592.19 | 741.47 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=26.96 |
| Stop hit — per-position SL triggered | 2023-11-30 00:00:00 | 773.60 | 611.56 | 774.56 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-01-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 00:00:00 | 848.00 | 660.54 | 793.96 | Stage2 pullback-breakout RSI=69 vol=11.6x ATR=29.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 00:00:00 | 906.19 | 670.24 | 820.34 | T1 booked 50% @ 906.19 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 848.00 | 672.13 | 824.10 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 749.30 | 687.20 | 667.98 | Stage2 pullback-breakout RSI=62 vol=10.8x ATR=34.75 |
| Stop hit — per-position SL triggered | 2024-04-16 00:00:00 | 748.45 | 693.64 | 723.20 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-11 00:00:00 | 659.70 | 2023-08-28 00:00:00 | 652.05 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest1 | 2023-10-03 00:00:00 | 742.30 | 2023-10-17 00:00:00 | 746.60 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest1 | 2023-11-15 00:00:00 | 790.70 | 2023-11-30 00:00:00 | 773.60 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest1 | 2024-01-16 00:00:00 | 848.00 | 2024-01-23 00:00:00 | 906.19 | PARTIAL | 0.50 | 6.86% |
| BUY | retest1 | 2024-01-16 00:00:00 | 848.00 | 2024-01-24 00:00:00 | 848.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-01 00:00:00 | 749.30 | 2024-04-16 00:00:00 | 748.45 | STOP_HIT | 1.00 | -0.11% |
