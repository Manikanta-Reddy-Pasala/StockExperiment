# Welspun Corp Ltd. (WELCORP)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 2 / 3 / 3
- **Avg / median % per leg:** 3.86% / 6.29%
- **Sum % (uncompounded):** 30.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 2 | 3 | 3 | 3.86% | 30.9% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 2 | 3 | 3 | 3.86% | 30.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 2 | 3 | 3 | 3.86% | 30.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 00:00:00 | 592.35 | 516.22 | 551.89 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=23.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 00:00:00 | 640.07 | 520.32 | 574.96 | T1 booked 50% @ 640.07 |
| Target hit | 2024-08-05 00:00:00 | 630.00 | 541.26 | 634.11 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-08-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 00:00:00 | 696.80 | 546.93 | 644.30 | Stage2 pullback-breakout RSI=67 vol=3.7x ATR=30.60 |
| Stop hit — per-position SL triggered | 2024-08-27 00:00:00 | 717.35 | 562.32 | 687.58 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 00:00:00 | 719.65 | 585.47 | 687.45 | Stage2 pullback-breakout RSI=62 vol=3.8x ATR=22.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 00:00:00 | 764.93 | 591.44 | 704.69 | T1 booked 50% @ 764.93 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 719.65 | 595.58 | 711.50 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 772.85 | 620.64 | 724.87 | Stage2 pullback-breakout RSI=65 vol=1.9x ATR=26.48 |
| Stop hit — per-position SL triggered | 2024-11-11 00:00:00 | 733.12 | 625.33 | 738.78 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 747.05 | 631.13 | 718.67 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=31.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 00:00:00 | 810.34 | 645.37 | 760.08 | T1 booked 50% @ 810.34 |
| Target hit | 2024-12-20 00:00:00 | 776.35 | 657.91 | 777.44 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-03 00:00:00 | 592.35 | 2024-07-09 00:00:00 | 640.07 | PARTIAL | 0.50 | 8.06% |
| BUY | retest1 | 2024-07-03 00:00:00 | 592.35 | 2024-08-05 00:00:00 | 630.00 | TARGET_HIT | 0.50 | 6.36% |
| BUY | retest1 | 2024-08-12 00:00:00 | 696.80 | 2024-08-27 00:00:00 | 717.35 | STOP_HIT | 1.00 | 2.95% |
| BUY | retest1 | 2024-09-24 00:00:00 | 719.65 | 2024-09-30 00:00:00 | 764.93 | PARTIAL | 0.50 | 6.29% |
| BUY | retest1 | 2024-09-24 00:00:00 | 719.65 | 2024-10-04 00:00:00 | 719.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-06 00:00:00 | 772.85 | 2024-11-11 00:00:00 | 733.12 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest1 | 2024-11-25 00:00:00 | 747.05 | 2024-12-09 00:00:00 | 810.34 | PARTIAL | 0.50 | 8.47% |
| BUY | retest1 | 2024-11-25 00:00:00 | 747.05 | 2024-12-20 00:00:00 | 776.35 | TARGET_HIT | 0.50 | 3.92% |
