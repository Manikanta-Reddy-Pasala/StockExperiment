# Marico Ltd. (MARICO)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 831.30
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
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 1.04% / 2.60%
- **Sum % (uncompounded):** 6.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.04% | 6.2% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.04% | 6.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 1.04% | 6.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 05:30:00 | 641.30 | 562.09 | 616.39 | Stage2 pullback-breakout RSI=62 vol=2.5x ATR=16.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 05:30:00 | 674.30 | 568.38 | 637.18 | T1 booked 50% @ 674.30 |
| Stop hit — per-position SL triggered | 2024-07-24 05:30:00 | 657.95 | 572.24 | 646.83 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 05:30:00 | 665.25 | 596.38 | 658.68 | Stage2 pullback-breakout RSI=53 vol=2.8x ATR=16.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 05:30:00 | 698.89 | 602.35 | 672.27 | T1 booked 50% @ 698.89 |
| Target hit | 2024-10-07 05:30:00 | 678.80 | 613.68 | 687.89 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 05:30:00 | 649.90 | 621.05 | 622.80 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=17.94 |
| Stop hit — per-position SL triggered | 2024-12-09 05:30:00 | 622.99 | 622.14 | 628.52 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2025-02-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 05:30:00 | 694.20 | 632.21 | 664.35 | Stage2 pullback-breakout RSI=68 vol=2.4x ATR=20.57 |
| Stop hit — per-position SL triggered | 2025-02-03 05:30:00 | 663.34 | 632.62 | 665.21 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-08 05:30:00 | 641.30 | 2024-07-18 05:30:00 | 674.30 | PARTIAL | 0.50 | 5.15% |
| BUY | retest1 | 2024-07-08 05:30:00 | 641.30 | 2024-07-24 05:30:00 | 657.95 | STOP_HIT | 0.50 | 2.60% |
| BUY | retest1 | 2024-09-06 05:30:00 | 665.25 | 2024-09-17 05:30:00 | 698.89 | PARTIAL | 0.50 | 5.06% |
| BUY | retest1 | 2024-09-06 05:30:00 | 665.25 | 2024-10-07 05:30:00 | 678.80 | TARGET_HIT | 0.50 | 2.04% |
| BUY | retest1 | 2024-11-27 05:30:00 | 649.90 | 2024-12-09 05:30:00 | 622.99 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest1 | 2025-02-01 05:30:00 | 694.20 | 2025-02-03 05:30:00 | 663.34 | STOP_HIT | 1.00 | -4.45% |
