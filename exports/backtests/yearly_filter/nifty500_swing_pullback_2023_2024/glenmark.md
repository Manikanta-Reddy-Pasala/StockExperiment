# Glenmark Pharmaceuticals Ltd. (GLENMARK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2326.70
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 3.00% / 3.00%
- **Sum % (uncompounded):** 27.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 3.00% | 27.0% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 3.00% | 27.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 2 | 4 | 3 | 3.00% | 27.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 00:00:00 | 655.70 | 487.66 | 632.46 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=17.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 00:00:00 | 690.05 | 499.91 | 651.11 | T1 booked 50% @ 690.05 |
| Target hit | 2023-08-17 00:00:00 | 779.30 | 565.67 | 782.38 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 791.75 | 589.60 | 768.50 | Stage2 pullback-breakout RSI=59 vol=2.1x ATR=21.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 00:00:00 | 835.57 | 602.36 | 788.26 | T1 booked 50% @ 835.57 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 791.75 | 616.38 | 811.64 | SL hit (bars_held=12) |

### Cycle 3 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 855.50 | 624.93 | 804.93 | Stage2 pullback-breakout RSI=62 vol=4.2x ATR=31.25 |
| Stop hit — per-position SL triggered | 2023-10-04 00:00:00 | 808.62 | 628.89 | 808.39 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 785.95 | 659.37 | 768.29 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=20.27 |
| Stop hit — per-position SL triggered | 2023-11-12 00:00:00 | 755.54 | 662.79 | 769.82 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-01-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 00:00:00 | 910.45 | 720.15 | 863.95 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=26.06 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 871.36 | 723.32 | 866.91 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-02-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 00:00:00 | 872.25 | 750.58 | 857.80 | Stage2 pullback-breakout RSI=53 vol=5.7x ATR=35.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 00:00:00 | 942.54 | 768.78 | 898.08 | T1 booked 50% @ 942.54 |
| Target hit | 2024-03-13 00:00:00 | 898.45 | 778.34 | 913.58 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-27 00:00:00 | 655.70 | 2023-07-07 00:00:00 | 690.05 | PARTIAL | 0.50 | 5.24% |
| BUY | retest1 | 2023-06-27 00:00:00 | 655.70 | 2023-08-17 00:00:00 | 779.30 | TARGET_HIT | 0.50 | 18.85% |
| BUY | retest1 | 2023-09-05 00:00:00 | 791.75 | 2023-09-13 00:00:00 | 835.57 | PARTIAL | 0.50 | 5.53% |
| BUY | retest1 | 2023-09-05 00:00:00 | 791.75 | 2023-09-22 00:00:00 | 791.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-29 00:00:00 | 855.50 | 2023-10-04 00:00:00 | 808.62 | STOP_HIT | 1.00 | -5.48% |
| BUY | retest1 | 2023-11-08 00:00:00 | 785.95 | 2023-11-12 00:00:00 | 755.54 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest1 | 2024-01-15 00:00:00 | 910.45 | 2024-01-17 00:00:00 | 871.36 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest1 | 2024-02-16 00:00:00 | 872.25 | 2024-03-04 00:00:00 | 942.54 | PARTIAL | 0.50 | 8.06% |
| BUY | retest1 | 2024-02-16 00:00:00 | 872.25 | 2024-03-13 00:00:00 | 898.45 | TARGET_HIT | 0.50 | 3.00% |
