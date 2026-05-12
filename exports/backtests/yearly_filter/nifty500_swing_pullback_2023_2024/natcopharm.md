# NATCO Pharma Ltd. (NATCOPHARM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1155.10
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
| TARGET_HIT | 3 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 1
- **Target hits / Stop hits / Partials:** 3 / 1 / 3
- **Avg / median % per leg:** 6.04% / 5.30%
- **Sum % (uncompounded):** 42.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 3 | 1 | 3 | 6.04% | 42.3% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 3 | 1 | 3 | 6.04% | 42.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 6 | 85.7% | 3 | 1 | 3 | 6.04% | 42.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 714.00 | 602.68 | 673.30 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=19.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 00:00:00 | 752.81 | 607.49 | 690.82 | T1 booked 50% @ 752.81 |
| Target hit | 2023-09-11 00:00:00 | 813.30 | 683.59 | 873.59 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 00:00:00 | 792.20 | 742.40 | 777.52 | Stage2 pullback-breakout RSI=55 vol=2.5x ATR=20.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 00:00:00 | 833.73 | 746.34 | 792.71 | T1 booked 50% @ 833.73 |
| Target hit | 2024-01-17 00:00:00 | 818.30 | 756.23 | 824.39 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 00:00:00 | 859.20 | 760.24 | 830.38 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=19.35 |
| Stop hit — per-position SL triggered | 2024-02-08 00:00:00 | 856.65 | 770.33 | 852.25 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-02-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 00:00:00 | 884.80 | 773.61 | 853.41 | Stage2 pullback-breakout RSI=63 vol=4.7x ATR=23.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 00:00:00 | 931.66 | 777.24 | 872.80 | T1 booked 50% @ 931.66 |
| Target hit | 2024-03-12 00:00:00 | 967.90 | 813.31 | 978.80 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 00:00:00 | 714.00 | 2023-07-20 00:00:00 | 752.81 | PARTIAL | 0.50 | 5.44% |
| BUY | retest1 | 2023-07-14 00:00:00 | 714.00 | 2023-09-11 00:00:00 | 813.30 | TARGET_HIT | 0.50 | 13.91% |
| BUY | retest1 | 2023-12-22 00:00:00 | 792.20 | 2024-01-02 00:00:00 | 833.73 | PARTIAL | 0.50 | 5.24% |
| BUY | retest1 | 2023-12-22 00:00:00 | 792.20 | 2024-01-17 00:00:00 | 818.30 | TARGET_HIT | 0.50 | 3.29% |
| BUY | retest1 | 2024-01-24 00:00:00 | 859.20 | 2024-02-08 00:00:00 | 856.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-14 00:00:00 | 884.80 | 2024-02-16 00:00:00 | 931.66 | PARTIAL | 0.50 | 5.30% |
| BUY | retest1 | 2024-02-14 00:00:00 | 884.80 | 2024-03-12 00:00:00 | 967.90 | TARGET_HIT | 0.50 | 9.39% |
