# Bharat Dynamics Ltd. (BDL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1415.00
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
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 7.30% / 0.00%
- **Sum % (uncompounded):** 65.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 7.30% | 65.7% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 7.30% | 65.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 1 | 5 | 3 | 7.30% | 65.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 00:00:00 | 615.80 | 493.29 | 563.49 | Stage2 pullback-breakout RSI=65 vol=8.4x ATR=25.36 |
| Stop hit — per-position SL triggered | 2023-07-18 00:00:00 | 577.76 | 498.23 | 575.00 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 588.78 | 520.41 | 571.35 | Stage2 pullback-breakout RSI=58 vol=4.0x ATR=18.66 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 560.79 | 523.81 | 573.96 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 519.17 | 517.76 | 499.36 | Stage2 pullback-breakout RSI=57 vol=4.1x ATR=17.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 00:00:00 | 554.53 | 519.09 | 520.84 | T1 booked 50% @ 554.53 |
| Target hit | 2024-01-23 00:00:00 | 834.83 | 607.46 | 838.61 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 879.70 | 623.96 | 846.06 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=34.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 00:00:00 | 948.79 | 634.70 | 864.08 | T1 booked 50% @ 948.79 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 879.70 | 639.23 | 863.50 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 943.13 | 660.45 | 868.56 | Stage2 pullback-breakout RSI=63 vol=2.2x ATR=45.39 |
| Stop hit — per-position SL triggered | 2024-03-11 00:00:00 | 884.30 | 684.13 | 892.15 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 00:00:00 | 923.45 | 720.71 | 876.72 | Stage2 pullback-breakout RSI=63 vol=4.2x ATR=34.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 00:00:00 | 992.05 | 731.71 | 906.15 | T1 booked 50% @ 992.05 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 923.45 | 750.88 | 944.13 | SL hit (bars_held=13) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 00:00:00 | 615.80 | 2023-07-18 00:00:00 | 577.76 | STOP_HIT | 1.00 | -6.18% |
| BUY | retest1 | 2023-09-04 00:00:00 | 588.78 | 2023-09-12 00:00:00 | 560.79 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest1 | 2023-11-03 00:00:00 | 519.17 | 2023-11-16 00:00:00 | 554.53 | PARTIAL | 0.50 | 6.81% |
| BUY | retest1 | 2023-11-03 00:00:00 | 519.17 | 2024-01-23 00:00:00 | 834.83 | TARGET_HIT | 0.50 | 60.80% |
| BUY | retest1 | 2024-02-02 00:00:00 | 879.70 | 2024-02-08 00:00:00 | 948.79 | PARTIAL | 0.50 | 7.85% |
| BUY | retest1 | 2024-02-02 00:00:00 | 879.70 | 2024-02-12 00:00:00 | 879.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-26 00:00:00 | 943.13 | 2024-03-11 00:00:00 | 884.30 | STOP_HIT | 1.00 | -6.24% |
| BUY | retest1 | 2024-04-16 00:00:00 | 923.45 | 2024-04-24 00:00:00 | 992.05 | PARTIAL | 0.50 | 7.43% |
| BUY | retest1 | 2024-04-16 00:00:00 | 923.45 | 2024-05-07 00:00:00 | 923.45 | STOP_HIT | 0.50 | 0.00% |
