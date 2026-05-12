# Elgi Equipments Ltd. (ELGIEQUIP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 546.20
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 1
- **Avg / median % per leg:** -1.66% / -4.38%
- **Sum % (uncompounded):** -13.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -1.66% | -13.3% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | -1.66% | -13.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 2 | 25.0% | 1 | 6 | 1 | -1.66% | -13.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 00:00:00 | 583.20 | 493.21 | 542.50 | Stage2 pullback-breakout RSI=70 vol=10.8x ATR=20.53 |
| Stop hit — per-position SL triggered | 2023-07-25 00:00:00 | 552.40 | 499.34 | 554.70 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-10-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 00:00:00 | 510.55 | 499.31 | 496.07 | Stage2 pullback-breakout RSI=57 vol=2.2x ATR=16.22 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 486.22 | 499.09 | 494.54 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-11-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 00:00:00 | 534.70 | 499.44 | 502.90 | Stage2 pullback-breakout RSI=66 vol=4.9x ATR=15.63 |
| Stop hit — per-position SL triggered | 2023-11-13 00:00:00 | 511.26 | 500.15 | 508.13 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2023-12-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 00:00:00 | 536.40 | 503.40 | 519.76 | Stage2 pullback-breakout RSI=62 vol=5.6x ATR=14.47 |
| Stop hit — per-position SL triggered | 2023-12-18 00:00:00 | 514.69 | 505.29 | 526.26 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2024-01-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 00:00:00 | 554.10 | 512.54 | 539.61 | Stage2 pullback-breakout RSI=62 vol=7.1x ATR=15.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 00:00:00 | 584.98 | 513.83 | 546.57 | T1 booked 50% @ 584.98 |
| Target hit | 2024-02-26 00:00:00 | 621.90 | 536.58 | 625.32 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-03-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 00:00:00 | 680.60 | 541.00 | 633.82 | Stage2 pullback-breakout RSI=65 vol=2.7x ATR=28.87 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 637.29 | 550.62 | 657.59 | SL hit (bars_held=7) |

### Cycle 7 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 653.35 | 558.54 | 617.08 | Stage2 pullback-breakout RSI=59 vol=9.4x ATR=27.38 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 612.28 | 562.87 | 624.10 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 00:00:00 | 583.20 | 2023-07-25 00:00:00 | 552.40 | STOP_HIT | 1.00 | -5.28% |
| BUY | retest1 | 2023-10-19 00:00:00 | 510.55 | 2023-10-23 00:00:00 | 486.22 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest1 | 2023-11-09 00:00:00 | 534.70 | 2023-11-13 00:00:00 | 511.26 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest1 | 2023-12-08 00:00:00 | 536.40 | 2023-12-18 00:00:00 | 514.69 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest1 | 2024-01-23 00:00:00 | 554.10 | 2024-01-25 00:00:00 | 584.98 | PARTIAL | 0.50 | 5.57% |
| BUY | retest1 | 2024-01-23 00:00:00 | 554.10 | 2024-02-26 00:00:00 | 621.90 | TARGET_HIT | 0.50 | 12.24% |
| BUY | retest1 | 2024-03-01 00:00:00 | 680.60 | 2024-03-12 00:00:00 | 637.29 | STOP_HIT | 1.00 | -6.36% |
| BUY | retest1 | 2024-04-04 00:00:00 | 653.35 | 2024-04-15 00:00:00 | 612.28 | STOP_HIT | 1.00 | -6.29% |
