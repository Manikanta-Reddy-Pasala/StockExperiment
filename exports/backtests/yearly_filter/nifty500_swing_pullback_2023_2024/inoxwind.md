# Inox Wind Ltd. (INOXWIND)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 101.26
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 4
- **Avg / median % per leg:** 15.99% / 8.24%
- **Sum % (uncompounded):** 143.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 1 | 4 | 4 | 15.99% | 143.9% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 4 | 4 | 15.99% | 143.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 1 | 4 | 4 | 15.99% | 143.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 51.59 | 39.26 | 48.79 | Stage2 pullback-breakout RSI=61 vol=3.5x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 00:00:00 | 55.84 | 40.07 | 50.00 | T1 booked 50% @ 55.84 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 51.59 | 40.45 | 50.76 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2023-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 00:00:00 | 55.11 | 40.99 | 51.14 | Stage2 pullback-breakout RSI=63 vol=4.0x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-12 00:00:00 | 61.13 | 42.49 | 55.13 | T1 booked 50% @ 61.13 |
| Target hit | 2024-01-10 00:00:00 | 112.31 | 60.36 | 112.87 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 125.37 | 68.10 | 114.23 | Stage2 pullback-breakout RSI=65 vol=2.3x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 00:00:00 | 139.21 | 70.65 | 120.49 | T1 booked 50% @ 139.21 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 125.37 | 71.20 | 121.03 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-02-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 00:00:00 | 145.63 | 74.22 | 126.22 | Stage2 pullback-breakout RSI=70 vol=1.6x ATR=8.97 |
| Stop hit — per-position SL triggered | 2024-03-01 00:00:00 | 144.16 | 81.47 | 141.22 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 00:00:00 | 146.00 | 94.21 | 133.29 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=7.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 00:00:00 | 161.91 | 95.91 | 138.19 | T1 booked 50% @ 161.91 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 146.00 | 99.70 | 144.66 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-06 00:00:00 | 51.59 | 2023-10-17 00:00:00 | 55.84 | PARTIAL | 0.50 | 8.24% |
| BUY | retest1 | 2023-10-06 00:00:00 | 51.59 | 2023-10-20 00:00:00 | 51.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-30 00:00:00 | 55.11 | 2023-11-12 00:00:00 | 61.13 | PARTIAL | 0.50 | 10.91% |
| BUY | retest1 | 2023-10-30 00:00:00 | 55.11 | 2024-01-10 00:00:00 | 112.31 | TARGET_HIT | 0.50 | 103.79% |
| BUY | retest1 | 2024-02-02 00:00:00 | 125.37 | 2024-02-08 00:00:00 | 139.21 | PARTIAL | 0.50 | 11.04% |
| BUY | retest1 | 2024-02-02 00:00:00 | 125.37 | 2024-02-09 00:00:00 | 125.37 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-16 00:00:00 | 145.63 | 2024-03-01 00:00:00 | 144.16 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest1 | 2024-04-22 00:00:00 | 146.00 | 2024-04-25 00:00:00 | 161.91 | PARTIAL | 0.50 | 10.90% |
| BUY | retest1 | 2024-04-22 00:00:00 | 146.00 | 2024-05-07 00:00:00 | 146.00 | STOP_HIT | 0.50 | 0.00% |
