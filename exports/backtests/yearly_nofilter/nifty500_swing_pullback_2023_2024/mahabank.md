# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 80.32
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 2 / 5 / 3
- **Avg / median % per leg:** 7.24% / 5.49%
- **Sum % (uncompounded):** 72.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 7.24% | 72.4% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 7.24% | 72.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 2 | 5 | 3 | 7.24% | 72.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 28.90 | 26.45 | 28.49 | Stage2 pullback-breakout RSI=52 vol=2.0x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 30.49 | 26.52 | 28.81 | T1 booked 50% @ 30.49 |
| Target hit | 2023-10-09 00:00:00 | 45.30 | 33.01 | 45.87 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 46.10 | 35.42 | 43.78 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-11-22 00:00:00 | 43.60 | 35.98 | 44.28 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 45.85 | 36.54 | 44.31 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 00:00:00 | 48.52 | 37.60 | 46.05 | T1 booked 50% @ 48.52 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 45.85 | 37.67 | 45.91 | SL hit (bars_held=12) |

### Cycle 4 — BUY (started 2024-01-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 00:00:00 | 47.30 | 38.36 | 45.81 | Stage2 pullback-breakout RSI=60 vol=2.2x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 00:00:00 | 50.20 | 39.02 | 46.56 | T1 booked 50% @ 50.20 |
| Target hit | 2024-02-12 00:00:00 | 55.30 | 41.97 | 56.18 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 63.60 | 44.90 | 59.99 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 59.43 | 45.54 | 60.29 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 62.35 | 46.93 | 59.58 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 61.65 | 48.62 | 62.67 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 67.65 | 49.80 | 63.56 | Stage2 pullback-breakout RSI=65 vol=3.1x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 63.88 | 51.16 | 65.47 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 28.90 | 2023-07-04 00:00:00 | 30.49 | PARTIAL | 0.50 | 5.49% |
| BUY | retest1 | 2023-06-30 00:00:00 | 28.90 | 2023-10-09 00:00:00 | 45.30 | TARGET_HIT | 0.50 | 56.75% |
| BUY | retest1 | 2023-11-13 00:00:00 | 46.10 | 2023-11-22 00:00:00 | 43.60 | STOP_HIT | 1.00 | -5.43% |
| BUY | retest1 | 2023-12-04 00:00:00 | 45.85 | 2023-12-19 00:00:00 | 48.52 | PARTIAL | 0.50 | 5.82% |
| BUY | retest1 | 2023-12-04 00:00:00 | 45.85 | 2023-12-20 00:00:00 | 45.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-03 00:00:00 | 47.30 | 2024-01-15 00:00:00 | 50.20 | PARTIAL | 0.50 | 6.13% |
| BUY | retest1 | 2024-01-03 00:00:00 | 47.30 | 2024-02-12 00:00:00 | 55.30 | TARGET_HIT | 0.50 | 16.91% |
| BUY | retest1 | 2024-03-05 00:00:00 | 63.60 | 2024-03-12 00:00:00 | 59.43 | STOP_HIT | 1.00 | -6.56% |
| BUY | retest1 | 2024-03-28 00:00:00 | 62.35 | 2024-04-15 00:00:00 | 61.65 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest1 | 2024-04-26 00:00:00 | 67.65 | 2024-05-09 00:00:00 | 63.88 | STOP_HIT | 1.00 | -5.57% |
