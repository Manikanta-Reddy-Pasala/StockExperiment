# UCO Bank (UCOBANK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 26.33
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 1 / 5 / 2
- **Avg / median % per leg:** 5.04% / 1.07%
- **Sum % (uncompounded):** 40.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 5 | 2 | 5.04% | 40.4% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 5 | 2 | 5.04% | 40.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 5 | 2 | 5.04% | 40.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 28.10 | 23.91 | 27.04 | Stage2 pullback-breakout RSI=61 vol=3.1x ATR=0.83 |
| Stop hit — per-position SL triggered | 2023-07-17 00:00:00 | 28.40 | 24.35 | 27.90 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 00:00:00 | 29.05 | 25.03 | 28.12 | Stage2 pullback-breakout RSI=60 vol=4.0x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 00:00:00 | 30.94 | 25.18 | 28.63 | T1 booked 50% @ 30.94 |
| Target hit | 2023-10-09 00:00:00 | 40.40 | 28.93 | 40.85 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 39.60 | 31.04 | 38.18 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-11-28 00:00:00 | 37.23 | 31.67 | 38.22 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 39.65 | 31.94 | 38.30 | Stage2 pullback-breakout RSI=59 vol=2.9x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-18 00:00:00 | 42.18 | 32.73 | 39.61 | T1 booked 50% @ 42.18 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 39.65 | 32.87 | 39.66 | SL hit (bars_held=12) |

### Cycle 5 — BUY (started 2024-01-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 00:00:00 | 41.05 | 33.47 | 39.83 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 40.90 | 34.19 | 40.58 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 56.80 | 43.87 | 54.66 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 53.21 | 44.68 | 55.09 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 28.10 | 2023-07-17 00:00:00 | 28.40 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest1 | 2023-08-11 00:00:00 | 29.05 | 2023-08-17 00:00:00 | 30.94 | PARTIAL | 0.50 | 6.51% |
| BUY | retest1 | 2023-08-11 00:00:00 | 29.05 | 2023-10-09 00:00:00 | 40.40 | TARGET_HIT | 0.50 | 39.07% |
| BUY | retest1 | 2023-11-13 00:00:00 | 39.60 | 2023-11-28 00:00:00 | 37.23 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest1 | 2023-12-04 00:00:00 | 39.65 | 2023-12-18 00:00:00 | 42.18 | PARTIAL | 0.50 | 6.37% |
| BUY | retest1 | 2023-12-04 00:00:00 | 39.65 | 2023-12-20 00:00:00 | 39.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-03 00:00:00 | 41.05 | 2024-01-17 00:00:00 | 40.90 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-04-25 00:00:00 | 56.80 | 2024-05-07 00:00:00 | 53.21 | STOP_HIT | 1.00 | -6.31% |
