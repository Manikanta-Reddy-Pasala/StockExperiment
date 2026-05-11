# NLC India Ltd. (NLCINDIA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 328.20
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
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 8.18% / 0.90%
- **Sum % (uncompounded):** 73.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 1 | 5 | 3 | 8.18% | 73.7% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 5 | 3 | 8.18% | 73.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 1 | 5 | 3 | 8.18% | 73.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 136.85 | 97.68 | 127.67 | Stage2 pullback-breakout RSI=69 vol=1.7x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 00:00:00 | 147.34 | 99.00 | 131.41 | T1 booked 50% @ 147.34 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 136.85 | 100.53 | 133.44 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 139.20 | 105.23 | 132.81 | Stage2 pullback-breakout RSI=59 vol=2.4x ATR=5.53 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 132.85 | 108.11 | 134.26 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 141.65 | 109.89 | 133.27 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=6.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 00:00:00 | 154.41 | 111.62 | 138.24 | T1 booked 50% @ 154.41 |
| Target hit | 2024-01-08 00:00:00 | 220.65 | 140.26 | 220.67 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 251.00 | 152.31 | 230.83 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=14.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 00:00:00 | 279.67 | 157.71 | 243.86 | T1 booked 50% @ 279.67 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 251.00 | 161.90 | 250.39 | SL hit (bars_held=9) |

### Cycle 5 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 233.10 | 179.71 | 220.90 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=14.03 |
| Stop hit — per-position SL triggered | 2024-04-12 00:00:00 | 235.20 | 184.66 | 228.14 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 250.55 | 188.61 | 230.52 | Stage2 pullback-breakout RSI=65 vol=3.5x ATR=10.95 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 234.12 | 191.09 | 233.76 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-01 00:00:00 | 136.85 | 2023-09-06 00:00:00 | 147.34 | PARTIAL | 0.50 | 7.66% |
| BUY | retest1 | 2023-09-01 00:00:00 | 136.85 | 2023-09-12 00:00:00 | 136.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-06 00:00:00 | 139.20 | 2023-10-20 00:00:00 | 132.85 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest1 | 2023-11-02 00:00:00 | 141.65 | 2023-11-09 00:00:00 | 154.41 | PARTIAL | 0.50 | 9.01% |
| BUY | retest1 | 2023-11-02 00:00:00 | 141.65 | 2024-01-08 00:00:00 | 220.65 | TARGET_HIT | 0.50 | 55.77% |
| BUY | retest1 | 2024-01-29 00:00:00 | 251.00 | 2024-02-05 00:00:00 | 279.67 | PARTIAL | 0.50 | 11.42% |
| BUY | retest1 | 2024-01-29 00:00:00 | 251.00 | 2024-02-09 00:00:00 | 251.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-27 00:00:00 | 233.10 | 2024-04-12 00:00:00 | 235.20 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest1 | 2024-04-26 00:00:00 | 250.55 | 2024-05-06 00:00:00 | 234.12 | STOP_HIT | 1.00 | -6.56% |
