# Nava Ltd. (NAVA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 727.05
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** -0.13% / -3.68%
- **Sum % (uncompounded):** -1.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.13% | -1.2% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.13% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.13% | -1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 212.50 | 142.09 | 194.46 | Stage2 pullback-breakout RSI=68 vol=3.0x ATR=8.03 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 200.45 | 147.08 | 201.08 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-09-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 00:00:00 | 225.00 | 150.65 | 204.81 | Stage2 pullback-breakout RSI=68 vol=8.5x ATR=9.37 |
| Stop hit — per-position SL triggered | 2023-10-06 00:00:00 | 216.73 | 157.25 | 214.06 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 00:00:00 | 227.45 | 160.74 | 215.80 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=8.79 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 214.26 | 163.71 | 218.00 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2023-12-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 00:00:00 | 203.13 | 172.65 | 196.93 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 00:00:00 | 216.21 | 173.74 | 200.33 | T1 booked 50% @ 216.21 |
| Target hit | 2024-01-23 00:00:00 | 226.75 | 187.94 | 229.41 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-02-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 00:00:00 | 244.35 | 190.68 | 232.29 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=8.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 00:00:00 | 261.91 | 191.89 | 235.86 | T1 booked 50% @ 261.91 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 244.35 | 194.17 | 240.50 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 256.73 | 210.80 | 247.34 | Stage2 pullback-breakout RSI=56 vol=1.7x ATR=11.19 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 239.95 | 213.62 | 249.67 | SL hit (bars_held=7) |

### Cycle 7 — BUY (started 2024-05-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 00:00:00 | 259.83 | 217.62 | 250.20 | Stage2 pullback-breakout RSI=62 vol=2.9x ATR=8.19 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 247.55 | 218.31 | 250.62 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-31 00:00:00 | 212.50 | 2023-09-12 00:00:00 | 200.45 | STOP_HIT | 1.00 | -5.67% |
| BUY | retest1 | 2023-09-21 00:00:00 | 225.00 | 2023-10-06 00:00:00 | 216.73 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest1 | 2023-10-16 00:00:00 | 227.45 | 2023-10-23 00:00:00 | 214.26 | STOP_HIT | 1.00 | -5.80% |
| BUY | retest1 | 2023-12-06 00:00:00 | 203.13 | 2023-12-11 00:00:00 | 216.21 | PARTIAL | 0.50 | 6.44% |
| BUY | retest1 | 2023-12-06 00:00:00 | 203.13 | 2024-01-23 00:00:00 | 226.75 | TARGET_HIT | 0.50 | 11.63% |
| BUY | retest1 | 2024-02-01 00:00:00 | 244.35 | 2024-02-05 00:00:00 | 261.91 | PARTIAL | 0.50 | 7.19% |
| BUY | retest1 | 2024-02-01 00:00:00 | 244.35 | 2024-02-09 00:00:00 | 244.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-03 00:00:00 | 256.73 | 2024-04-15 00:00:00 | 239.95 | STOP_HIT | 1.00 | -6.54% |
| BUY | retest1 | 2024-05-03 00:00:00 | 259.83 | 2024-05-07 00:00:00 | 247.55 | STOP_HIT | 1.00 | -4.73% |
