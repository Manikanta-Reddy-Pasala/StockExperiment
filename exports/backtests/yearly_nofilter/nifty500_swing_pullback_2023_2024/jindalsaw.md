# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 242.81
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 5
- **Target hits / Stop hits / Partials:** 2 / 6 / 4
- **Avg / median % per leg:** 3.19% / 7.45%
- **Sum % (uncompounded):** 38.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 2 | 6 | 4 | 3.19% | 38.3% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 2 | 6 | 4 | 3.19% | 38.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 7 | 58.3% | 2 | 6 | 4 | 3.19% | 38.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 00:00:00 | 166.30 | 106.24 | 161.65 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=8.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 183.00 | 107.66 | 164.59 | T1 booked 50% @ 183.00 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 166.30 | 112.73 | 173.50 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-10-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 00:00:00 | 188.95 | 125.30 | 176.49 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=8.21 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 176.64 | 129.09 | 182.38 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2023-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 00:00:00 | 206.60 | 131.11 | 186.38 | Stage2 pullback-breakout RSI=65 vol=2.1x ATR=12.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 00:00:00 | 230.77 | 134.69 | 198.42 | T1 booked 50% @ 230.77 |
| Target hit | 2023-11-22 00:00:00 | 227.73 | 147.63 | 229.01 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 00:00:00 | 221.30 | 164.50 | 215.51 | Stage2 pullback-breakout RSI=54 vol=2.6x ATR=9.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 00:00:00 | 239.98 | 167.68 | 221.34 | T1 booked 50% @ 239.98 |
| Target hit | 2024-01-23 00:00:00 | 237.78 | 175.54 | 239.41 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-02-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 00:00:00 | 268.88 | 183.92 | 250.85 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=10.57 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 253.03 | 185.37 | 251.90 | SL hit (bars_held=2) |

### Cycle 6 — BUY (started 2024-02-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 00:00:00 | 268.63 | 189.45 | 254.06 | Stage2 pullback-breakout RSI=60 vol=1.6x ATR=13.11 |
| Stop hit — per-position SL triggered | 2024-02-28 00:00:00 | 248.97 | 193.32 | 254.52 | SL hit (bars_held=6) |

### Cycle 7 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 236.48 | 198.78 | 223.23 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=12.69 |
| Stop hit — per-position SL triggered | 2024-04-16 00:00:00 | 239.83 | 203.42 | 238.50 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2024-04-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 00:00:00 | 266.90 | 205.63 | 242.98 | Stage2 pullback-breakout RSI=66 vol=3.2x ATR=11.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 00:00:00 | 290.21 | 210.77 | 262.94 | T1 booked 50% @ 290.21 |
| Stop hit — per-position SL triggered | 2024-05-08 00:00:00 | 266.90 | 212.04 | 265.05 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-30 00:00:00 | 166.30 | 2023-09-01 00:00:00 | 183.00 | PARTIAL | 0.50 | 10.04% |
| BUY | retest1 | 2023-08-30 00:00:00 | 166.30 | 2023-09-12 00:00:00 | 166.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-16 00:00:00 | 188.95 | 2023-10-25 00:00:00 | 176.64 | STOP_HIT | 1.00 | -6.52% |
| BUY | retest1 | 2023-10-30 00:00:00 | 206.60 | 2023-11-03 00:00:00 | 230.77 | PARTIAL | 0.50 | 11.70% |
| BUY | retest1 | 2023-10-30 00:00:00 | 206.60 | 2023-11-22 00:00:00 | 227.73 | TARGET_HIT | 0.50 | 10.23% |
| BUY | retest1 | 2024-01-02 00:00:00 | 221.30 | 2024-01-09 00:00:00 | 239.98 | PARTIAL | 0.50 | 8.44% |
| BUY | retest1 | 2024-01-02 00:00:00 | 221.30 | 2024-01-23 00:00:00 | 237.78 | TARGET_HIT | 0.50 | 7.45% |
| BUY | retest1 | 2024-02-08 00:00:00 | 268.88 | 2024-02-12 00:00:00 | 253.03 | STOP_HIT | 1.00 | -5.90% |
| BUY | retest1 | 2024-02-20 00:00:00 | 268.63 | 2024-02-28 00:00:00 | 248.97 | STOP_HIT | 1.00 | -7.32% |
| BUY | retest1 | 2024-04-01 00:00:00 | 236.48 | 2024-04-16 00:00:00 | 239.83 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest1 | 2024-04-24 00:00:00 | 266.90 | 2024-05-06 00:00:00 | 290.21 | PARTIAL | 0.50 | 8.73% |
| BUY | retest1 | 2024-04-24 00:00:00 | 266.90 | 2024-05-08 00:00:00 | 266.90 | STOP_HIT | 0.50 | 0.00% |
