# RITES Ltd. (RITES)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 227.15
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
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 2.83% / 2.35%
- **Sum % (uncompounded):** 28.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 1 | 6 | 3 | 2.83% | 28.3% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 1 | 6 | 3 | 2.83% | 28.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 6 | 60.0% | 1 | 6 | 3 | 2.83% | 28.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 05:30:00 | 197.58 | 178.38 | 189.59 | Stage2 pullback-breakout RSI=61 vol=2.6x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 05:30:00 | 208.35 | 179.20 | 192.94 | T1 booked 50% @ 208.35 |
| Target hit | 2023-08-18 05:30:00 | 226.35 | 189.54 | 227.29 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 05:30:00 | 252.48 | 193.29 | 233.84 | Stage2 pullback-breakout RSI=68 vol=4.7x ATR=9.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 05:30:00 | 271.89 | 197.62 | 246.08 | T1 booked 50% @ 271.89 |
| Stop hit — per-position SL triggered | 2023-09-13 05:30:00 | 252.48 | 199.68 | 251.45 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2023-10-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 05:30:00 | 249.23 | 208.04 | 244.38 | Stage2 pullback-breakout RSI=54 vol=6.1x ATR=8.99 |
| Stop hit — per-position SL triggered | 2023-10-23 05:30:00 | 235.75 | 210.33 | 245.32 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2023-11-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 05:30:00 | 238.63 | 212.87 | 229.23 | Stage2 pullback-breakout RSI=58 vol=4.7x ATR=7.99 |
| Stop hit — per-position SL triggered | 2023-12-04 05:30:00 | 239.80 | 215.02 | 233.23 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2023-12-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 05:30:00 | 249.15 | 216.30 | 236.52 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=7.36 |
| Stop hit — per-position SL triggered | 2023-12-21 05:30:00 | 238.11 | 219.09 | 244.92 | SL hit (bars_held=8) |

### Cycle 6 — BUY (started 2024-01-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 05:30:00 | 259.43 | 221.58 | 248.82 | Stage2 pullback-breakout RSI=63 vol=3.1x ATR=8.75 |
| Stop hit — per-position SL triggered | 2024-01-17 05:30:00 | 265.52 | 225.30 | 256.85 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-01-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 05:30:00 | 276.73 | 226.15 | 259.03 | Stage2 pullback-breakout RSI=66 vol=4.2x ATR=10.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 05:30:00 | 297.80 | 227.03 | 264.31 | T1 booked 50% @ 297.80 |
| Stop hit — per-position SL triggered | 2024-01-24 05:30:00 | 276.73 | 228.19 | 268.18 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 05:30:00 | 197.58 | 2023-07-18 05:30:00 | 208.35 | PARTIAL | 0.50 | 5.45% |
| BUY | retest1 | 2023-07-12 05:30:00 | 197.58 | 2023-08-18 05:30:00 | 226.35 | TARGET_HIT | 0.50 | 14.56% |
| BUY | retest1 | 2023-08-30 05:30:00 | 252.48 | 2023-09-08 05:30:00 | 271.89 | PARTIAL | 0.50 | 7.69% |
| BUY | retest1 | 2023-08-30 05:30:00 | 252.48 | 2023-09-13 05:30:00 | 252.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-13 05:30:00 | 249.23 | 2023-10-23 05:30:00 | 235.75 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest1 | 2023-11-17 05:30:00 | 238.63 | 2023-12-04 05:30:00 | 239.80 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest1 | 2023-12-11 05:30:00 | 249.15 | 2023-12-21 05:30:00 | 238.11 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2024-01-03 05:30:00 | 259.43 | 2024-01-17 05:30:00 | 265.52 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest1 | 2024-01-19 05:30:00 | 276.73 | 2024-01-20 05:30:00 | 297.80 | PARTIAL | 0.50 | 7.62% |
| BUY | retest1 | 2024-01-19 05:30:00 | 276.73 | 2024-01-24 05:30:00 | 276.73 | STOP_HIT | 0.50 | 0.00% |
