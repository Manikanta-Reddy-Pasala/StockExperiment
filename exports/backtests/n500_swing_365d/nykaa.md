# FSN E-Commerce Ventures Ltd. (NYKAA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 272.80
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 2
- **Target hits / Stop hits / Partials:** 1 / 7 / 3
- **Avg / median % per leg:** 2.87% / 2.82%
- **Sum % (uncompounded):** 31.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 9 | 81.8% | 1 | 7 | 3 | 2.87% | 31.5% |
| BUY @ 2nd Alert (retest1) | 11 | 9 | 81.8% | 1 | 7 | 3 | 2.87% | 31.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 9 | 81.8% | 1 | 7 | 3 | 2.87% | 31.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 05:30:00 | 209.86 | 188.02 | 199.97 | Stage2 pullback-breakout RSI=64 vol=2.6x ATR=5.74 |
| Stop hit — per-position SL triggered | 2025-07-03 05:30:00 | 201.24 | 188.77 | 202.34 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-07-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 05:30:00 | 213.37 | 189.37 | 203.06 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=6.16 |
| Stop hit — per-position SL triggered | 2025-07-23 05:30:00 | 218.71 | 192.00 | 211.92 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-08-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 05:30:00 | 215.04 | 194.49 | 209.80 | Stage2 pullback-breakout RSI=56 vol=8.6x ATR=6.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 05:30:00 | 227.32 | 195.25 | 212.65 | T1 booked 50% @ 227.32 |
| Target hit | 2025-09-24 05:30:00 | 237.05 | 204.56 | 237.44 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-10-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 05:30:00 | 255.34 | 206.84 | 238.46 | Stage2 pullback-breakout RSI=69 vol=5.5x ATR=6.80 |
| Stop hit — per-position SL triggered | 2025-10-20 05:30:00 | 257.48 | 212.00 | 252.58 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-11-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 05:30:00 | 260.82 | 216.99 | 252.58 | Stage2 pullback-breakout RSI=59 vol=6.1x ATR=7.70 |
| Stop hit — per-position SL triggered | 2025-11-26 05:30:00 | 264.45 | 222.49 | 262.37 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2025-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 05:30:00 | 263.55 | 229.18 | 255.75 | Stage2 pullback-breakout RSI=61 vol=4.3x ATR=6.41 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 253.94 | 231.84 | 259.80 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2026-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 05:30:00 | 258.29 | 233.95 | 246.99 | Stage2 pullback-breakout RSI=62 vol=2.7x ATR=7.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 05:30:00 | 273.65 | 234.38 | 249.85 | T1 booked 50% @ 273.65 |
| Stop hit — per-position SL triggered | 2026-02-20 05:30:00 | 265.58 | 238.13 | 264.27 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2026-04-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 05:30:00 | 252.52 | 240.63 | 245.87 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=8.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 05:30:00 | 269.94 | 242.01 | 253.53 | T1 booked 50% @ 269.94 |
| Stop hit — per-position SL triggered | 2026-04-22 05:30:00 | 259.94 | 242.63 | 255.98 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-27 05:30:00 | 209.86 | 2025-07-03 05:30:00 | 201.24 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest1 | 2025-07-09 05:30:00 | 213.37 | 2025-07-23 05:30:00 | 218.71 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest1 | 2025-08-13 05:30:00 | 215.04 | 2025-08-19 05:30:00 | 227.32 | PARTIAL | 0.50 | 5.71% |
| BUY | retest1 | 2025-08-13 05:30:00 | 215.04 | 2025-09-24 05:30:00 | 237.05 | TARGET_HIT | 0.50 | 10.24% |
| BUY | retest1 | 2025-10-06 05:30:00 | 255.34 | 2025-10-20 05:30:00 | 257.48 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest1 | 2025-11-10 05:30:00 | 260.82 | 2025-11-26 05:30:00 | 264.45 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest1 | 2025-12-30 05:30:00 | 263.55 | 2026-01-09 05:30:00 | 253.94 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest1 | 2026-02-05 05:30:00 | 258.29 | 2026-02-06 05:30:00 | 273.65 | PARTIAL | 0.50 | 5.95% |
| BUY | retest1 | 2026-02-05 05:30:00 | 258.29 | 2026-02-20 05:30:00 | 265.58 | STOP_HIT | 0.50 | 2.82% |
| BUY | retest1 | 2026-04-06 05:30:00 | 252.52 | 2026-04-17 05:30:00 | 269.94 | PARTIAL | 0.50 | 6.90% |
| BUY | retest1 | 2026-04-06 05:30:00 | 252.52 | 2026-04-22 05:30:00 | 259.94 | STOP_HIT | 0.50 | 2.94% |
