# WIPRO (WIPRO)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 197.91
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.75% / 1.52%
- **Sum % (uncompounded):** 5.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.75% | 5.3% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.75% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.75% | 5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 05:30:00 | 267.30 | 241.05 | 257.37 | Stage2 pullback-breakout RSI=60 vol=2.7x ATR=6.20 |
| Stop hit — per-position SL triggered | 2024-09-09 05:30:00 | 258.00 | 242.81 | 260.55 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-09-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 05:30:00 | 275.30 | 243.69 | 262.24 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=6.45 |
| Stop hit — per-position SL triggered | 2024-09-19 05:30:00 | 265.63 | 244.80 | 265.29 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-10-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 05:30:00 | 274.77 | 248.23 | 267.14 | Stage2 pullback-breakout RSI=59 vol=2.5x ATR=6.95 |
| Stop hit — per-position SL triggered | 2024-10-17 05:30:00 | 264.34 | 248.74 | 266.74 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 05:30:00 | 291.23 | 255.52 | 280.41 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=7.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 05:30:00 | 306.00 | 259.67 | 291.41 | T1 booked 50% @ 306.00 |
| Target hit | 2024-12-31 05:30:00 | 301.85 | 265.95 | 303.13 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2025-01-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 05:30:00 | 300.25 | 269.68 | 295.93 | Stage2 pullback-breakout RSI=53 vol=3.7x ATR=8.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 05:30:00 | 317.70 | 270.83 | 299.33 | T1 booked 50% @ 317.70 |
| Stop hit — per-position SL triggered | 2025-02-01 05:30:00 | 304.80 | 273.47 | 304.47 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-28 05:30:00 | 267.30 | 2024-09-09 05:30:00 | 258.00 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest1 | 2024-09-13 05:30:00 | 275.30 | 2024-09-19 05:30:00 | 265.63 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2024-10-14 05:30:00 | 274.77 | 2024-10-17 05:30:00 | 264.34 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest1 | 2024-11-25 05:30:00 | 291.23 | 2024-12-10 05:30:00 | 306.00 | PARTIAL | 0.50 | 5.07% |
| BUY | retest1 | 2024-11-25 05:30:00 | 291.23 | 2024-12-31 05:30:00 | 301.85 | TARGET_HIT | 0.50 | 3.65% |
| BUY | retest1 | 2025-01-20 05:30:00 | 300.25 | 2025-01-23 05:30:00 | 317.70 | PARTIAL | 0.50 | 5.81% |
| BUY | retest1 | 2025-01-20 05:30:00 | 300.25 | 2025-02-01 05:30:00 | 304.80 | STOP_HIT | 0.50 | 1.52% |
