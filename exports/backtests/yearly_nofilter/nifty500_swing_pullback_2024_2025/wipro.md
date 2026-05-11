# Wipro Ltd. (WIPRO)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -1.43% / -3.48%
- **Sum % (uncompounded):** -5.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.43% | -5.7% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.43% | -5.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.43% | -5.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 00:00:00 | 267.30 | 241.05 | 257.37 | Stage2 pullback-breakout RSI=60 vol=2.7x ATR=6.20 |
| Stop hit — per-position SL triggered | 2024-09-09 00:00:00 | 258.00 | 242.81 | 260.55 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 00:00:00 | 275.30 | 243.69 | 262.24 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=6.45 |
| Stop hit — per-position SL triggered | 2024-09-19 00:00:00 | 265.63 | 244.80 | 265.29 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2024-10-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 00:00:00 | 274.77 | 248.23 | 267.14 | Stage2 pullback-breakout RSI=59 vol=2.5x ATR=6.95 |
| Stop hit — per-position SL triggered | 2024-10-17 00:00:00 | 264.34 | 248.74 | 266.74 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 291.23 | 255.52 | 280.41 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=7.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 00:00:00 | 306.00 | 259.67 | 291.41 | T1 booked 50% @ 306.00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-28 00:00:00 | 267.30 | 2024-09-09 00:00:00 | 258.00 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest1 | 2024-09-13 00:00:00 | 275.30 | 2024-09-19 00:00:00 | 265.63 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2024-10-14 00:00:00 | 274.77 | 2024-10-17 00:00:00 | 264.34 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest1 | 2024-11-25 00:00:00 | 291.23 | 2024-12-10 00:00:00 | 306.00 | PARTIAL | 0.50 | 5.07% |
