# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 214.49
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
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 5
- **Avg / median % per leg:** 1.93% / 3.49%
- **Sum % (uncompounded):** 21.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 1 | 5 | 5 | 1.93% | 21.3% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 1 | 5 | 5 | 1.93% | 21.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 1 | 5 | 5 | 1.93% | 21.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 00:00:00 | 115.30 | 109.01 | 111.91 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 00:00:00 | 119.32 | 109.35 | 113.55 | T1 booked 50% @ 119.32 |
| Stop hit — per-position SL triggered | 2023-07-24 00:00:00 | 115.30 | 109.69 | 114.68 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2023-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 00:00:00 | 119.25 | 109.79 | 115.11 | Stage2 pullback-breakout RSI=68 vol=2.5x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 00:00:00 | 123.72 | 110.22 | 117.06 | T1 booked 50% @ 123.72 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 119.25 | 110.44 | 117.77 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2023-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 00:00:00 | 122.10 | 111.81 | 118.35 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 126.38 | 112.07 | 119.57 | T1 booked 50% @ 126.38 |
| Target hit | 2023-09-22 00:00:00 | 126.75 | 114.40 | 127.10 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 00:00:00 | 137.25 | 122.29 | 134.57 | Stage2 pullback-breakout RSI=60 vol=1.5x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 132.85 | 122.38 | 134.29 | SL hit (bars_held=1) |

### Cycle 5 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 138.70 | 123.63 | 134.64 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 00:00:00 | 145.45 | 124.22 | 136.97 | T1 booked 50% @ 145.45 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 138.70 | 124.70 | 137.91 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2024-02-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 00:00:00 | 143.90 | 125.83 | 139.77 | Stage2 pullback-breakout RSI=61 vol=2.0x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-02 00:00:00 | 151.15 | 127.36 | 143.31 | T1 booked 50% @ 151.15 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 143.90 | 129.02 | 147.44 | SL hit (bars_held=15) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-10 00:00:00 | 115.30 | 2023-07-17 00:00:00 | 119.32 | PARTIAL | 0.50 | 3.49% |
| BUY | retest1 | 2023-07-10 00:00:00 | 115.30 | 2023-07-24 00:00:00 | 115.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-25 00:00:00 | 119.25 | 2023-07-31 00:00:00 | 123.72 | PARTIAL | 0.50 | 3.75% |
| BUY | retest1 | 2023-07-25 00:00:00 | 119.25 | 2023-08-02 00:00:00 | 119.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-30 00:00:00 | 122.10 | 2023-09-01 00:00:00 | 126.38 | PARTIAL | 0.50 | 3.51% |
| BUY | retest1 | 2023-08-30 00:00:00 | 122.10 | 2023-09-22 00:00:00 | 126.75 | TARGET_HIT | 0.50 | 3.81% |
| BUY | retest1 | 2024-01-16 00:00:00 | 137.25 | 2024-01-17 00:00:00 | 132.85 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest1 | 2024-02-02 00:00:00 | 138.70 | 2024-02-07 00:00:00 | 145.45 | PARTIAL | 0.50 | 4.86% |
| BUY | retest1 | 2024-02-02 00:00:00 | 138.70 | 2024-02-12 00:00:00 | 138.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-21 00:00:00 | 143.90 | 2024-03-02 00:00:00 | 151.15 | PARTIAL | 0.50 | 5.04% |
| BUY | retest1 | 2024-02-21 00:00:00 | 143.90 | 2024-03-13 00:00:00 | 143.90 | STOP_HIT | 0.50 | 0.00% |
