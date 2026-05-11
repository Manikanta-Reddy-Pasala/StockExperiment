# Indian Overseas Bank (IOB)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 34.82
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
| TARGET_HIT | 3 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 3 / 3 / 3
- **Avg / median % per leg:** 4.18% / 4.41%
- **Sum % (uncompounded):** 37.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 3 | 3 | 3 | 4.18% | 37.6% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 3 | 3 | 3 | 4.18% | 37.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 3 | 3 | 3 | 4.18% | 37.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 24.95 | 23.71 | 24.40 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 26.08 | 23.73 | 24.53 | T1 booked 50% @ 26.08 |
| Target hit | 2023-08-03 00:00:00 | 26.05 | 24.25 | 26.15 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 41.10 | 32.41 | 40.05 | Stage2 pullback-breakout RSI=57 vol=2.1x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 00:00:00 | 43.61 | 32.72 | 40.83 | T1 booked 50% @ 43.61 |
| Target hit | 2023-12-20 00:00:00 | 42.30 | 33.67 | 42.65 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 48.45 | 36.05 | 44.56 | Stage2 pullback-breakout RSI=69 vol=2.9x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 00:00:00 | 52.34 | 36.46 | 45.95 | T1 booked 50% @ 52.34 |
| Target hit | 2024-02-28 00:00:00 | 63.20 | 41.84 | 64.08 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 69.25 | 42.98 | 64.59 | Stage2 pullback-breakout RSI=62 vol=1.5x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-03-11 00:00:00 | 63.63 | 43.64 | 64.72 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 64.75 | 45.55 | 60.95 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=3.68 |
| Stop hit — per-position SL triggered | 2024-04-16 00:00:00 | 62.00 | 47.38 | 63.14 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 66.90 | 48.31 | 63.34 | Stage2 pullback-breakout RSI=60 vol=2.5x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 62.58 | 49.52 | 64.65 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 24.95 | 2023-07-04 00:00:00 | 26.08 | PARTIAL | 0.50 | 4.53% |
| BUY | retest1 | 2023-07-03 00:00:00 | 24.95 | 2023-08-03 00:00:00 | 26.05 | TARGET_HIT | 0.50 | 4.41% |
| BUY | retest1 | 2023-12-04 00:00:00 | 41.10 | 2023-12-07 00:00:00 | 43.61 | PARTIAL | 0.50 | 6.10% |
| BUY | retest1 | 2023-12-04 00:00:00 | 41.10 | 2023-12-20 00:00:00 | 42.30 | TARGET_HIT | 0.50 | 2.92% |
| BUY | retest1 | 2024-01-29 00:00:00 | 48.45 | 2024-02-01 00:00:00 | 52.34 | PARTIAL | 0.50 | 8.02% |
| BUY | retest1 | 2024-01-29 00:00:00 | 48.45 | 2024-02-28 00:00:00 | 63.20 | TARGET_HIT | 0.50 | 30.44% |
| BUY | retest1 | 2024-03-05 00:00:00 | 69.25 | 2024-03-11 00:00:00 | 63.63 | STOP_HIT | 1.00 | -8.11% |
| BUY | retest1 | 2024-04-01 00:00:00 | 64.75 | 2024-04-16 00:00:00 | 62.00 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest1 | 2024-04-25 00:00:00 | 66.90 | 2024-05-07 00:00:00 | 62.58 | STOP_HIT | 1.00 | -6.46% |
