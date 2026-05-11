# Central Bank of India (CENTRALBK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 35.53
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
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 1.05% / 0.00%
- **Sum % (uncompounded):** 9.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.05% | 9.5% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.05% | 9.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.05% | 9.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 29.90 | 25.96 | 28.06 | Stage2 pullback-breakout RSI=69 vol=2.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 00:00:00 | 31.51 | 26.10 | 28.78 | T1 booked 50% @ 31.51 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 29.90 | 26.32 | 29.50 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 46.85 | 35.00 | 45.03 | Stage2 pullback-breakout RSI=58 vol=1.5x ATR=1.94 |
| Stop hit — per-position SL triggered | 2023-11-28 00:00:00 | 43.94 | 35.89 | 45.09 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 48.30 | 36.26 | 45.19 | Stage2 pullback-breakout RSI=64 vol=5.5x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-18 00:00:00 | 51.49 | 37.40 | 47.23 | T1 booked 50% @ 51.49 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 48.30 | 37.64 | 47.68 | SL hit (bars_held=12) |

### Cycle 4 — BUY (started 2024-01-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 00:00:00 | 54.55 | 40.19 | 51.21 | Stage2 pullback-breakout RSI=66 vol=3.3x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 00:00:00 | 58.76 | 41.22 | 53.42 | T1 booked 50% @ 58.76 |
| Target hit | 2024-02-12 00:00:00 | 59.95 | 43.10 | 60.28 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 67.40 | 46.50 | 63.79 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-03-11 00:00:00 | 62.32 | 47.01 | 63.78 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 65.80 | 50.71 | 63.05 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=2.81 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 61.58 | 51.70 | 64.00 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 29.90 | 2023-07-06 00:00:00 | 31.51 | PARTIAL | 0.50 | 5.39% |
| BUY | retest1 | 2023-07-03 00:00:00 | 29.90 | 2023-07-13 00:00:00 | 29.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-13 00:00:00 | 46.85 | 2023-11-28 00:00:00 | 43.94 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest1 | 2023-12-04 00:00:00 | 48.30 | 2023-12-18 00:00:00 | 51.49 | PARTIAL | 0.50 | 6.60% |
| BUY | retest1 | 2023-12-04 00:00:00 | 48.30 | 2023-12-20 00:00:00 | 48.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-19 00:00:00 | 54.55 | 2024-01-31 00:00:00 | 58.76 | PARTIAL | 0.50 | 7.72% |
| BUY | retest1 | 2024-01-19 00:00:00 | 54.55 | 2024-02-12 00:00:00 | 59.95 | TARGET_HIT | 0.50 | 9.90% |
| BUY | retest1 | 2024-03-05 00:00:00 | 67.40 | 2024-03-11 00:00:00 | 62.32 | STOP_HIT | 1.00 | -7.53% |
| BUY | retest1 | 2024-04-25 00:00:00 | 65.80 | 2024-05-07 00:00:00 | 61.58 | STOP_HIT | 1.00 | -6.41% |
