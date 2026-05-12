# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 121.59
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** -0.58% / -3.33%
- **Sum % (uncompounded):** -3.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.58% | -3.5% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.58% | -3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.58% | -3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 00:00:00 | 94.49 | 75.13 | 86.14 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=4.58 |
| Stop hit — per-position SL triggered | 2024-08-02 00:00:00 | 87.62 | 75.57 | 87.11 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-10-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 00:00:00 | 89.63 | 78.87 | 85.21 | Stage2 pullback-breakout RSI=60 vol=3.0x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-10-04 00:00:00 | 84.85 | 78.99 | 85.19 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 00:00:00 | 91.83 | 79.39 | 86.10 | Stage2 pullback-breakout RSI=62 vol=2.1x ATR=3.66 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 86.34 | 80.19 | 88.60 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-10-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 00:00:00 | 95.48 | 80.67 | 89.28 | Stage2 pullback-breakout RSI=60 vol=5.1x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 00:00:00 | 105.13 | 81.12 | 91.82 | T1 booked 50% @ 105.13 |
| Target hit | 2024-11-28 00:00:00 | 103.44 | 85.28 | 103.57 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 00:00:00 | 111.08 | 85.53 | 104.29 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=5.31 |
| Stop hit — per-position SL triggered | 2024-12-13 00:00:00 | 107.38 | 87.77 | 107.25 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-12-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 00:00:00 | 113.13 | 88.26 | 108.20 | Stage2 pullback-breakout RSI=61 vol=2.5x ATR=4.94 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-30 00:00:00 | 94.49 | 2024-08-02 00:00:00 | 87.62 | STOP_HIT | 1.00 | -7.27% |
| BUY | retest1 | 2024-10-01 00:00:00 | 89.63 | 2024-10-04 00:00:00 | 84.85 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest1 | 2024-10-11 00:00:00 | 91.83 | 2024-10-22 00:00:00 | 86.34 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest1 | 2024-10-29 00:00:00 | 95.48 | 2024-10-31 00:00:00 | 105.13 | PARTIAL | 0.50 | 10.10% |
| BUY | retest1 | 2024-10-29 00:00:00 | 95.48 | 2024-11-28 00:00:00 | 103.44 | TARGET_HIT | 0.50 | 8.34% |
| BUY | retest1 | 2024-11-29 00:00:00 | 111.08 | 2024-12-13 00:00:00 | 107.38 | STOP_HIT | 1.00 | -3.33% |
