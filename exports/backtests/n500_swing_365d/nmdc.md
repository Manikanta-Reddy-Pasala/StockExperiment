# NMDC Ltd. (NMDC)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 88.80
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 3
- **Avg / median % per leg:** -0.60% / -3.05%
- **Sum % (uncompounded):** -7.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | -0.60% | -7.2% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | -0.60% | -7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 4 | 33.3% | 1 | 8 | 3 | -0.60% | -7.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 05:30:00 | 72.58 | 69.72 | 71.17 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-08-14 05:30:00 | 70.22 | 69.72 | 71.00 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2025-09-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 05:30:00 | 72.80 | 69.78 | 70.58 | Stage2 pullback-breakout RSI=59 vol=2.6x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 05:30:00 | 76.08 | 70.11 | 72.69 | T1 booked 50% @ 76.08 |
| Target hit | 2025-09-26 05:30:00 | 74.99 | 70.77 | 75.16 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-10-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 05:30:00 | 78.80 | 71.21 | 75.97 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-10-14 05:30:00 | 76.25 | 71.37 | 76.18 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2025-10-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 05:30:00 | 76.70 | 71.74 | 75.55 | Stage2 pullback-breakout RSI=56 vol=2.2x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-11-04 05:30:00 | 74.22 | 71.88 | 75.50 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2025-11-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 05:30:00 | 77.18 | 72.04 | 75.41 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-11-20 05:30:00 | 74.65 | 72.27 | 75.60 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2025-12-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 05:30:00 | 76.09 | 72.41 | 74.89 | Stage2 pullback-breakout RSI=56 vol=1.5x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-12-09 05:30:00 | 73.77 | 72.57 | 75.14 | SL hit (bars_held=5) |

### Cycle 7 — BUY (started 2025-12-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 05:30:00 | 81.53 | 73.02 | 76.77 | Stage2 pullback-breakout RSI=69 vol=4.4x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 05:30:00 | 85.04 | 73.79 | 80.29 | T1 booked 50% @ 85.04 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 81.53 | 74.09 | 81.20 | SL hit (bars_held=11) |

### Cycle 8 — BUY (started 2026-01-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 05:30:00 | 84.60 | 74.89 | 80.77 | Stage2 pullback-breakout RSI=60 vol=2.4x ATR=2.56 |
| Stop hit — per-position SL triggered | 2026-01-30 05:30:00 | 80.76 | 74.96 | 80.81 | SL hit (bars_held=1) |

### Cycle 9 — BUY (started 2026-02-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 05:30:00 | 85.88 | 75.25 | 81.39 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=3.09 |
| Stop hit — per-position SL triggered | 2026-02-13 05:30:00 | 81.25 | 75.84 | 82.59 | SL hit (bars_held=7) |

### Cycle 10 — BUY (started 2026-04-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 05:30:00 | 81.40 | 76.73 | 78.72 | Stage2 pullback-breakout RSI=56 vol=2.1x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 05:30:00 | 87.09 | 77.19 | 81.47 | T1 booked 50% @ 87.09 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-13 05:30:00 | 72.58 | 2025-08-14 05:30:00 | 70.22 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest1 | 2025-09-02 05:30:00 | 72.80 | 2025-09-11 05:30:00 | 76.08 | PARTIAL | 0.50 | 4.51% |
| BUY | retest1 | 2025-09-02 05:30:00 | 72.80 | 2025-09-26 05:30:00 | 74.99 | TARGET_HIT | 0.50 | 3.01% |
| BUY | retest1 | 2025-10-09 05:30:00 | 78.80 | 2025-10-14 05:30:00 | 76.25 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest1 | 2025-10-29 05:30:00 | 76.70 | 2025-11-04 05:30:00 | 74.22 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest1 | 2025-11-12 05:30:00 | 77.18 | 2025-11-20 05:30:00 | 74.65 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest1 | 2025-12-02 05:30:00 | 76.09 | 2025-12-09 05:30:00 | 73.77 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest1 | 2025-12-23 05:30:00 | 81.53 | 2026-01-05 05:30:00 | 85.04 | PARTIAL | 0.50 | 4.30% |
| BUY | retest1 | 2025-12-23 05:30:00 | 81.53 | 2026-01-08 05:30:00 | 81.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-29 05:30:00 | 84.60 | 2026-01-30 05:30:00 | 80.76 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest1 | 2026-02-04 05:30:00 | 85.88 | 2026-02-13 05:30:00 | 81.25 | STOP_HIT | 1.00 | -5.39% |
| BUY | retest1 | 2026-04-06 05:30:00 | 81.40 | 2026-04-15 05:30:00 | 87.09 | PARTIAL | 0.50 | 6.99% |
