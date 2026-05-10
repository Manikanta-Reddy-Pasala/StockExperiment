# Suzlon Energy Ltd. (SUZLON)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 54.90
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 4
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 1.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.05% | 0.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.05% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.08% | 0.9% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.08% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.07% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 47.06 | 47.30 | 0.00 | ORB-short ORB[47.18,47.74] vol=1.5x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 46.89 | 47.24 | 0.00 | T1 1.5R @ 46.89 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 47.06 | 47.22 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 46.94 | 47.12 | 0.00 | ORB-short ORB[46.95,47.55] vol=1.5x ATR=0.09 |
| Stop hit — per-position SL triggered | 2026-02-12 12:10:00 | 47.03 | 47.09 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 46.05 | 46.33 | 0.00 | ORB-short ORB[46.09,46.74] vol=1.6x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 14:30:00 | 45.87 | 46.14 | 0.00 | T1 1.5R @ 45.87 |
| Target hit | 2026-02-13 15:20:00 | 45.67 | 46.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 43.59 | 43.91 | 0.00 | ORB-short ORB[43.82,44.33] vol=1.8x ATR=0.14 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 43.73 | 43.83 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 43.48 | 44.11 | 0.00 | ORB-short ORB[44.24,44.74] vol=2.3x ATR=0.20 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 43.68 | 44.02 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 43.78 | 43.41 | 0.00 | ORB-long ORB[43.08,43.54] vol=1.6x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-26 09:50:00 | 43.59 | 43.46 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:00:00 | 42.49 | 42.85 | 0.00 | ORB-short ORB[42.91,43.30] vol=1.6x ATR=0.14 |
| Stop hit — per-position SL triggered | 2026-02-27 10:10:00 | 42.63 | 42.82 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 39.97 | 40.27 | 0.00 | ORB-short ORB[40.15,40.64] vol=1.9x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:05:00 | 39.68 | 40.06 | 0.00 | T1 1.5R @ 39.68 |
| Target hit | 2026-03-05 14:40:00 | 39.87 | 39.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-03-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:55:00 | 42.38 | 41.58 | 0.00 | ORB-long ORB[40.93,41.54] vol=2.4x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:00:00 | 42.67 | 41.69 | 0.00 | T1 1.5R @ 42.67 |
| Target hit | 2026-03-12 14:45:00 | 42.51 | 42.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-03-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:55:00 | 41.98 | 41.58 | 0.00 | ORB-long ORB[41.24,41.79] vol=1.7x ATR=0.14 |
| Stop hit — per-position SL triggered | 2026-03-18 11:35:00 | 41.84 | 41.64 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 56.94 | 57.40 | 0.00 | ORB-short ORB[57.37,58.06] vol=1.5x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 57.16 | 57.34 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:35:00 | 47.06 | 2026-02-11 09:40:00 | 46.89 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-11 09:35:00 | 47.06 | 2026-02-11 09:45:00 | 47.06 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 11:15:00 | 46.94 | 2026-02-12 12:10:00 | 47.03 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-13 11:00:00 | 46.05 | 2026-02-13 14:30:00 | 45.87 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-13 11:00:00 | 46.05 | 2026-02-13 15:20:00 | 45.67 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2026-02-24 09:30:00 | 43.59 | 2026-02-24 09:45:00 | 43.73 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-25 09:40:00 | 43.48 | 2026-02-25 09:45:00 | 43.68 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-02-26 09:40:00 | 43.78 | 2026-02-26 09:50:00 | 43.59 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-27 10:00:00 | 42.49 | 2026-02-27 10:10:00 | 42.63 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-05 09:30:00 | 39.97 | 2026-03-05 10:05:00 | 39.68 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-03-05 09:30:00 | 39.97 | 2026-03-05 14:40:00 | 39.87 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2026-03-12 10:55:00 | 42.38 | 2026-03-12 11:00:00 | 42.67 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-12 10:55:00 | 42.38 | 2026-03-12 14:45:00 | 42.51 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2026-03-18 10:55:00 | 41.98 | 2026-03-18 11:35:00 | 41.84 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-29 09:55:00 | 56.94 | 2026-04-29 10:15:00 | 57.16 | STOP_HIT | 1.00 | -0.39% |
