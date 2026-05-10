# Sagility Ltd. (SAGILITY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 44.44
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 6
- **Avg / median % per leg:** 0.46% / 0.42%
- **Sum % (uncompounded):** 8.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | 0.02% | 0.2% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | 0.02% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 7 | 70.0% | 3 | 3 | 4 | 0.82% | 8.2% |
| SELL @ 2nd Alert (retest1) | 10 | 7 | 70.0% | 3 | 3 | 4 | 0.82% | 8.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 9 | 50.0% | 3 | 9 | 6 | 0.46% | 8.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 49.60 | 49.36 | 0.00 | ORB-long ORB[49.06,49.44] vol=2.3x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:55:00 | 49.84 | 49.48 | 0.00 | T1 1.5R @ 49.84 |
| Stop hit — per-position SL triggered | 2026-02-11 10:05:00 | 49.60 | 49.52 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 46.04 | 45.83 | 0.00 | ORB-long ORB[45.55,46.00] vol=1.6x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-02-18 09:55:00 | 45.89 | 45.91 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 44.97 | 45.27 | 0.00 | ORB-short ORB[45.08,45.75] vol=1.7x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:05:00 | 44.70 | 45.02 | 0.00 | T1 1.5R @ 44.70 |
| Target hit | 2026-02-20 13:30:00 | 44.52 | 44.39 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2026-02-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:05:00 | 42.29 | 42.68 | 0.00 | ORB-short ORB[42.71,43.20] vol=1.6x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:15:00 | 42.05 | 42.63 | 0.00 | T1 1.5R @ 42.05 |
| Target hit | 2026-02-26 15:20:00 | 40.29 | 41.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-03-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:25:00 | 38.18 | 37.84 | 0.00 | ORB-long ORB[37.58,38.10] vol=2.3x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:35:00 | 38.46 | 37.93 | 0.00 | T1 1.5R @ 38.46 |
| Stop hit — per-position SL triggered | 2026-03-20 11:45:00 | 38.18 | 38.08 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:00:00 | 42.89 | 43.07 | 0.00 | ORB-short ORB[42.90,43.25] vol=1.7x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:05:00 | 42.71 | 43.03 | 0.00 | T1 1.5R @ 42.71 |
| Stop hit — per-position SL triggered | 2026-04-10 10:20:00 | 42.89 | 42.99 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:45:00 | 43.45 | 43.14 | 0.00 | ORB-long ORB[42.82,43.32] vol=3.8x ATR=0.13 |
| Stop hit — per-position SL triggered | 2026-04-16 10:50:00 | 43.32 | 43.18 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 11:05:00 | 42.35 | 42.57 | 0.00 | ORB-short ORB[42.43,43.00] vol=2.2x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-04-20 11:25:00 | 42.46 | 42.55 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:30:00 | 41.94 | 42.25 | 0.00 | ORB-short ORB[42.23,42.63] vol=2.0x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:10:00 | 41.74 | 42.15 | 0.00 | T1 1.5R @ 41.74 |
| Target hit | 2026-04-22 15:20:00 | 41.59 | 41.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 42.11 | 41.92 | 0.00 | ORB-long ORB[41.60,42.08] vol=1.8x ATR=0.17 |
| Stop hit — per-position SL triggered | 2026-04-27 15:20:00 | 42.09 | 42.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 42.02 | 41.78 | 0.00 | ORB-long ORB[41.40,41.84] vol=4.1x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 41.87 | 41.81 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 41.55 | 41.62 | 0.00 | ORB-short ORB[41.62,41.97] vol=1.7x ATR=0.08 |
| Stop hit — per-position SL triggered | 2026-05-06 12:00:00 | 41.63 | 41.60 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 09:35:00 | 49.60 | 2026-02-11 09:55:00 | 49.84 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-11 09:35:00 | 49.60 | 2026-02-11 10:05:00 | 49.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:30:00 | 46.04 | 2026-02-18 09:55:00 | 45.89 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-20 09:40:00 | 44.97 | 2026-02-20 10:05:00 | 44.70 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-02-20 09:40:00 | 44.97 | 2026-02-20 13:30:00 | 44.52 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2026-02-26 11:05:00 | 42.29 | 2026-02-26 11:15:00 | 42.05 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-26 11:05:00 | 42.29 | 2026-02-26 15:20:00 | 40.29 | TARGET_HIT | 0.50 | 4.73% |
| BUY | retest1 | 2026-03-20 10:25:00 | 38.18 | 2026-03-20 10:35:00 | 38.46 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-03-20 10:25:00 | 38.18 | 2026-03-20 11:45:00 | 38.18 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-10 10:00:00 | 42.89 | 2026-04-10 10:05:00 | 42.71 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-10 10:00:00 | 42.89 | 2026-04-10 10:20:00 | 42.89 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 10:45:00 | 43.45 | 2026-04-16 10:50:00 | 43.32 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-20 11:05:00 | 42.35 | 2026-04-20 11:25:00 | 42.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-22 10:30:00 | 41.94 | 2026-04-22 11:10:00 | 41.74 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-22 10:30:00 | 41.94 | 2026-04-22 15:20:00 | 41.59 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2026-04-27 09:30:00 | 42.11 | 2026-04-27 15:20:00 | 42.09 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest1 | 2026-05-05 09:50:00 | 42.02 | 2026-05-05 10:00:00 | 41.87 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-05-06 11:15:00 | 41.55 | 2026-05-06 12:00:00 | 41.63 | STOP_HIT | 1.00 | -0.20% |
