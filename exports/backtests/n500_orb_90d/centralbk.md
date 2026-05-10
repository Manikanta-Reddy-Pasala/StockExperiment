# Central Bank of India (CENTRALBK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 36.55
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 7
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 4.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.23% | 3.4% |
| BUY @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.23% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.13% | 1.0% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.13% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 10 | 43.5% | 3 | 13 | 7 | 0.19% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 37.50 | 37.77 | 0.00 | ORB-short ORB[37.63,38.12] vol=2.2x ATR=0.10 |
| Stop hit — per-position SL triggered | 2026-02-11 09:55:00 | 37.60 | 37.70 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 36.92 | 37.03 | 0.00 | ORB-short ORB[36.95,37.23] vol=2.0x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 37.03 | 36.99 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 37.84 | 37.55 | 0.00 | ORB-long ORB[37.23,37.53] vol=3.5x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:25:00 | 38.02 | 37.69 | 0.00 | T1 1.5R @ 38.02 |
| Target hit | 2026-02-17 10:40:00 | 38.12 | 38.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 38.44 | 38.72 | 0.00 | ORB-short ORB[38.52,39.00] vol=1.8x ATR=0.11 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 38.55 | 38.71 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 38.59 | 38.39 | 0.00 | ORB-long ORB[38.25,38.53] vol=1.8x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:50:00 | 38.78 | 38.59 | 0.00 | T1 1.5R @ 38.78 |
| Target hit | 2026-02-24 12:10:00 | 39.42 | 39.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:40:00 | 35.89 | 36.06 | 0.00 | ORB-short ORB[36.03,36.46] vol=3.0x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-03-10 10:00:00 | 36.04 | 36.04 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 35.94 | 36.19 | 0.00 | ORB-short ORB[36.17,36.50] vol=2.2x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 35.74 | 36.09 | 0.00 | T1 1.5R @ 35.74 |
| Target hit | 2026-03-13 15:20:00 | 35.51 | 35.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:15:00 | 35.41 | 35.11 | 0.00 | ORB-long ORB[34.85,35.23] vol=2.8x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-04-08 12:30:00 | 35.26 | 35.28 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 36.28 | 36.08 | 0.00 | ORB-long ORB[35.73,36.23] vol=1.8x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 36.48 | 36.16 | 0.00 | T1 1.5R @ 36.48 |
| Stop hit — per-position SL triggered | 2026-04-21 10:00:00 | 36.28 | 36.23 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 36.75 | 36.57 | 0.00 | ORB-long ORB[36.16,36.70] vol=2.2x ATR=0.13 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 36.62 | 36.57 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:00:00 | 35.87 | 35.91 | 0.00 | ORB-short ORB[35.91,36.40] vol=2.3x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:25:00 | 35.68 | 35.90 | 0.00 | T1 1.5R @ 35.68 |
| Stop hit — per-position SL triggered | 2026-04-24 10:35:00 | 35.87 | 35.87 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 36.55 | 36.36 | 0.00 | ORB-long ORB[36.19,36.48] vol=1.9x ATR=0.12 |
| Stop hit — per-position SL triggered | 2026-04-27 11:35:00 | 36.43 | 36.44 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 36.41 | 36.28 | 0.00 | ORB-long ORB[36.07,36.34] vol=1.6x ATR=0.08 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 36.33 | 36.28 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 36.51 | 36.39 | 0.00 | ORB-long ORB[36.20,36.50] vol=1.7x ATR=0.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:20:00 | 36.66 | 36.42 | 0.00 | T1 1.5R @ 36.66 |
| Stop hit — per-position SL triggered | 2026-04-29 10:35:00 | 36.51 | 36.44 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 36.32 | 36.14 | 0.00 | ORB-long ORB[35.96,36.30] vol=1.8x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-04-30 09:50:00 | 36.17 | 36.18 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 35.81 | 35.70 | 0.00 | ORB-long ORB[35.45,35.76] vol=2.2x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:45:00 | 35.95 | 35.80 | 0.00 | T1 1.5R @ 35.95 |
| Stop hit — per-position SL triggered | 2026-05-05 11:45:00 | 35.81 | 35.86 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 37.50 | 2026-02-11 09:55:00 | 37.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-13 09:30:00 | 36.92 | 2026-02-13 09:40:00 | 37.03 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 10:20:00 | 37.84 | 2026-02-17 10:25:00 | 38.02 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-17 10:20:00 | 37.84 | 2026-02-17 10:40:00 | 38.12 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2026-02-19 09:30:00 | 38.44 | 2026-02-19 09:35:00 | 38.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-24 09:45:00 | 38.59 | 2026-02-24 09:50:00 | 38.78 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-24 09:45:00 | 38.59 | 2026-02-24 12:10:00 | 39.42 | TARGET_HIT | 0.50 | 2.15% |
| SELL | retest1 | 2026-03-10 09:40:00 | 35.89 | 2026-03-10 10:00:00 | 36.04 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-13 09:50:00 | 35.94 | 2026-03-13 10:15:00 | 35.74 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-13 09:50:00 | 35.94 | 2026-03-13 15:20:00 | 35.51 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2026-04-08 10:15:00 | 35.41 | 2026-04-08 12:30:00 | 35.26 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-21 09:30:00 | 36.28 | 2026-04-21 09:45:00 | 36.48 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-21 09:30:00 | 36.28 | 2026-04-21 10:00:00 | 36.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:45:00 | 36.75 | 2026-04-22 09:50:00 | 36.62 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-24 10:00:00 | 35.87 | 2026-04-24 10:25:00 | 35.68 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-04-24 10:00:00 | 35.87 | 2026-04-24 10:35:00 | 35.87 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:45:00 | 36.55 | 2026-04-27 11:35:00 | 36.43 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-28 10:55:00 | 36.41 | 2026-04-28 11:05:00 | 36.33 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-29 10:00:00 | 36.51 | 2026-04-29 10:20:00 | 36.66 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-29 10:00:00 | 36.51 | 2026-04-29 10:35:00 | 36.51 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-30 09:30:00 | 36.32 | 2026-04-30 09:50:00 | 36.17 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-05 09:30:00 | 35.81 | 2026-05-05 09:45:00 | 35.95 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-05-05 09:30:00 | 35.81 | 2026-05-05 11:45:00 | 35.81 | STOP_HIT | 0.50 | 0.00% |
