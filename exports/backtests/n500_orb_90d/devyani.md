# Devyani International Ltd. (DEVYANI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 118.50
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
| PARTIAL | 8 |
| TARGET_HIT | 5 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 7
- **Target hits / Stop hits / Partials:** 5 / 7 / 8
- **Avg / median % per leg:** 0.96% / 0.53%
- **Sum % (uncompounded):** 19.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 9 | 69.2% | 4 | 4 | 5 | 1.35% | 17.6% |
| BUY @ 2nd Alert (retest1) | 13 | 9 | 69.2% | 4 | 4 | 5 | 1.35% | 17.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.23% | 1.6% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.23% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 13 | 65.0% | 5 | 7 | 8 | 0.96% | 19.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:20:00 | 133.05 | 132.02 | 0.00 | ORB-long ORB[131.43,132.68] vol=2.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:25:00 | 133.81 | 132.23 | 0.00 | T1 1.5R @ 133.81 |
| Target hit | 2026-02-10 15:20:00 | 136.73 | 135.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 129.74 | 130.25 | 0.00 | ORB-short ORB[129.87,131.50] vol=2.3x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:20:00 | 129.15 | 129.82 | 0.00 | T1 1.5R @ 129.15 |
| Target hit | 2026-02-16 15:20:00 | 128.99 | 129.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 130.39 | 129.61 | 0.00 | ORB-long ORB[128.65,130.00] vol=2.1x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:00:00 | 131.04 | 130.22 | 0.00 | T1 1.5R @ 131.04 |
| Target hit | 2026-02-17 12:00:00 | 131.08 | 131.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 133.40 | 132.17 | 0.00 | ORB-long ORB[130.63,132.51] vol=2.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2026-02-18 10:35:00 | 132.74 | 132.82 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 127.20 | 127.88 | 0.00 | ORB-short ORB[127.39,129.29] vol=1.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 126.55 | 127.45 | 0.00 | T1 1.5R @ 126.55 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 127.20 | 127.08 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:55:00 | 119.12 | 118.45 | 0.00 | ORB-long ORB[117.11,118.80] vol=2.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2026-03-06 10:35:00 | 118.34 | 118.67 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 110.23 | 109.00 | 0.00 | ORB-long ORB[107.80,109.10] vol=4.2x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:05:00 | 110.89 | 109.33 | 0.00 | T1 1.5R @ 110.89 |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 110.23 | 109.58 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:45:00 | 98.99 | 100.49 | 0.00 | ORB-short ORB[100.40,101.80] vol=1.9x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-03-27 10:55:00 | 99.46 | 100.41 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 108.15 | 107.29 | 0.00 | ORB-long ORB[106.57,107.60] vol=1.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:00:00 | 108.89 | 107.91 | 0.00 | T1 1.5R @ 108.89 |
| Target hit | 2026-04-21 11:15:00 | 109.06 | 109.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:15:00 | 112.53 | 111.26 | 0.00 | ORB-long ORB[110.42,112.00] vol=2.0x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-04-28 10:20:00 | 112.04 | 111.40 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 114.86 | 112.78 | 0.00 | ORB-long ORB[112.04,113.71] vol=4.7x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:30:00 | 115.74 | 113.93 | 0.00 | T1 1.5R @ 115.74 |
| Target hit | 2026-04-29 15:20:00 | 128.59 | 124.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 117.30 | 118.47 | 0.00 | ORB-short ORB[118.39,119.34] vol=1.8x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:05:00 | 116.65 | 118.01 | 0.00 | T1 1.5R @ 116.65 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 117.30 | 117.99 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:20:00 | 133.05 | 2026-02-10 10:25:00 | 133.81 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-10 10:20:00 | 133.05 | 2026-02-10 15:20:00 | 136.73 | TARGET_HIT | 0.50 | 2.77% |
| SELL | retest1 | 2026-02-16 11:00:00 | 129.74 | 2026-02-16 11:20:00 | 129.15 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-16 11:00:00 | 129.74 | 2026-02-16 15:20:00 | 128.99 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-17 09:50:00 | 130.39 | 2026-02-17 10:00:00 | 131.04 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-17 09:50:00 | 130.39 | 2026-02-17 12:00:00 | 131.08 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-18 09:35:00 | 133.40 | 2026-02-18 10:35:00 | 132.74 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-02-27 09:35:00 | 127.20 | 2026-02-27 10:15:00 | 126.55 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-27 09:35:00 | 127.20 | 2026-02-27 11:15:00 | 127.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 09:55:00 | 119.12 | 2026-03-06 10:35:00 | 118.34 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2026-03-18 11:00:00 | 110.23 | 2026-03-18 11:05:00 | 110.89 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-18 11:00:00 | 110.23 | 2026-03-18 11:15:00 | 110.23 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 10:45:00 | 98.99 | 2026-03-27 10:55:00 | 99.46 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-21 09:35:00 | 108.15 | 2026-04-21 10:00:00 | 108.89 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-21 09:35:00 | 108.15 | 2026-04-21 11:15:00 | 109.06 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2026-04-28 10:15:00 | 112.53 | 2026-04-28 10:20:00 | 112.04 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-29 11:15:00 | 114.86 | 2026-04-29 11:30:00 | 115.74 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-04-29 11:15:00 | 114.86 | 2026-04-29 15:20:00 | 128.59 | TARGET_HIT | 0.50 | 11.95% |
| SELL | retest1 | 2026-05-08 09:45:00 | 117.30 | 2026-05-08 10:05:00 | 116.65 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-08 09:45:00 | 117.30 | 2026-05-08 10:10:00 | 117.30 | STOP_HIT | 0.50 | 0.00% |
