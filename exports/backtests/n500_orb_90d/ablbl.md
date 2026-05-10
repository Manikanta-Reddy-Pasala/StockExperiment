# Aditya Birla Lifestyle Brands Ltd. (ABLBL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 114.00
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 8 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 7
- **Target hits / Stop hits / Partials:** 8 / 7 / 9
- **Avg / median % per leg:** 0.56% / 0.51%
- **Sum % (uncompounded):** 13.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 8 | 61.5% | 3 | 5 | 5 | 0.55% | 7.2% |
| BUY @ 2nd Alert (retest1) | 13 | 8 | 61.5% | 3 | 5 | 5 | 0.55% | 7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 9 | 81.8% | 5 | 2 | 4 | 0.56% | 6.2% |
| SELL @ 2nd Alert (retest1) | 11 | 9 | 81.8% | 5 | 2 | 4 | 0.56% | 6.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 17 | 70.8% | 8 | 7 | 9 | 0.56% | 13.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 114.09 | 113.16 | 0.00 | ORB-long ORB[111.31,113.01] vol=2.5x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:40:00 | 115.09 | 113.42 | 0.00 | T1 1.5R @ 115.09 |
| Target hit | 2026-02-09 15:20:00 | 114.94 | 114.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 107.52 | 106.82 | 0.00 | ORB-long ORB[106.01,107.18] vol=2.0x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-02-20 10:10:00 | 107.06 | 106.97 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 105.00 | 105.43 | 0.00 | ORB-short ORB[105.31,106.38] vol=1.6x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 104.59 | 105.21 | 0.00 | T1 1.5R @ 104.59 |
| Target hit | 2026-02-24 14:55:00 | 104.46 | 104.45 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 105.43 | 105.09 | 0.00 | ORB-long ORB[104.55,105.40] vol=2.5x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:00:00 | 105.97 | 105.33 | 0.00 | T1 1.5R @ 105.97 |
| Stop hit — per-position SL triggered | 2026-02-26 10:10:00 | 105.43 | 105.39 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-02 09:40:00 | 100.62 | 99.74 | 0.00 | ORB-long ORB[99.00,100.10] vol=1.5x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-03-02 10:30:00 | 100.11 | 100.04 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:10:00 | 100.00 | 98.96 | 0.00 | ORB-long ORB[98.08,99.34] vol=3.1x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:20:00 | 100.74 | 99.49 | 0.00 | T1 1.5R @ 100.74 |
| Target hit | 2026-03-10 15:20:00 | 103.98 | 102.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 98.73 | 99.95 | 0.00 | ORB-short ORB[99.85,101.32] vol=2.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:55:00 | 98.05 | 99.56 | 0.00 | T1 1.5R @ 98.05 |
| Target hit | 2026-03-13 14:30:00 | 98.47 | 97.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2026-03-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:55:00 | 95.60 | 96.27 | 0.00 | ORB-short ORB[96.22,97.37] vol=1.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:55:00 | 95.08 | 95.85 | 0.00 | T1 1.5R @ 95.08 |
| Target hit | 2026-03-19 15:20:00 | 93.55 | 94.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:55:00 | 91.09 | 92.15 | 0.00 | ORB-short ORB[92.28,93.54] vol=4.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 91.54 | 92.03 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 100.91 | 100.68 | 0.00 | ORB-long ORB[99.79,100.50] vol=1.5x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:15:00 | 101.49 | 100.73 | 0.00 | T1 1.5R @ 101.49 |
| Target hit | 2026-04-10 11:35:00 | 101.27 | 101.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 107.57 | 105.88 | 0.00 | ORB-long ORB[104.25,105.80] vol=1.9x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:25:00 | 108.29 | 106.47 | 0.00 | T1 1.5R @ 108.29 |
| Stop hit — per-position SL triggered | 2026-04-15 10:35:00 | 107.57 | 106.58 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:50:00 | 105.81 | 106.16 | 0.00 | ORB-short ORB[105.90,107.05] vol=2.1x ATR=0.44 |
| Target hit | 2026-04-17 15:20:00 | 105.76 | 105.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 107.58 | 106.50 | 0.00 | ORB-long ORB[105.30,106.37] vol=3.4x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 107.21 | 106.68 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:40:00 | 107.96 | 108.58 | 0.00 | ORB-short ORB[108.00,109.00] vol=1.7x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:20:00 | 107.48 | 108.48 | 0.00 | T1 1.5R @ 107.48 |
| Target hit | 2026-04-23 15:20:00 | 105.94 | 107.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2026-04-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:30:00 | 105.20 | 105.50 | 0.00 | ORB-short ORB[105.54,105.97] vol=1.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-04-28 11:10:00 | 105.48 | 105.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 114.09 | 2026-02-09 10:40:00 | 115.09 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2026-02-09 10:30:00 | 114.09 | 2026-02-09 15:20:00 | 114.94 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2026-02-20 09:45:00 | 107.52 | 2026-02-20 10:10:00 | 107.06 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-24 10:45:00 | 105.00 | 2026-02-24 11:45:00 | 104.59 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-24 10:45:00 | 105.00 | 2026-02-24 14:55:00 | 104.46 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-26 09:45:00 | 105.43 | 2026-02-26 10:00:00 | 105.97 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-26 09:45:00 | 105.43 | 2026-02-26 10:10:00 | 105.43 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-02 09:40:00 | 100.62 | 2026-03-02 10:30:00 | 100.11 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-03-10 10:10:00 | 100.00 | 2026-03-10 11:20:00 | 100.74 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-03-10 10:10:00 | 100.00 | 2026-03-10 15:20:00 | 103.98 | TARGET_HIT | 0.50 | 3.98% |
| SELL | retest1 | 2026-03-13 09:50:00 | 98.73 | 2026-03-13 09:55:00 | 98.05 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-13 09:50:00 | 98.73 | 2026-03-13 14:30:00 | 98.47 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2026-03-19 10:55:00 | 95.60 | 2026-03-19 11:55:00 | 95.08 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-19 10:55:00 | 95.60 | 2026-03-19 15:20:00 | 93.55 | TARGET_HIT | 0.50 | 2.14% |
| SELL | retest1 | 2026-03-23 10:55:00 | 91.09 | 2026-03-23 11:05:00 | 91.54 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-10 10:05:00 | 100.91 | 2026-04-10 10:15:00 | 101.49 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-10 10:05:00 | 100.91 | 2026-04-10 11:35:00 | 101.27 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-15 10:05:00 | 107.57 | 2026-04-15 10:25:00 | 108.29 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-04-15 10:05:00 | 107.57 | 2026-04-15 10:35:00 | 107.57 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-17 10:50:00 | 105.81 | 2026-04-17 15:20:00 | 105.76 | TARGET_HIT | 1.00 | 0.05% |
| BUY | retest1 | 2026-04-21 10:40:00 | 107.58 | 2026-04-21 11:00:00 | 107.21 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-23 10:40:00 | 107.96 | 2026-04-23 11:20:00 | 107.48 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-23 10:40:00 | 107.96 | 2026-04-23 15:20:00 | 105.94 | TARGET_HIT | 0.50 | 1.87% |
| SELL | retest1 | 2026-04-28 10:30:00 | 105.20 | 2026-04-28 11:10:00 | 105.48 | STOP_HIT | 1.00 | -0.27% |
