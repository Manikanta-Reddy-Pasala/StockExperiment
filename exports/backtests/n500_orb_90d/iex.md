# Indian Energy Exchange Ltd. (IEX)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 134.07
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 5
- **Avg / median % per leg:** -0.03% / -0.24%
- **Sum % (uncompounded):** -0.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.03% | 0.3% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.03% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 3 | 23.1% | 1 | 10 | 2 | -0.08% | -1.1% |
| SELL @ 2nd Alert (retest1) | 13 | 3 | 23.1% | 1 | 10 | 2 | -0.08% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 8 | 33.3% | 3 | 16 | 5 | -0.03% | -0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:05:00 | 123.16 | 123.90 | 0.00 | ORB-short ORB[123.64,125.29] vol=4.3x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:10:00 | 122.48 | 123.18 | 0.00 | T1 1.5R @ 122.48 |
| Target hit | 2026-02-13 11:45:00 | 122.09 | 122.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 125.92 | 125.39 | 0.00 | ORB-long ORB[124.26,125.45] vol=2.0x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 125.55 | 125.41 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:05:00 | 125.67 | 125.79 | 0.00 | ORB-short ORB[125.68,126.49] vol=1.7x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:25:00 | 125.30 | 125.74 | 0.00 | T1 1.5R @ 125.30 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 125.67 | 125.64 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 125.04 | 125.84 | 0.00 | ORB-short ORB[125.60,126.52] vol=1.5x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 125.34 | 125.46 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:00:00 | 126.46 | 126.05 | 0.00 | ORB-long ORB[125.45,126.39] vol=2.3x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-02-24 10:10:00 | 126.10 | 126.07 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 127.08 | 126.55 | 0.00 | ORB-long ORB[125.63,127.05] vol=2.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:20:00 | 127.63 | 127.19 | 0.00 | T1 1.5R @ 127.63 |
| Target hit | 2026-02-25 12:35:00 | 127.36 | 127.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:40:00 | 119.42 | 120.11 | 0.00 | ORB-short ORB[119.86,120.95] vol=2.4x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-03-05 10:55:00 | 119.88 | 119.84 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 123.20 | 122.02 | 0.00 | ORB-long ORB[121.01,122.60] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-03-06 09:50:00 | 122.79 | 122.31 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 119.68 | 120.05 | 0.00 | ORB-short ORB[119.82,120.89] vol=1.8x ATR=0.42 |
| Stop hit — per-position SL triggered | 2026-03-16 09:35:00 | 120.10 | 120.04 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 122.14 | 121.35 | 0.00 | ORB-long ORB[120.02,121.37] vol=1.5x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 121.84 | 121.41 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:00:00 | 119.75 | 120.39 | 0.00 | ORB-short ORB[120.04,121.79] vol=1.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-03-19 10:35:00 | 120.19 | 120.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-23 09:40:00 | 119.32 | 118.83 | 0.00 | ORB-long ORB[118.23,119.18] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-03-23 09:50:00 | 118.82 | 118.85 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:05:00 | 116.39 | 117.11 | 0.00 | ORB-short ORB[117.01,118.56] vol=1.6x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-03-30 12:00:00 | 116.79 | 116.85 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 129.44 | 130.32 | 0.00 | ORB-short ORB[130.00,131.40] vol=2.4x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-04-10 10:10:00 | 129.89 | 130.29 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:45:00 | 132.35 | 131.37 | 0.00 | ORB-long ORB[130.45,131.73] vol=1.9x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:00:00 | 132.94 | 131.57 | 0.00 | T1 1.5R @ 132.94 |
| Target hit | 2026-04-15 15:20:00 | 132.85 | 132.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 132.91 | 133.63 | 0.00 | ORB-short ORB[133.52,134.59] vol=2.0x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 133.26 | 133.46 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 125.85 | 126.23 | 0.00 | ORB-short ORB[126.20,127.48] vol=2.4x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-29 11:10:00 | 126.17 | 126.21 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:00:00 | 124.17 | 124.59 | 0.00 | ORB-short ORB[124.44,125.70] vol=2.2x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 124.47 | 124.58 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 127.18 | 126.50 | 0.00 | ORB-long ORB[125.79,126.89] vol=5.2x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:05:00 | 127.66 | 126.68 | 0.00 | T1 1.5R @ 127.66 |
| Stop hit — per-position SL triggered | 2026-05-05 11:30:00 | 127.18 | 126.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 11:05:00 | 123.16 | 2026-02-13 11:10:00 | 122.48 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-02-13 11:05:00 | 123.16 | 2026-02-13 11:45:00 | 122.09 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2026-02-17 09:45:00 | 125.92 | 2026-02-17 09:50:00 | 125.55 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 10:05:00 | 125.67 | 2026-02-19 10:25:00 | 125.30 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-19 10:05:00 | 125.67 | 2026-02-19 11:00:00 | 125.67 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 10:40:00 | 125.04 | 2026-02-23 12:15:00 | 125.34 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-24 10:00:00 | 126.46 | 2026-02-24 10:10:00 | 126.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-25 10:05:00 | 127.08 | 2026-02-25 10:20:00 | 127.63 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-25 10:05:00 | 127.08 | 2026-02-25 12:35:00 | 127.36 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2026-03-05 09:40:00 | 119.42 | 2026-03-05 10:55:00 | 119.88 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-06 09:30:00 | 123.20 | 2026-03-06 09:50:00 | 122.79 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-16 09:30:00 | 119.68 | 2026-03-16 09:35:00 | 120.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-18 11:00:00 | 122.14 | 2026-03-18 11:15:00 | 121.84 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-19 10:00:00 | 119.75 | 2026-03-19 10:35:00 | 120.19 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-23 09:40:00 | 119.32 | 2026-03-23 09:50:00 | 118.82 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-30 11:05:00 | 116.39 | 2026-03-30 12:00:00 | 116.79 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-10 10:05:00 | 129.44 | 2026-04-10 10:10:00 | 129.89 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-15 10:45:00 | 132.35 | 2026-04-15 11:00:00 | 132.94 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-15 10:45:00 | 132.35 | 2026-04-15 15:20:00 | 132.85 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-16 09:40:00 | 132.91 | 2026-04-16 09:50:00 | 133.26 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-29 11:00:00 | 125.85 | 2026-04-29 11:10:00 | 126.17 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-30 11:00:00 | 124.17 | 2026-04-30 11:15:00 | 124.47 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-05 11:00:00 | 127.18 | 2026-05-05 11:05:00 | 127.66 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-05-05 11:00:00 | 127.18 | 2026-05-05 11:30:00 | 127.18 | STOP_HIT | 0.50 | 0.00% |
