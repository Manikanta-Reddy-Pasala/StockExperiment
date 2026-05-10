# RITES Ltd. (RITES)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 226.80
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 18
- **Target hits / Stop hits / Partials:** 4 / 18 / 12
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 7.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 9 | 50.0% | 2 | 9 | 7 | 0.19% | 3.5% |
| BUY @ 2nd Alert (retest1) | 18 | 9 | 50.0% | 2 | 9 | 7 | 0.19% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.23% | 3.7% |
| SELL @ 2nd Alert (retest1) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.23% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 34 | 16 | 47.1% | 4 | 18 | 12 | 0.21% | 7.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 222.00 | 223.03 | 0.00 | ORB-short ORB[222.52,225.55] vol=1.7x ATR=0.53 |
| Stop hit — per-position SL triggered | 2026-02-12 12:25:00 | 222.53 | 222.78 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 219.06 | 219.89 | 0.00 | ORB-short ORB[219.17,221.50] vol=1.6x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:35:00 | 218.08 | 219.49 | 0.00 | T1 1.5R @ 218.08 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 219.06 | 219.51 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:00:00 | 220.12 | 218.80 | 0.00 | ORB-long ORB[216.50,219.40] vol=1.7x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:40:00 | 221.27 | 219.27 | 0.00 | T1 1.5R @ 221.27 |
| Stop hit — per-position SL triggered | 2026-02-16 12:50:00 | 220.12 | 219.67 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 219.43 | 220.49 | 0.00 | ORB-short ORB[220.51,221.56] vol=3.9x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:25:00 | 218.67 | 220.07 | 0.00 | T1 1.5R @ 218.67 |
| Stop hit — per-position SL triggered | 2026-02-18 10:30:00 | 219.43 | 220.05 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 220.41 | 221.21 | 0.00 | ORB-short ORB[220.50,222.25] vol=2.1x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 220.92 | 221.07 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:40:00 | 216.50 | 216.87 | 0.00 | ORB-short ORB[216.70,218.79] vol=1.9x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:25:00 | 215.56 | 216.46 | 0.00 | T1 1.5R @ 215.56 |
| Stop hit — per-position SL triggered | 2026-02-24 15:00:00 | 216.50 | 216.20 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 213.60 | 214.31 | 0.00 | ORB-short ORB[213.77,215.88] vol=1.7x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-02-27 13:55:00 | 214.11 | 213.96 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:00:00 | 210.65 | 208.33 | 0.00 | ORB-long ORB[200.87,204.00] vol=3.9x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 209.22 | 208.72 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 200.97 | 201.59 | 0.00 | ORB-short ORB[201.21,204.11] vol=2.7x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 201.84 | 201.46 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 201.86 | 202.78 | 0.00 | ORB-short ORB[202.50,205.00] vol=1.6x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 200.93 | 202.63 | 0.00 | T1 1.5R @ 200.93 |
| Target hit | 2026-03-11 15:20:00 | 199.28 | 201.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 197.26 | 198.09 | 0.00 | ORB-short ORB[197.85,200.08] vol=3.0x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-03-13 10:25:00 | 197.89 | 197.99 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:50:00 | 187.57 | 188.79 | 0.00 | ORB-short ORB[188.10,190.44] vol=2.0x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:05:00 | 186.38 | 188.43 | 0.00 | T1 1.5R @ 186.38 |
| Target hit | 2026-03-27 15:20:00 | 184.24 | 185.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 199.68 | 198.15 | 0.00 | ORB-long ORB[196.11,198.90] vol=3.0x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-04-08 09:55:00 | 198.59 | 198.24 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:05:00 | 204.18 | 201.65 | 0.00 | ORB-long ORB[199.00,202.00] vol=2.1x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:45:00 | 205.62 | 202.34 | 0.00 | T1 1.5R @ 205.62 |
| Target hit | 2026-04-13 15:20:00 | 205.44 | 204.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 212.11 | 210.78 | 0.00 | ORB-long ORB[208.67,211.40] vol=3.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:50:00 | 213.65 | 211.37 | 0.00 | T1 1.5R @ 213.65 |
| Stop hit — per-position SL triggered | 2026-04-15 09:55:00 | 212.11 | 211.44 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 216.60 | 214.80 | 0.00 | ORB-long ORB[214.10,216.26] vol=3.1x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:20:00 | 217.69 | 215.18 | 0.00 | T1 1.5R @ 217.69 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 216.60 | 215.28 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 223.10 | 221.00 | 0.00 | ORB-long ORB[218.50,220.89] vol=5.8x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:10:00 | 224.33 | 222.37 | 0.00 | T1 1.5R @ 224.33 |
| Stop hit — per-position SL triggered | 2026-04-21 11:20:00 | 223.10 | 222.73 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 221.94 | 220.56 | 0.00 | ORB-long ORB[218.50,221.61] vol=3.4x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 221.20 | 220.63 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 220.33 | 219.42 | 0.00 | ORB-long ORB[217.30,219.92] vol=1.8x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 221.54 | 220.31 | 0.00 | T1 1.5R @ 221.54 |
| Target hit | 2026-04-27 11:30:00 | 221.48 | 221.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — SELL (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 220.80 | 221.48 | 0.00 | ORB-short ORB[221.15,223.59] vol=1.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-04-29 11:20:00 | 221.30 | 221.46 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 219.98 | 218.33 | 0.00 | ORB-long ORB[217.00,219.29] vol=2.7x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-04-30 09:35:00 | 219.23 | 218.37 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 223.38 | 221.55 | 0.00 | ORB-long ORB[218.15,220.39] vol=6.9x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:40:00 | 224.94 | 222.68 | 0.00 | T1 1.5R @ 224.94 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 223.38 | 222.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:55:00 | 222.00 | 2026-02-12 12:25:00 | 222.53 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-13 09:30:00 | 219.06 | 2026-02-13 09:35:00 | 218.08 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-13 09:30:00 | 219.06 | 2026-02-13 09:40:00 | 219.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:00:00 | 220.12 | 2026-02-16 10:40:00 | 221.27 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-16 10:00:00 | 220.12 | 2026-02-16 12:50:00 | 220.12 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 09:55:00 | 219.43 | 2026-02-18 10:25:00 | 218.67 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-18 09:55:00 | 219.43 | 2026-02-18 10:30:00 | 219.43 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 09:35:00 | 220.41 | 2026-02-19 09:45:00 | 220.92 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-24 09:40:00 | 216.50 | 2026-02-24 12:25:00 | 215.56 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-24 09:40:00 | 216.50 | 2026-02-24 15:00:00 | 216.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:55:00 | 213.60 | 2026-02-27 13:55:00 | 214.11 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-06 10:00:00 | 210.65 | 2026-03-06 10:15:00 | 209.22 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest1 | 2026-03-10 09:30:00 | 200.97 | 2026-03-10 10:15:00 | 201.84 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-11 10:55:00 | 201.86 | 2026-03-11 11:20:00 | 200.93 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-11 10:55:00 | 201.86 | 2026-03-11 15:20:00 | 199.28 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2026-03-13 10:00:00 | 197.26 | 2026-03-13 10:25:00 | 197.89 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-27 09:50:00 | 187.57 | 2026-03-27 10:05:00 | 186.38 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-27 09:50:00 | 187.57 | 2026-03-27 15:20:00 | 184.24 | TARGET_HIT | 0.50 | 1.78% |
| BUY | retest1 | 2026-04-08 09:45:00 | 199.68 | 2026-04-08 09:55:00 | 198.59 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-04-13 10:05:00 | 204.18 | 2026-04-13 10:45:00 | 205.62 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-04-13 10:05:00 | 204.18 | 2026-04-13 15:20:00 | 205.44 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-15 09:35:00 | 212.11 | 2026-04-15 09:50:00 | 213.65 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-04-15 09:35:00 | 212.11 | 2026-04-15 09:55:00 | 212.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-16 11:15:00 | 216.60 | 2026-04-16 11:20:00 | 217.69 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-16 11:15:00 | 216.60 | 2026-04-16 11:25:00 | 216.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:50:00 | 223.10 | 2026-04-21 10:10:00 | 224.33 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-21 09:50:00 | 223.10 | 2026-04-21 11:20:00 | 223.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:45:00 | 221.94 | 2026-04-22 10:55:00 | 221.20 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-27 09:30:00 | 220.33 | 2026-04-27 09:45:00 | 221.54 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-27 09:30:00 | 220.33 | 2026-04-27 11:30:00 | 221.48 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2026-04-29 11:00:00 | 220.80 | 2026-04-29 11:20:00 | 221.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-30 09:30:00 | 219.98 | 2026-04-30 09:35:00 | 219.23 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-05 09:35:00 | 223.38 | 2026-05-05 09:40:00 | 224.94 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-05-05 09:35:00 | 223.38 | 2026-05-05 09:45:00 | 223.38 | STOP_HIT | 0.50 | 0.00% |
