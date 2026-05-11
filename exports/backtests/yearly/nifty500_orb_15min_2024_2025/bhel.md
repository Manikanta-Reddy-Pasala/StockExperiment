# Bharat Heavy Electricals Ltd. (BHEL)

## Backtest Summary

- **Window:** 2024-10-08 09:15:00 → 2026-05-08 15:25:00 (27550 bars)
- **Last close:** 403.20
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
| ENTRY1 | 27 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 4 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 23
- **Target hits / Stop hits / Partials:** 4 / 23 / 13
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 5.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 11 | 44.0% | 3 | 14 | 8 | 0.15% | 3.9% |
| BUY @ 2nd Alert (retest1) | 25 | 11 | 44.0% | 3 | 14 | 8 | 0.15% | 3.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.12% | 1.8% |
| SELL @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.12% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 40 | 17 | 42.5% | 4 | 23 | 13 | 0.14% | 5.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:35:00 | 272.00 | 271.10 | 0.00 | ORB-long ORB[270.00,271.65] vol=2.2x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-10-14 10:05:00 | 271.12 | 271.35 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:55:00 | 240.06 | 237.69 | 0.00 | ORB-long ORB[234.81,238.29] vol=1.6x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 11:10:00 | 241.52 | 238.24 | 0.00 | T1 1.5R @ 241.52 |
| Stop hit — per-position SL triggered | 2024-11-11 12:10:00 | 240.06 | 238.77 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:00:00 | 235.30 | 236.84 | 0.00 | ORB-short ORB[236.85,239.84] vol=1.6x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 10:10:00 | 233.82 | 236.44 | 0.00 | T1 1.5R @ 233.82 |
| Stop hit — per-position SL triggered | 2024-11-12 10:25:00 | 235.30 | 236.19 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:30:00 | 221.96 | 223.84 | 0.00 | ORB-short ORB[223.01,226.00] vol=2.9x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:40:00 | 220.49 | 223.26 | 0.00 | T1 1.5R @ 220.49 |
| Stop hit — per-position SL triggered | 2024-11-18 09:45:00 | 221.96 | 222.80 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:30:00 | 245.35 | 243.13 | 0.00 | ORB-long ORB[241.50,243.78] vol=2.5x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-11-27 09:35:00 | 244.48 | 243.31 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-11-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:40:00 | 251.40 | 249.60 | 0.00 | ORB-long ORB[247.66,250.30] vol=2.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:00:00 | 252.73 | 250.78 | 0.00 | T1 1.5R @ 252.73 |
| Stop hit — per-position SL triggered | 2024-11-28 10:35:00 | 251.40 | 251.32 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-12-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:55:00 | 250.10 | 251.53 | 0.00 | ORB-short ORB[250.70,252.85] vol=1.6x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-12-06 10:00:00 | 250.90 | 251.37 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-12-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:00:00 | 251.80 | 249.45 | 0.00 | ORB-long ORB[247.60,250.70] vol=1.7x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:05:00 | 253.11 | 250.40 | 0.00 | T1 1.5R @ 253.11 |
| Stop hit — per-position SL triggered | 2024-12-10 11:35:00 | 251.80 | 252.23 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 249.60 | 251.68 | 0.00 | ORB-short ORB[251.75,254.45] vol=2.7x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:45:00 | 248.47 | 251.06 | 0.00 | T1 1.5R @ 248.47 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 249.60 | 250.85 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:35:00 | 239.55 | 241.73 | 0.00 | ORB-short ORB[240.90,244.15] vol=1.5x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-12-13 09:40:00 | 240.51 | 241.54 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:05:00 | 230.33 | 231.65 | 0.00 | ORB-short ORB[231.00,233.50] vol=1.6x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:15:00 | 229.33 | 231.38 | 0.00 | T1 1.5R @ 229.33 |
| Stop hit — per-position SL triggered | 2025-01-02 12:50:00 | 230.33 | 230.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:15:00 | 206.90 | 208.26 | 0.00 | ORB-short ORB[208.25,211.10] vol=1.8x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:50:00 | 205.91 | 208.06 | 0.00 | T1 1.5R @ 205.91 |
| Target hit | 2025-02-06 15:20:00 | 204.92 | 206.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-03-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:25:00 | 200.60 | 198.80 | 0.00 | ORB-long ORB[197.04,199.85] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-03-06 10:45:00 | 199.63 | 199.32 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-03-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:20:00 | 197.00 | 195.09 | 0.00 | ORB-long ORB[192.90,195.45] vol=3.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-03-13 10:35:00 | 196.25 | 195.26 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:10:00 | 199.99 | 198.42 | 0.00 | ORB-long ORB[196.16,198.50] vol=3.3x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 12:10:00 | 200.77 | 198.97 | 0.00 | T1 1.5R @ 200.77 |
| Stop hit — per-position SL triggered | 2025-03-18 12:30:00 | 199.99 | 199.42 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-03-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:05:00 | 202.54 | 205.32 | 0.00 | ORB-short ORB[205.21,207.40] vol=2.1x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-03-20 10:10:00 | 203.40 | 205.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 209.81 | 208.35 | 0.00 | ORB-long ORB[206.30,209.00] vol=2.4x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-03-21 10:00:00 | 209.10 | 209.05 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-03-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:05:00 | 216.50 | 214.67 | 0.00 | ORB-long ORB[213.23,215.61] vol=1.7x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 10:10:00 | 218.02 | 215.35 | 0.00 | T1 1.5R @ 218.02 |
| Target hit | 2025-03-24 12:00:00 | 217.32 | 217.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2025-04-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:35:00 | 217.47 | 216.00 | 0.00 | ORB-long ORB[214.55,216.90] vol=2.1x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-04-11 09:55:00 | 216.56 | 216.17 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:35:00 | 218.12 | 216.84 | 0.00 | ORB-long ORB[214.15,217.30] vol=1.6x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 09:40:00 | 219.21 | 217.74 | 0.00 | T1 1.5R @ 219.21 |
| Target hit | 2025-04-15 15:20:00 | 222.64 | 220.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:45:00 | 223.50 | 222.26 | 0.00 | ORB-long ORB[220.75,223.00] vol=1.8x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:00:00 | 224.68 | 222.80 | 0.00 | T1 1.5R @ 224.68 |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 223.50 | 223.05 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 231.15 | 229.41 | 0.00 | ORB-long ORB[227.10,229.90] vol=2.4x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-04-22 10:05:00 | 230.22 | 230.54 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 228.67 | 229.37 | 0.00 | ORB-short ORB[229.15,230.54] vol=1.6x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-04-23 09:40:00 | 229.39 | 229.35 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-04-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 11:05:00 | 231.00 | 229.88 | 0.00 | ORB-long ORB[228.60,230.84] vol=3.4x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-04-24 11:10:00 | 230.23 | 229.90 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:55:00 | 234.50 | 233.05 | 0.00 | ORB-long ORB[230.24,233.55] vol=2.5x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-04-29 10:20:00 | 233.37 | 233.37 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:15:00 | 227.53 | 226.62 | 0.00 | ORB-long ORB[225.00,226.99] vol=1.8x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 12:00:00 | 228.53 | 226.82 | 0.00 | T1 1.5R @ 228.53 |
| Target hit | 2025-05-05 15:20:00 | 229.15 | 227.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 09:50:00 | 224.99 | 226.64 | 0.00 | ORB-short ORB[226.25,228.95] vol=1.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-05-06 10:45:00 | 225.75 | 225.94 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-14 09:35:00 | 272.00 | 2024-10-14 10:05:00 | 271.12 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-11 10:55:00 | 240.06 | 2024-11-11 11:10:00 | 241.52 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-11-11 10:55:00 | 240.06 | 2024-11-11 12:10:00 | 240.06 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 10:00:00 | 235.30 | 2024-11-12 10:10:00 | 233.82 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-11-12 10:00:00 | 235.30 | 2024-11-12 10:25:00 | 235.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-18 09:30:00 | 221.96 | 2024-11-18 09:40:00 | 220.49 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-11-18 09:30:00 | 221.96 | 2024-11-18 09:45:00 | 221.96 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:30:00 | 245.35 | 2024-11-27 09:35:00 | 244.48 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-28 09:40:00 | 251.40 | 2024-11-28 10:00:00 | 252.73 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-11-28 09:40:00 | 251.40 | 2024-11-28 10:35:00 | 251.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-06 09:55:00 | 250.10 | 2024-12-06 10:00:00 | 250.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-10 10:00:00 | 251.80 | 2024-12-10 10:05:00 | 253.11 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-12-10 10:00:00 | 251.80 | 2024-12-10 11:35:00 | 251.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 09:40:00 | 249.60 | 2024-12-12 09:45:00 | 248.47 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-12 09:40:00 | 249.60 | 2024-12-12 09:50:00 | 249.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 09:35:00 | 239.55 | 2024-12-13 09:40:00 | 240.51 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-02 11:05:00 | 230.33 | 2025-01-02 11:15:00 | 229.33 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-02 11:05:00 | 230.33 | 2025-01-02 12:50:00 | 230.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 11:15:00 | 206.90 | 2025-02-06 11:50:00 | 205.91 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-02-06 11:15:00 | 206.90 | 2025-02-06 15:20:00 | 204.92 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2025-03-06 10:25:00 | 200.60 | 2025-03-06 10:45:00 | 199.63 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-03-13 10:20:00 | 197.00 | 2025-03-13 10:35:00 | 196.25 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-03-18 11:10:00 | 199.99 | 2025-03-18 12:10:00 | 200.77 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-03-18 11:10:00 | 199.99 | 2025-03-18 12:30:00 | 199.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-20 10:05:00 | 202.54 | 2025-03-20 10:10:00 | 203.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-21 09:30:00 | 209.81 | 2025-03-21 10:00:00 | 209.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-24 10:05:00 | 216.50 | 2025-03-24 10:10:00 | 218.02 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-03-24 10:05:00 | 216.50 | 2025-03-24 12:00:00 | 217.32 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2025-04-11 09:35:00 | 217.47 | 2025-04-11 09:55:00 | 216.56 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-15 09:35:00 | 218.12 | 2025-04-15 09:40:00 | 219.21 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-15 09:35:00 | 218.12 | 2025-04-15 15:20:00 | 222.64 | TARGET_HIT | 0.50 | 2.07% |
| BUY | retest1 | 2025-04-16 09:45:00 | 223.50 | 2025-04-16 10:00:00 | 224.68 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-16 09:45:00 | 223.50 | 2025-04-16 10:15:00 | 223.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 09:30:00 | 231.15 | 2025-04-22 10:05:00 | 230.22 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-23 09:35:00 | 228.67 | 2025-04-23 09:40:00 | 229.39 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-24 11:05:00 | 231.00 | 2025-04-24 11:10:00 | 230.23 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-29 09:55:00 | 234.50 | 2025-04-29 10:20:00 | 233.37 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-05-05 11:15:00 | 227.53 | 2025-05-05 12:00:00 | 228.53 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-05-05 11:15:00 | 227.53 | 2025-05-05 15:20:00 | 229.15 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2025-05-06 09:50:00 | 224.99 | 2025-05-06 10:45:00 | 225.75 | STOP_HIT | 1.00 | -0.34% |
