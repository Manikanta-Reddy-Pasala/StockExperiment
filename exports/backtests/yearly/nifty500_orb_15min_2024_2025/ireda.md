# Indian Renewable Energy Development Agency Ltd. (IREDA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 134.70
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
| ENTRY1 | 30 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 26
- **Target hits / Stop hits / Partials:** 4 / 26 / 12
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 1.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 4 | 23.5% | 1 | 13 | 3 | -0.13% | -2.3% |
| BUY @ 2nd Alert (retest1) | 17 | 4 | 23.5% | 1 | 13 | 3 | -0.13% | -2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 12 | 48.0% | 3 | 13 | 9 | 0.15% | 3.7% |
| SELL @ 2nd Alert (retest1) | 25 | 12 | 48.0% | 3 | 13 | 9 | 0.15% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 42 | 16 | 38.1% | 4 | 26 | 12 | 0.03% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 172.00 | 172.95 | 0.00 | ORB-short ORB[172.35,174.25] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-05-16 10:10:00 | 172.74 | 172.67 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 181.45 | 182.93 | 0.00 | ORB-short ORB[181.70,184.20] vol=1.5x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:50:00 | 180.12 | 182.64 | 0.00 | T1 1.5R @ 180.12 |
| Target hit | 2024-05-31 12:50:00 | 180.70 | 180.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 180.12 | 181.55 | 0.00 | ORB-short ORB[180.60,182.70] vol=2.0x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 09:35:00 | 178.82 | 181.20 | 0.00 | T1 1.5R @ 178.82 |
| Stop hit — per-position SL triggered | 2024-06-10 09:40:00 | 180.12 | 181.04 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 181.98 | 182.82 | 0.00 | ORB-short ORB[182.10,184.00] vol=2.3x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:45:00 | 181.19 | 182.50 | 0.00 | T1 1.5R @ 181.19 |
| Stop hit — per-position SL triggered | 2024-06-13 09:55:00 | 181.98 | 182.41 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:50:00 | 180.91 | 179.96 | 0.00 | ORB-long ORB[178.36,180.80] vol=2.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-06-18 09:55:00 | 180.26 | 179.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:40:00 | 188.76 | 187.36 | 0.00 | ORB-long ORB[183.10,185.90] vol=5.8x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-06-26 09:55:00 | 187.67 | 187.46 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 195.10 | 196.02 | 0.00 | ORB-short ORB[195.15,197.75] vol=1.6x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 09:45:00 | 194.08 | 195.72 | 0.00 | T1 1.5R @ 194.08 |
| Stop hit — per-position SL triggered | 2024-07-02 09:50:00 | 195.10 | 195.70 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 266.96 | 264.92 | 0.00 | ORB-long ORB[262.85,265.80] vol=2.6x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-07-26 09:35:00 | 265.67 | 265.07 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-08-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:45:00 | 241.50 | 240.06 | 0.00 | ORB-long ORB[238.25,241.40] vol=2.3x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-08-16 09:55:00 | 240.36 | 240.11 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 242.20 | 243.40 | 0.00 | ORB-short ORB[242.35,245.95] vol=1.7x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:35:00 | 241.04 | 242.57 | 0.00 | T1 1.5R @ 241.04 |
| Target hit | 2024-08-20 15:20:00 | 240.50 | 241.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 251.90 | 252.71 | 0.00 | ORB-short ORB[252.00,254.00] vol=1.8x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:55:00 | 250.84 | 252.47 | 0.00 | T1 1.5R @ 250.84 |
| Stop hit — per-position SL triggered | 2024-08-27 10:05:00 | 251.90 | 252.37 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:15:00 | 234.91 | 237.13 | 0.00 | ORB-short ORB[236.26,238.65] vol=1.5x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-09-05 10:25:00 | 235.60 | 236.96 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:05:00 | 224.16 | 226.13 | 0.00 | ORB-short ORB[226.61,228.58] vol=2.0x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:25:00 | 223.30 | 225.61 | 0.00 | T1 1.5R @ 223.30 |
| Stop hit — per-position SL triggered | 2024-09-25 11:05:00 | 224.16 | 225.05 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 219.42 | 220.62 | 0.00 | ORB-short ORB[219.98,222.85] vol=4.1x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:35:00 | 218.35 | 219.98 | 0.00 | T1 1.5R @ 218.35 |
| Stop hit — per-position SL triggered | 2024-09-26 10:05:00 | 219.42 | 219.22 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:05:00 | 229.49 | 227.48 | 0.00 | ORB-long ORB[225.51,227.99] vol=1.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-10-09 11:55:00 | 228.57 | 227.73 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:40:00 | 192.11 | 190.67 | 0.00 | ORB-long ORB[189.50,191.30] vol=2.3x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 09:45:00 | 193.29 | 191.28 | 0.00 | T1 1.5R @ 193.29 |
| Stop hit — per-position SL triggered | 2024-11-27 10:10:00 | 192.11 | 191.79 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 206.11 | 203.78 | 0.00 | ORB-long ORB[201.00,203.90] vol=3.4x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-12-02 09:50:00 | 204.97 | 204.65 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:40:00 | 210.89 | 209.56 | 0.00 | ORB-long ORB[206.36,209.47] vol=7.4x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:00:00 | 212.31 | 210.33 | 0.00 | T1 1.5R @ 212.31 |
| Stop hit — per-position SL triggered | 2024-12-04 10:20:00 | 210.89 | 210.62 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 224.84 | 223.83 | 0.00 | ORB-long ORB[222.00,224.80] vol=1.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-12-09 09:35:00 | 223.73 | 223.83 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:45:00 | 225.59 | 223.39 | 0.00 | ORB-long ORB[221.69,224.79] vol=3.0x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-12-11 09:50:00 | 224.45 | 223.52 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 221.42 | 222.87 | 0.00 | ORB-short ORB[222.50,224.74] vol=2.0x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-12-12 10:00:00 | 222.36 | 222.65 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 214.30 | 215.24 | 0.00 | ORB-short ORB[214.35,216.74] vol=1.6x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-12-17 10:10:00 | 214.99 | 214.80 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:50:00 | 199.90 | 201.39 | 0.00 | ORB-short ORB[201.01,203.49] vol=1.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 200.49 | 201.34 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-12-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:30:00 | 198.49 | 199.56 | 0.00 | ORB-short ORB[198.54,201.00] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-12-27 10:35:00 | 199.05 | 199.55 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:45:00 | 199.00 | 197.38 | 0.00 | ORB-long ORB[195.32,198.14] vol=1.7x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:50:00 | 200.47 | 198.45 | 0.00 | T1 1.5R @ 200.47 |
| Target hit | 2024-12-30 10:30:00 | 200.62 | 200.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 204.70 | 207.43 | 0.00 | ORB-short ORB[206.80,209.89] vol=2.3x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-01-15 09:35:00 | 205.86 | 207.06 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:45:00 | 209.50 | 208.21 | 0.00 | ORB-long ORB[206.26,209.35] vol=1.8x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-01-20 09:55:00 | 208.64 | 208.29 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-01-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:45:00 | 207.07 | 208.17 | 0.00 | ORB-short ORB[207.10,209.60] vol=2.0x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:05:00 | 205.90 | 207.74 | 0.00 | T1 1.5R @ 205.90 |
| Target hit | 2025-01-21 15:15:00 | 206.05 | 205.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — BUY (started 2025-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:55:00 | 142.10 | 140.85 | 0.00 | ORB-long ORB[140.10,142.00] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 141.30 | 140.97 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 141.89 | 141.04 | 0.00 | ORB-long ORB[139.75,141.52] vol=2.2x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 141.46 | 141.07 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:30:00 | 172.00 | 2024-05-16 10:10:00 | 172.74 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-05-31 09:45:00 | 181.45 | 2024-05-31 09:50:00 | 180.12 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-05-31 09:45:00 | 181.45 | 2024-05-31 12:50:00 | 180.70 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2024-06-10 09:30:00 | 180.12 | 2024-06-10 09:35:00 | 178.82 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-06-10 09:30:00 | 180.12 | 2024-06-10 09:40:00 | 180.12 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 09:30:00 | 181.98 | 2024-06-13 09:45:00 | 181.19 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-06-13 09:30:00 | 181.98 | 2024-06-13 09:55:00 | 181.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-18 09:50:00 | 180.91 | 2024-06-18 09:55:00 | 180.26 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-26 09:40:00 | 188.76 | 2024-06-26 09:55:00 | 187.67 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-07-02 09:30:00 | 195.10 | 2024-07-02 09:45:00 | 194.08 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-02 09:30:00 | 195.10 | 2024-07-02 09:50:00 | 195.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 09:30:00 | 266.96 | 2024-07-26 09:35:00 | 265.67 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-08-16 09:45:00 | 241.50 | 2024-08-16 09:55:00 | 240.36 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-08-20 09:30:00 | 242.20 | 2024-08-20 10:35:00 | 241.04 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-08-20 09:30:00 | 242.20 | 2024-08-20 15:20:00 | 240.50 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2024-08-27 09:40:00 | 251.90 | 2024-08-27 09:55:00 | 250.84 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-08-27 09:40:00 | 251.90 | 2024-08-27 10:05:00 | 251.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-05 10:15:00 | 234.91 | 2024-09-05 10:25:00 | 235.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-25 10:05:00 | 224.16 | 2024-09-25 10:25:00 | 223.30 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-25 10:05:00 | 224.16 | 2024-09-25 11:05:00 | 224.16 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-26 09:30:00 | 219.42 | 2024-09-26 09:35:00 | 218.35 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-09-26 09:30:00 | 219.42 | 2024-09-26 10:05:00 | 219.42 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 11:05:00 | 229.49 | 2024-10-09 11:55:00 | 228.57 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-11-27 09:40:00 | 192.11 | 2024-11-27 09:45:00 | 193.29 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-11-27 09:40:00 | 192.11 | 2024-11-27 10:10:00 | 192.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 09:30:00 | 206.11 | 2024-12-02 09:50:00 | 204.97 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-12-04 09:40:00 | 210.89 | 2024-12-04 10:00:00 | 212.31 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-12-04 09:40:00 | 210.89 | 2024-12-04 10:20:00 | 210.89 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-09 09:30:00 | 224.84 | 2024-12-09 09:35:00 | 223.73 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-12-11 09:45:00 | 225.59 | 2024-12-11 09:50:00 | 224.45 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-12-12 09:40:00 | 221.42 | 2024-12-12 10:00:00 | 222.36 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-17 09:35:00 | 214.30 | 2024-12-17 10:10:00 | 214.99 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-26 10:50:00 | 199.90 | 2024-12-26 11:00:00 | 200.49 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-27 10:30:00 | 198.49 | 2024-12-27 10:35:00 | 199.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-30 09:45:00 | 199.00 | 2024-12-30 09:50:00 | 200.47 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-12-30 09:45:00 | 199.00 | 2024-12-30 10:30:00 | 200.62 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2025-01-15 09:30:00 | 204.70 | 2025-01-15 09:35:00 | 205.86 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-01-20 09:45:00 | 209.50 | 2025-01-20 09:55:00 | 208.64 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-21 09:45:00 | 207.07 | 2025-01-21 10:05:00 | 205.90 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-21 09:45:00 | 207.07 | 2025-01-21 15:15:00 | 206.05 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2025-03-11 10:55:00 | 142.10 | 2025-03-11 11:15:00 | 141.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2025-03-17 09:30:00 | 141.89 | 2025-03-17 09:35:00 | 141.46 | STOP_HIT | 1.00 | -0.30% |
