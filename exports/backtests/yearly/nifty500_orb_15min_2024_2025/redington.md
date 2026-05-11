# Redington Ltd. (REDINGTON)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 223.29
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
| ENTRY1 | 48 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 8 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 40
- **Target hits / Stop hits / Partials:** 8 / 40 / 23
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 10.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 13 | 44.8% | 4 | 16 | 9 | 0.14% | 4.0% |
| BUY @ 2nd Alert (retest1) | 29 | 13 | 44.8% | 4 | 16 | 9 | 0.14% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 18 | 42.9% | 4 | 24 | 14 | 0.15% | 6.3% |
| SELL @ 2nd Alert (retest1) | 42 | 18 | 42.9% | 4 | 24 | 14 | 0.15% | 6.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 71 | 31 | 43.7% | 8 | 40 | 23 | 0.14% | 10.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 206.05 | 207.51 | 0.00 | ORB-short ORB[206.15,208.90] vol=2.5x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-05-27 09:50:00 | 206.88 | 207.45 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:35:00 | 204.50 | 206.26 | 0.00 | ORB-short ORB[205.85,207.45] vol=1.7x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-05-28 10:50:00 | 205.28 | 206.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:05:00 | 218.00 | 217.18 | 0.00 | ORB-long ORB[214.55,217.39] vol=6.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2024-06-11 10:10:00 | 216.64 | 216.94 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:50:00 | 219.08 | 218.01 | 0.00 | ORB-long ORB[216.11,218.89] vol=1.8x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:05:00 | 220.37 | 218.49 | 0.00 | T1 1.5R @ 220.37 |
| Stop hit — per-position SL triggered | 2024-06-12 10:25:00 | 219.08 | 218.65 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:55:00 | 214.03 | 213.21 | 0.00 | ORB-long ORB[211.64,213.95] vol=3.0x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:40:00 | 215.20 | 213.57 | 0.00 | T1 1.5R @ 215.20 |
| Stop hit — per-position SL triggered | 2024-06-28 12:45:00 | 214.03 | 213.86 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 11:00:00 | 223.17 | 222.53 | 0.00 | ORB-long ORB[220.51,222.98] vol=2.4x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-07-04 11:10:00 | 222.62 | 222.56 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:25:00 | 207.79 | 208.63 | 0.00 | ORB-short ORB[208.50,209.70] vol=2.2x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 206.29 | 208.37 | 0.00 | T1 1.5R @ 206.29 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 207.79 | 208.30 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:45:00 | 209.21 | 209.69 | 0.00 | ORB-short ORB[209.33,211.00] vol=3.1x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:10:00 | 208.28 | 209.46 | 0.00 | T1 1.5R @ 208.28 |
| Target hit | 2024-07-11 13:00:00 | 208.22 | 207.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — SELL (started 2024-07-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 10:20:00 | 207.82 | 208.92 | 0.00 | ORB-short ORB[208.51,211.36] vol=2.0x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-07-16 15:20:00 | 208.00 | 208.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 206.27 | 207.47 | 0.00 | ORB-short ORB[207.30,208.80] vol=3.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 206.77 | 207.27 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:05:00 | 210.24 | 208.96 | 0.00 | ORB-long ORB[207.11,210.00] vol=4.3x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-07-25 11:10:00 | 209.56 | 209.16 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:50:00 | 211.70 | 210.46 | 0.00 | ORB-long ORB[209.10,210.46] vol=2.3x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:55:00 | 212.73 | 211.54 | 0.00 | T1 1.5R @ 212.73 |
| Target hit | 2024-07-26 14:25:00 | 213.10 | 213.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2024-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:35:00 | 216.08 | 215.21 | 0.00 | ORB-long ORB[213.90,215.80] vol=2.0x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:55:00 | 217.35 | 215.79 | 0.00 | T1 1.5R @ 217.35 |
| Stop hit — per-position SL triggered | 2024-07-30 10:05:00 | 216.08 | 215.93 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:30:00 | 213.71 | 214.28 | 0.00 | ORB-short ORB[213.94,216.86] vol=5.8x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-07-31 14:10:00 | 214.47 | 213.98 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 09:55:00 | 192.30 | 193.35 | 0.00 | ORB-short ORB[192.61,194.80] vol=1.8x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 10:55:00 | 191.19 | 192.85 | 0.00 | T1 1.5R @ 191.19 |
| Stop hit — per-position SL triggered | 2024-08-08 12:20:00 | 192.30 | 192.64 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 09:40:00 | 193.26 | 192.16 | 0.00 | ORB-long ORB[190.30,193.10] vol=1.6x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-08-12 09:45:00 | 192.32 | 192.17 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 09:45:00 | 197.15 | 195.90 | 0.00 | ORB-long ORB[194.51,196.99] vol=3.0x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 10:10:00 | 198.65 | 196.68 | 0.00 | T1 1.5R @ 198.65 |
| Target hit | 2024-08-14 15:20:00 | 200.10 | 199.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 10:15:00 | 202.17 | 201.34 | 0.00 | ORB-long ORB[200.40,202.00] vol=2.0x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 10:20:00 | 203.04 | 201.47 | 0.00 | T1 1.5R @ 203.04 |
| Stop hit — per-position SL triggered | 2024-08-16 12:05:00 | 202.17 | 202.11 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:40:00 | 204.71 | 203.55 | 0.00 | ORB-long ORB[201.45,203.80] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-08-19 10:05:00 | 203.85 | 203.70 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:55:00 | 206.88 | 205.45 | 0.00 | ORB-long ORB[204.29,206.70] vol=2.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:10:00 | 207.83 | 206.24 | 0.00 | T1 1.5R @ 207.83 |
| Stop hit — per-position SL triggered | 2024-08-20 12:15:00 | 206.88 | 206.78 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:40:00 | 208.03 | 207.11 | 0.00 | ORB-long ORB[205.96,207.88] vol=3.3x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-08-22 09:45:00 | 207.31 | 207.12 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:00:00 | 207.05 | 207.63 | 0.00 | ORB-short ORB[207.31,209.85] vol=1.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:05:00 | 205.87 | 207.22 | 0.00 | T1 1.5R @ 205.87 |
| Target hit | 2024-08-26 15:20:00 | 202.85 | 204.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:00:00 | 201.43 | 203.08 | 0.00 | ORB-short ORB[202.61,204.97] vol=2.3x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-08-29 11:15:00 | 202.08 | 202.88 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 09:45:00 | 199.83 | 201.12 | 0.00 | ORB-short ORB[201.08,203.65] vol=1.8x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-09-02 09:55:00 | 200.66 | 201.07 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 200.00 | 199.21 | 0.00 | ORB-long ORB[198.41,199.99] vol=2.2x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-09-03 09:45:00 | 199.58 | 199.28 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 11:15:00 | 191.01 | 191.99 | 0.00 | ORB-short ORB[192.55,193.83] vol=6.6x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-09-12 13:05:00 | 191.46 | 191.52 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:05:00 | 190.18 | 190.56 | 0.00 | ORB-short ORB[190.20,192.60] vol=2.1x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 11:50:00 | 189.56 | 190.24 | 0.00 | T1 1.5R @ 189.56 |
| Target hit | 2024-09-16 15:20:00 | 189.43 | 189.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 11:15:00 | 187.74 | 188.82 | 0.00 | ORB-short ORB[188.65,191.18] vol=3.8x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:25:00 | 187.04 | 188.67 | 0.00 | T1 1.5R @ 187.04 |
| Stop hit — per-position SL triggered | 2024-09-17 12:20:00 | 187.74 | 188.35 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:35:00 | 189.38 | 191.74 | 0.00 | ORB-short ORB[190.96,193.67] vol=2.3x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-09-19 10:55:00 | 190.13 | 191.55 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:25:00 | 188.60 | 189.42 | 0.00 | ORB-short ORB[189.00,190.83] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 10:45:00 | 187.81 | 189.20 | 0.00 | T1 1.5R @ 187.81 |
| Stop hit — per-position SL triggered | 2024-09-23 12:25:00 | 188.60 | 188.70 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:55:00 | 187.11 | 187.85 | 0.00 | ORB-short ORB[187.20,189.35] vol=1.8x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:10:00 | 186.33 | 187.55 | 0.00 | T1 1.5R @ 186.33 |
| Target hit | 2024-09-25 13:00:00 | 184.94 | 184.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — SELL (started 2024-10-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 10:30:00 | 185.47 | 186.22 | 0.00 | ORB-short ORB[185.72,187.69] vol=1.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 10:40:00 | 184.73 | 185.94 | 0.00 | T1 1.5R @ 184.73 |
| Stop hit — per-position SL triggered | 2024-10-09 11:20:00 | 185.47 | 185.52 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 11:15:00 | 184.27 | 183.58 | 0.00 | ORB-long ORB[182.42,183.95] vol=4.0x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-10-11 11:25:00 | 183.87 | 183.61 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:00:00 | 182.74 | 183.47 | 0.00 | ORB-short ORB[183.11,185.30] vol=1.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-10-14 10:05:00 | 183.23 | 183.45 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:00:00 | 180.50 | 181.14 | 0.00 | ORB-short ORB[180.81,182.39] vol=1.7x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:40:00 | 179.89 | 180.81 | 0.00 | T1 1.5R @ 179.89 |
| Stop hit — per-position SL triggered | 2024-10-16 12:00:00 | 180.50 | 180.75 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 174.26 | 175.10 | 0.00 | ORB-short ORB[174.30,176.90] vol=2.0x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:45:00 | 173.42 | 174.64 | 0.00 | T1 1.5R @ 173.42 |
| Stop hit — per-position SL triggered | 2024-10-21 10:05:00 | 174.26 | 174.26 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:35:00 | 166.70 | 168.03 | 0.00 | ORB-short ORB[167.42,169.30] vol=2.6x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:45:00 | 165.52 | 167.00 | 0.00 | T1 1.5R @ 165.52 |
| Stop hit — per-position SL triggered | 2024-10-29 13:10:00 | 166.70 | 165.92 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:30:00 | 192.34 | 193.22 | 0.00 | ORB-short ORB[192.57,195.00] vol=2.4x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-11-21 09:40:00 | 193.33 | 193.23 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:10:00 | 193.51 | 194.82 | 0.00 | ORB-short ORB[194.05,196.84] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-11-29 10:45:00 | 194.25 | 194.64 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 199.80 | 198.04 | 0.00 | ORB-long ORB[196.51,198.60] vol=2.1x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 09:35:00 | 201.06 | 199.75 | 0.00 | T1 1.5R @ 201.06 |
| Target hit | 2024-12-02 10:25:00 | 201.10 | 201.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2024-12-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:45:00 | 207.02 | 208.52 | 0.00 | ORB-short ORB[208.18,211.03] vol=1.9x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 207.73 | 208.49 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:15:00 | 211.01 | 213.06 | 0.00 | ORB-short ORB[211.31,214.35] vol=2.8x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-12-16 12:20:00 | 211.73 | 212.77 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:40:00 | 210.92 | 212.81 | 0.00 | ORB-short ORB[213.78,215.17] vol=1.7x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:35:00 | 210.03 | 212.16 | 0.00 | T1 1.5R @ 210.03 |
| Stop hit — per-position SL triggered | 2024-12-17 11:55:00 | 210.92 | 211.74 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:35:00 | 207.15 | 204.64 | 0.00 | ORB-long ORB[203.00,205.84] vol=1.7x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-12-19 10:00:00 | 206.13 | 204.99 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:35:00 | 203.93 | 202.07 | 0.00 | ORB-long ORB[200.21,202.06] vol=1.7x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-01-03 09:40:00 | 202.97 | 202.19 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-01-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 09:40:00 | 205.39 | 204.15 | 0.00 | ORB-long ORB[202.00,204.90] vol=1.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 09:50:00 | 207.10 | 205.15 | 0.00 | T1 1.5R @ 207.10 |
| Target hit | 2025-01-15 10:30:00 | 205.81 | 206.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — SELL (started 2025-04-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:35:00 | 226.60 | 228.24 | 0.00 | ORB-short ORB[227.46,229.85] vol=2.2x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:45:00 | 224.95 | 226.50 | 0.00 | T1 1.5R @ 224.95 |
| Stop hit — per-position SL triggered | 2025-04-03 10:10:00 | 226.60 | 225.75 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 255.22 | 254.02 | 0.00 | ORB-long ORB[252.00,255.09] vol=2.3x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-05-08 10:20:00 | 253.95 | 254.50 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-27 09:45:00 | 206.05 | 2024-05-27 09:50:00 | 206.88 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-28 10:35:00 | 204.50 | 2024-05-28 10:50:00 | 205.28 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-11 10:05:00 | 218.00 | 2024-06-11 10:10:00 | 216.64 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-06-12 09:50:00 | 219.08 | 2024-06-12 10:05:00 | 220.37 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-06-12 09:50:00 | 219.08 | 2024-06-12 10:25:00 | 219.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-28 10:55:00 | 214.03 | 2024-06-28 11:40:00 | 215.20 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-28 10:55:00 | 214.03 | 2024-06-28 12:45:00 | 214.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 11:00:00 | 223.17 | 2024-07-04 11:10:00 | 222.62 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-10 10:25:00 | 207.79 | 2024-07-10 10:35:00 | 206.29 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-07-10 10:25:00 | 207.79 | 2024-07-10 10:40:00 | 207.79 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 09:45:00 | 209.21 | 2024-07-11 10:10:00 | 208.28 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-11 09:45:00 | 209.21 | 2024-07-11 13:00:00 | 208.22 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2024-07-16 10:20:00 | 207.82 | 2024-07-16 15:20:00 | 208.00 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest1 | 2024-07-18 09:30:00 | 206.27 | 2024-07-18 09:40:00 | 206.77 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-25 11:05:00 | 210.24 | 2024-07-25 11:10:00 | 209.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-26 09:50:00 | 211.70 | 2024-07-26 09:55:00 | 212.73 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-07-26 09:50:00 | 211.70 | 2024-07-26 14:25:00 | 213.10 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-07-30 09:35:00 | 216.08 | 2024-07-30 09:55:00 | 217.35 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-07-30 09:35:00 | 216.08 | 2024-07-30 10:05:00 | 216.08 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-31 10:30:00 | 213.71 | 2024-07-31 14:10:00 | 214.47 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-08 09:55:00 | 192.30 | 2024-08-08 10:55:00 | 191.19 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-08-08 09:55:00 | 192.30 | 2024-08-08 12:20:00 | 192.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 09:40:00 | 193.26 | 2024-08-12 09:45:00 | 192.32 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-08-14 09:45:00 | 197.15 | 2024-08-14 10:10:00 | 198.65 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-08-14 09:45:00 | 197.15 | 2024-08-14 15:20:00 | 200.10 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2024-08-16 10:15:00 | 202.17 | 2024-08-16 10:20:00 | 203.04 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-08-16 10:15:00 | 202.17 | 2024-08-16 12:05:00 | 202.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-19 09:40:00 | 204.71 | 2024-08-19 10:05:00 | 203.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-20 10:55:00 | 206.88 | 2024-08-20 11:10:00 | 207.83 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-20 10:55:00 | 206.88 | 2024-08-20 12:15:00 | 206.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 09:40:00 | 208.03 | 2024-08-22 09:45:00 | 207.31 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-26 10:00:00 | 207.05 | 2024-08-26 11:05:00 | 205.87 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-08-26 10:00:00 | 207.05 | 2024-08-26 15:20:00 | 202.85 | TARGET_HIT | 0.50 | 2.03% |
| SELL | retest1 | 2024-08-29 11:00:00 | 201.43 | 2024-08-29 11:15:00 | 202.08 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-02 09:45:00 | 199.83 | 2024-09-02 09:55:00 | 200.66 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-09-03 09:35:00 | 200.00 | 2024-09-03 09:45:00 | 199.58 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-12 11:15:00 | 191.01 | 2024-09-12 13:05:00 | 191.46 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-16 10:05:00 | 190.18 | 2024-09-16 11:50:00 | 189.56 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-16 10:05:00 | 190.18 | 2024-09-16 15:20:00 | 189.43 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2024-09-17 11:15:00 | 187.74 | 2024-09-17 11:25:00 | 187.04 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-17 11:15:00 | 187.74 | 2024-09-17 12:20:00 | 187.74 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:35:00 | 189.38 | 2024-09-19 10:55:00 | 190.13 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-23 10:25:00 | 188.60 | 2024-09-23 10:45:00 | 187.81 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-23 10:25:00 | 188.60 | 2024-09-23 12:25:00 | 188.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 09:55:00 | 187.11 | 2024-09-25 10:10:00 | 186.33 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-09-25 09:55:00 | 187.11 | 2024-09-25 13:00:00 | 184.94 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2024-10-09 10:30:00 | 185.47 | 2024-10-09 10:40:00 | 184.73 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-09 10:30:00 | 185.47 | 2024-10-09 11:20:00 | 185.47 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 11:15:00 | 184.27 | 2024-10-11 11:25:00 | 183.87 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-14 10:00:00 | 182.74 | 2024-10-14 10:05:00 | 183.23 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-16 11:00:00 | 180.50 | 2024-10-16 11:40:00 | 179.89 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-10-16 11:00:00 | 180.50 | 2024-10-16 12:00:00 | 180.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:30:00 | 174.26 | 2024-10-21 09:45:00 | 173.42 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-10-21 09:30:00 | 174.26 | 2024-10-21 10:05:00 | 174.26 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 09:35:00 | 166.70 | 2024-10-29 09:45:00 | 165.52 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-10-29 09:35:00 | 166.70 | 2024-10-29 13:10:00 | 166.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-21 09:30:00 | 192.34 | 2024-11-21 09:40:00 | 193.33 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-11-29 10:10:00 | 193.51 | 2024-11-29 10:45:00 | 194.25 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-02 09:30:00 | 199.80 | 2024-12-02 09:35:00 | 201.06 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-12-02 09:30:00 | 199.80 | 2024-12-02 10:25:00 | 201.10 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-12-13 10:45:00 | 207.02 | 2024-12-13 10:50:00 | 207.73 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-16 11:15:00 | 211.01 | 2024-12-16 12:20:00 | 211.73 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-17 10:40:00 | 210.92 | 2024-12-17 11:35:00 | 210.03 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-17 10:40:00 | 210.92 | 2024-12-17 11:55:00 | 210.92 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-19 09:35:00 | 207.15 | 2024-12-19 10:00:00 | 206.13 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-03 09:35:00 | 203.93 | 2025-01-03 09:40:00 | 202.97 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-01-15 09:40:00 | 205.39 | 2025-01-15 09:50:00 | 207.10 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2025-01-15 09:40:00 | 205.39 | 2025-01-15 10:30:00 | 205.81 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2025-04-03 09:35:00 | 226.60 | 2025-04-03 09:45:00 | 224.95 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-04-03 09:35:00 | 226.60 | 2025-04-03 10:10:00 | 226.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 09:30:00 | 255.22 | 2025-05-08 10:20:00 | 253.95 | STOP_HIT | 1.00 | -0.50% |
