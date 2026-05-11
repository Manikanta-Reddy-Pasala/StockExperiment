# Aditya Birla Capital Ltd. (ABCAPITAL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-10-07 15:25:00 (7596 bars)
- **Last close:** 225.00
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
| ENTRY1 | 41 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 7 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 34
- **Target hits / Stop hits / Partials:** 7 / 34 / 13
- **Avg / median % per leg:** 0.02% / -0.26%
- **Sum % (uncompounded):** 1.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 12 | 37.5% | 5 | 20 | 7 | 0.03% | 1.0% |
| BUY @ 2nd Alert (retest1) | 32 | 12 | 37.5% | 5 | 20 | 7 | 0.03% | 1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 8 | 36.4% | 2 | 14 | 6 | 0.01% | 0.2% |
| SELL @ 2nd Alert (retest1) | 22 | 8 | 36.4% | 2 | 14 | 6 | 0.01% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 54 | 20 | 37.0% | 7 | 34 | 13 | 0.02% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:35:00 | 225.50 | 224.25 | 0.00 | ORB-long ORB[222.00,224.50] vol=4.3x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 09:40:00 | 226.74 | 225.67 | 0.00 | T1 1.5R @ 226.74 |
| Target hit | 2024-05-17 10:15:00 | 226.85 | 227.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 222.80 | 225.55 | 0.00 | ORB-short ORB[225.30,227.35] vol=2.7x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 223.85 | 224.72 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:50:00 | 223.85 | 224.83 | 0.00 | ORB-short ORB[224.00,226.00] vol=1.9x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-05-23 11:00:00 | 224.65 | 224.41 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:20:00 | 230.25 | 228.82 | 0.00 | ORB-long ORB[227.10,229.70] vol=4.3x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-05-24 10:30:00 | 229.33 | 229.03 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:45:00 | 229.50 | 227.83 | 0.00 | ORB-long ORB[226.75,228.40] vol=1.7x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:55:00 | 231.08 | 229.33 | 0.00 | T1 1.5R @ 231.08 |
| Stop hit — per-position SL triggered | 2024-05-28 10:05:00 | 229.50 | 229.49 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 223.30 | 224.56 | 0.00 | ORB-short ORB[224.00,226.70] vol=3.3x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-05-30 09:45:00 | 224.13 | 224.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:40:00 | 234.80 | 232.72 | 0.00 | ORB-long ORB[231.19,232.99] vol=2.9x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:30:00 | 236.30 | 234.21 | 0.00 | T1 1.5R @ 236.30 |
| Target hit | 2024-06-11 14:15:00 | 235.54 | 235.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2024-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 11:05:00 | 233.68 | 234.94 | 0.00 | ORB-short ORB[233.90,236.00] vol=1.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 234.30 | 234.74 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 237.67 | 236.50 | 0.00 | ORB-long ORB[234.61,237.51] vol=4.4x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-06-13 09:40:00 | 236.80 | 236.58 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:05:00 | 242.25 | 240.83 | 0.00 | ORB-long ORB[238.69,241.59] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-06-14 11:25:00 | 241.28 | 241.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:50:00 | 244.75 | 244.00 | 0.00 | ORB-long ORB[243.00,244.70] vol=2.0x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-06-21 10:05:00 | 243.85 | 244.06 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 235.97 | 238.86 | 0.00 | ORB-short ORB[238.50,241.00] vol=2.2x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-06-25 11:45:00 | 236.59 | 238.44 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:50:00 | 238.00 | 236.67 | 0.00 | ORB-long ORB[234.31,237.47] vol=1.9x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-06-27 10:05:00 | 236.93 | 236.79 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:30:00 | 238.40 | 236.58 | 0.00 | ORB-long ORB[235.20,237.58] vol=1.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 10:10:00 | 239.62 | 237.50 | 0.00 | T1 1.5R @ 239.62 |
| Target hit | 2024-06-28 11:55:00 | 240.39 | 240.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2024-07-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:45:00 | 237.93 | 240.10 | 0.00 | ORB-short ORB[240.20,242.41] vol=1.9x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-07-02 10:00:00 | 238.68 | 239.79 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:45:00 | 239.60 | 238.79 | 0.00 | ORB-long ORB[237.70,239.19] vol=4.0x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-07-04 10:00:00 | 238.93 | 239.00 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 232.06 | 234.17 | 0.00 | ORB-short ORB[234.60,237.04] vol=1.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-07-08 11:30:00 | 232.72 | 233.98 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 229.08 | 229.96 | 0.00 | ORB-short ORB[229.50,231.15] vol=2.1x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:00:00 | 228.19 | 229.52 | 0.00 | T1 1.5R @ 228.19 |
| Target hit | 2024-07-10 11:55:00 | 227.58 | 227.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2024-07-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:30:00 | 227.52 | 229.50 | 0.00 | ORB-short ORB[229.10,231.28] vol=1.7x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-07-11 10:35:00 | 228.18 | 229.40 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 223.29 | 224.61 | 0.00 | ORB-short ORB[224.00,225.65] vol=2.0x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 224.05 | 224.08 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 215.17 | 215.87 | 0.00 | ORB-short ORB[215.80,218.85] vol=2.3x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:35:00 | 214.19 | 215.81 | 0.00 | T1 1.5R @ 214.19 |
| Stop hit — per-position SL triggered | 2024-07-23 11:40:00 | 215.17 | 215.77 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:10:00 | 220.95 | 219.37 | 0.00 | ORB-long ORB[217.30,219.90] vol=2.4x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-07-26 10:20:00 | 220.22 | 219.78 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:35:00 | 225.05 | 224.11 | 0.00 | ORB-long ORB[222.71,224.20] vol=4.1x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-07-31 09:55:00 | 224.38 | 224.64 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:50:00 | 228.58 | 227.28 | 0.00 | ORB-long ORB[226.28,228.40] vol=3.1x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-08-01 11:00:00 | 227.59 | 227.37 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:15:00 | 212.68 | 211.18 | 0.00 | ORB-long ORB[208.60,210.90] vol=1.9x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-08-08 11:50:00 | 211.95 | 211.36 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 09:55:00 | 209.02 | 210.30 | 0.00 | ORB-short ORB[209.60,211.79] vol=1.8x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-08-12 10:00:00 | 209.86 | 210.21 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:40:00 | 209.78 | 208.67 | 0.00 | ORB-long ORB[207.15,209.49] vol=1.5x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-08-14 10:45:00 | 208.79 | 208.69 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:15:00 | 213.99 | 215.67 | 0.00 | ORB-short ORB[215.20,217.70] vol=1.7x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 11:35:00 | 212.85 | 215.42 | 0.00 | T1 1.5R @ 212.85 |
| Stop hit — per-position SL triggered | 2024-08-19 12:20:00 | 213.99 | 215.10 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:40:00 | 215.35 | 214.21 | 0.00 | ORB-long ORB[213.25,214.90] vol=1.7x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:55:00 | 216.22 | 214.95 | 0.00 | T1 1.5R @ 216.22 |
| Target hit | 2024-08-20 15:20:00 | 218.20 | 217.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2024-08-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:50:00 | 220.51 | 219.78 | 0.00 | ORB-long ORB[218.04,219.95] vol=3.4x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 09:55:00 | 221.40 | 220.05 | 0.00 | T1 1.5R @ 221.40 |
| Target hit | 2024-08-21 14:30:00 | 221.60 | 221.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2024-08-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:10:00 | 224.18 | 223.28 | 0.00 | ORB-long ORB[222.19,223.79] vol=2.0x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-08-22 10:15:00 | 223.61 | 223.30 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:35:00 | 221.90 | 221.02 | 0.00 | ORB-long ORB[219.60,221.43] vol=1.7x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-08-26 09:55:00 | 221.27 | 221.21 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:35:00 | 221.19 | 222.38 | 0.00 | ORB-short ORB[221.54,222.90] vol=2.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:30:00 | 220.24 | 221.83 | 0.00 | T1 1.5R @ 220.24 |
| Target hit | 2024-08-29 15:00:00 | 220.80 | 220.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2024-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 09:45:00 | 222.04 | 221.20 | 0.00 | ORB-long ORB[220.66,221.74] vol=2.0x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-09-02 09:55:00 | 221.46 | 221.24 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 223.80 | 223.04 | 0.00 | ORB-long ORB[221.90,223.77] vol=1.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-09-05 09:35:00 | 223.22 | 223.08 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:00:00 | 220.03 | 221.84 | 0.00 | ORB-short ORB[223.08,224.35] vol=3.3x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 219.05 | 221.28 | 0.00 | T1 1.5R @ 219.05 |
| Stop hit — per-position SL triggered | 2024-09-06 10:10:00 | 220.03 | 221.07 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:35:00 | 224.77 | 223.31 | 0.00 | ORB-long ORB[220.87,222.62] vol=1.5x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-09-13 10:40:00 | 224.08 | 223.35 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:55:00 | 224.38 | 225.27 | 0.00 | ORB-short ORB[224.73,227.70] vol=1.7x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:25:00 | 223.39 | 224.86 | 0.00 | T1 1.5R @ 223.39 |
| Stop hit — per-position SL triggered | 2024-09-17 12:25:00 | 224.38 | 224.59 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:45:00 | 234.00 | 232.39 | 0.00 | ORB-long ORB[230.50,232.02] vol=2.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-09-25 10:00:00 | 233.22 | 232.85 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:50:00 | 238.46 | 237.65 | 0.00 | ORB-long ORB[235.20,238.20] vol=2.1x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-09-27 10:25:00 | 237.48 | 237.80 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 240.24 | 239.19 | 0.00 | ORB-long ORB[236.40,239.40] vol=4.0x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:55:00 | 241.51 | 240.15 | 0.00 | T1 1.5R @ 241.51 |
| Stop hit — per-position SL triggered | 2024-10-01 10:00:00 | 240.24 | 240.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-17 09:35:00 | 225.50 | 2024-05-17 09:40:00 | 226.74 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-05-17 09:35:00 | 225.50 | 2024-05-17 10:15:00 | 226.85 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-22 09:40:00 | 222.80 | 2024-05-22 09:55:00 | 223.85 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-05-23 09:50:00 | 223.85 | 2024-05-23 11:00:00 | 224.65 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-24 10:20:00 | 230.25 | 2024-05-24 10:30:00 | 229.33 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-05-28 09:45:00 | 229.50 | 2024-05-28 09:55:00 | 231.08 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-05-28 09:45:00 | 229.50 | 2024-05-28 10:05:00 | 229.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 09:35:00 | 223.30 | 2024-05-30 09:45:00 | 224.13 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-11 09:40:00 | 234.80 | 2024-06-11 10:30:00 | 236.30 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-06-11 09:40:00 | 234.80 | 2024-06-11 14:15:00 | 235.54 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2024-06-12 11:05:00 | 233.68 | 2024-06-12 11:15:00 | 234.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-13 09:35:00 | 237.67 | 2024-06-13 09:40:00 | 236.80 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-14 10:05:00 | 242.25 | 2024-06-14 11:25:00 | 241.28 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-21 09:50:00 | 244.75 | 2024-06-21 10:05:00 | 243.85 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-06-25 11:15:00 | 235.97 | 2024-06-25 11:45:00 | 236.59 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-27 09:50:00 | 238.00 | 2024-06-27 10:05:00 | 236.93 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-06-28 09:30:00 | 238.40 | 2024-06-28 10:10:00 | 239.62 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-06-28 09:30:00 | 238.40 | 2024-06-28 11:55:00 | 240.39 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2024-07-02 09:45:00 | 237.93 | 2024-07-02 10:00:00 | 238.68 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-04 09:45:00 | 239.60 | 2024-07-04 10:00:00 | 238.93 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-08 11:10:00 | 232.06 | 2024-07-08 11:30:00 | 232.72 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-10 09:45:00 | 229.08 | 2024-07-10 10:00:00 | 228.19 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-10 09:45:00 | 229.08 | 2024-07-10 11:55:00 | 227.58 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-07-11 10:30:00 | 227.52 | 2024-07-11 10:35:00 | 228.18 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-18 09:30:00 | 223.29 | 2024-07-18 09:40:00 | 224.05 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-23 11:15:00 | 215.17 | 2024-07-23 11:35:00 | 214.19 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-23 11:15:00 | 215.17 | 2024-07-23 11:40:00 | 215.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:10:00 | 220.95 | 2024-07-26 10:20:00 | 220.22 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-31 09:35:00 | 225.05 | 2024-07-31 09:55:00 | 224.38 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-01 10:50:00 | 228.58 | 2024-08-01 11:00:00 | 227.59 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-08-08 11:15:00 | 212.68 | 2024-08-08 11:50:00 | 211.95 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-12 09:55:00 | 209.02 | 2024-08-12 10:00:00 | 209.86 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-14 10:40:00 | 209.78 | 2024-08-14 10:45:00 | 208.79 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-08-19 11:15:00 | 213.99 | 2024-08-19 11:35:00 | 212.85 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-08-19 11:15:00 | 213.99 | 2024-08-19 12:20:00 | 213.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 09:40:00 | 215.35 | 2024-08-20 09:55:00 | 216.22 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-20 09:40:00 | 215.35 | 2024-08-20 15:20:00 | 218.20 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2024-08-21 09:50:00 | 220.51 | 2024-08-21 09:55:00 | 221.40 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-21 09:50:00 | 220.51 | 2024-08-21 14:30:00 | 221.60 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-08-22 10:10:00 | 224.18 | 2024-08-22 10:15:00 | 223.61 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-26 09:35:00 | 221.90 | 2024-08-26 09:55:00 | 221.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-29 10:35:00 | 221.19 | 2024-08-29 11:30:00 | 220.24 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-29 10:35:00 | 221.19 | 2024-08-29 15:00:00 | 220.80 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2024-09-02 09:45:00 | 222.04 | 2024-09-02 09:55:00 | 221.46 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-05 09:30:00 | 223.80 | 2024-09-05 09:35:00 | 223.22 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-06 10:00:00 | 220.03 | 2024-09-06 10:05:00 | 219.05 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-06 10:00:00 | 220.03 | 2024-09-06 10:10:00 | 220.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 10:35:00 | 224.77 | 2024-09-13 10:40:00 | 224.08 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-17 10:55:00 | 224.38 | 2024-09-17 11:25:00 | 223.39 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-17 10:55:00 | 224.38 | 2024-09-17 12:25:00 | 224.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-25 09:45:00 | 234.00 | 2024-09-25 10:00:00 | 233.22 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-27 09:50:00 | 238.46 | 2024-09-27 10:25:00 | 237.48 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-10-01 09:30:00 | 240.24 | 2024-10-01 09:55:00 | 241.51 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-10-01 09:30:00 | 240.24 | 2024-10-01 10:00:00 | 240.24 | STOP_HIT | 0.50 | 0.00% |
