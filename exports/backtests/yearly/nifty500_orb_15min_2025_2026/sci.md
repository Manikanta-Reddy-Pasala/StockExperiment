# Shipping Corporation of India Ltd. (SCI)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 339.60
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
| ENTRY1 | 52 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 8 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 44
- **Target hits / Stop hits / Partials:** 8 / 44 / 20
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 8.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 17 | 43.6% | 6 | 22 | 11 | 0.15% | 5.7% |
| BUY @ 2nd Alert (retest1) | 39 | 17 | 43.6% | 6 | 22 | 11 | 0.15% | 5.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 11 | 33.3% | 2 | 22 | 9 | 0.07% | 2.4% |
| SELL @ 2nd Alert (retest1) | 33 | 11 | 33.3% | 2 | 22 | 9 | 0.07% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 72 | 28 | 38.9% | 8 | 44 | 20 | 0.11% | 8.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:45:00 | 206.50 | 203.68 | 0.00 | ORB-long ORB[201.68,204.00] vol=4.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-05-28 09:50:00 | 205.54 | 205.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 204.92 | 205.67 | 0.00 | ORB-short ORB[205.00,207.50] vol=1.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 10:25:00 | 203.54 | 205.19 | 0.00 | T1 1.5R @ 203.54 |
| Stop hit — per-position SL triggered | 2025-05-29 14:55:00 | 204.92 | 204.66 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:30:00 | 202.90 | 200.74 | 0.00 | ORB-long ORB[199.10,201.36] vol=2.5x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:35:00 | 204.10 | 204.20 | 0.00 | T1 1.5R @ 204.10 |
| Target hit | 2025-06-02 09:50:00 | 204.12 | 204.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 205.36 | 204.04 | 0.00 | ORB-long ORB[202.31,204.11] vol=4.0x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-06-03 09:50:00 | 204.36 | 204.42 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:20:00 | 215.49 | 213.13 | 0.00 | ORB-long ORB[209.76,212.00] vol=8.2x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-06-05 10:25:00 | 214.24 | 213.33 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 09:30:00 | 222.91 | 223.87 | 0.00 | ORB-short ORB[223.01,225.50] vol=1.7x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-06-26 09:45:00 | 223.65 | 223.73 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 224.91 | 223.61 | 0.00 | ORB-long ORB[222.16,224.14] vol=3.0x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-06-27 09:35:00 | 224.00 | 224.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:35:00 | 225.99 | 224.90 | 0.00 | ORB-long ORB[223.70,225.89] vol=1.8x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-07-01 09:40:00 | 225.22 | 224.95 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 224.12 | 225.88 | 0.00 | ORB-short ORB[224.80,227.41] vol=1.6x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-07-02 09:50:00 | 224.92 | 225.67 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 10:45:00 | 221.97 | 223.21 | 0.00 | ORB-short ORB[222.29,224.14] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-07-04 11:10:00 | 222.58 | 223.07 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:15:00 | 218.81 | 219.99 | 0.00 | ORB-short ORB[219.00,221.90] vol=2.1x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-07-16 11:50:00 | 219.28 | 219.93 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 219.46 | 221.13 | 0.00 | ORB-short ORB[220.60,222.75] vol=1.5x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:30:00 | 218.61 | 220.81 | 0.00 | T1 1.5R @ 218.61 |
| Stop hit — per-position SL triggered | 2025-07-18 10:50:00 | 219.46 | 220.58 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:35:00 | 216.89 | 217.55 | 0.00 | ORB-short ORB[217.00,218.54] vol=1.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-07-23 10:00:00 | 217.54 | 217.36 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:40:00 | 219.75 | 220.86 | 0.00 | ORB-short ORB[220.03,222.61] vol=1.7x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 12:05:00 | 218.80 | 220.49 | 0.00 | T1 1.5R @ 218.80 |
| Stop hit — per-position SL triggered | 2025-07-25 14:55:00 | 219.75 | 219.70 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:35:00 | 208.01 | 209.76 | 0.00 | ORB-short ORB[209.38,211.76] vol=1.8x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:45:00 | 207.05 | 209.02 | 0.00 | T1 1.5R @ 207.05 |
| Stop hit — per-position SL triggered | 2025-08-06 14:50:00 | 208.01 | 207.16 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:10:00 | 208.39 | 205.96 | 0.00 | ORB-long ORB[204.35,207.40] vol=1.7x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:30:00 | 209.88 | 206.82 | 0.00 | T1 1.5R @ 209.88 |
| Stop hit — per-position SL triggered | 2025-08-13 10:35:00 | 208.39 | 206.85 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:30:00 | 217.92 | 216.40 | 0.00 | ORB-long ORB[215.05,217.66] vol=2.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-08-22 10:50:00 | 217.21 | 216.62 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 210.66 | 212.27 | 0.00 | ORB-short ORB[211.70,214.62] vol=1.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-08-26 09:40:00 | 211.33 | 212.08 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 10:55:00 | 216.20 | 211.95 | 0.00 | ORB-long ORB[209.00,211.45] vol=3.3x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-08-28 11:00:00 | 215.36 | 212.45 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 214.90 | 213.73 | 0.00 | ORB-long ORB[212.29,214.50] vol=1.7x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:50:00 | 216.07 | 214.32 | 0.00 | T1 1.5R @ 216.07 |
| Stop hit — per-position SL triggered | 2025-09-01 10:05:00 | 214.90 | 214.52 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 11:10:00 | 220.04 | 219.02 | 0.00 | ORB-long ORB[217.75,219.75] vol=3.0x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-09-02 11:20:00 | 219.52 | 219.06 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:20:00 | 214.94 | 213.69 | 0.00 | ORB-long ORB[212.01,214.20] vol=2.7x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-09-12 10:30:00 | 214.19 | 213.72 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:40:00 | 216.77 | 216.01 | 0.00 | ORB-long ORB[214.61,216.68] vol=1.8x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 09:50:00 | 217.74 | 216.71 | 0.00 | T1 1.5R @ 217.74 |
| Stop hit — per-position SL triggered | 2025-09-15 10:05:00 | 216.77 | 216.77 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:50:00 | 216.05 | 215.22 | 0.00 | ORB-long ORB[213.49,215.60] vol=2.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-09-16 10:25:00 | 215.36 | 215.51 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:45:00 | 227.10 | 225.20 | 0.00 | ORB-long ORB[223.00,226.10] vol=2.3x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-09-29 10:10:00 | 225.99 | 225.57 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 09:50:00 | 222.33 | 220.29 | 0.00 | ORB-long ORB[218.08,220.50] vol=2.8x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 09:55:00 | 224.03 | 221.24 | 0.00 | T1 1.5R @ 224.03 |
| Stop hit — per-position SL triggered | 2025-09-30 10:05:00 | 222.33 | 221.53 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:20:00 | 224.64 | 223.07 | 0.00 | ORB-long ORB[221.01,223.40] vol=2.4x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-10-01 10:50:00 | 223.73 | 223.87 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 223.95 | 226.50 | 0.00 | ORB-short ORB[226.21,229.20] vol=1.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-10-06 13:15:00 | 224.53 | 225.73 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 222.34 | 223.45 | 0.00 | ORB-short ORB[223.45,225.59] vol=2.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-10-07 12:05:00 | 222.80 | 223.30 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:40:00 | 225.34 | 222.41 | 0.00 | ORB-long ORB[218.73,222.00] vol=4.9x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 09:45:00 | 227.27 | 224.91 | 0.00 | T1 1.5R @ 227.27 |
| Target hit | 2025-10-13 15:20:00 | 229.80 | 229.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 09:35:00 | 233.99 | 232.60 | 0.00 | ORB-long ORB[229.75,232.40] vol=4.7x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:50:00 | 235.51 | 234.22 | 0.00 | T1 1.5R @ 235.51 |
| Target hit | 2025-10-14 10:15:00 | 235.49 | 235.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 230.94 | 230.40 | 0.00 | ORB-long ORB[227.70,229.89] vol=6.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-10-17 09:45:00 | 230.08 | 230.56 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:40:00 | 263.12 | 265.66 | 0.00 | ORB-short ORB[265.51,268.00] vol=2.5x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-10-30 10:40:00 | 264.40 | 265.03 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-31 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 10:10:00 | 266.13 | 264.32 | 0.00 | ORB-long ORB[262.90,265.17] vol=2.1x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 265.15 | 264.45 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 09:30:00 | 259.20 | 260.40 | 0.00 | ORB-short ORB[259.25,262.35] vol=1.5x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-11-03 09:35:00 | 260.46 | 260.39 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 11:15:00 | 264.70 | 263.56 | 0.00 | ORB-long ORB[260.65,264.40] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-11-14 11:30:00 | 263.89 | 263.58 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 11:05:00 | 262.20 | 264.58 | 0.00 | ORB-short ORB[263.95,267.40] vol=2.9x ATR=1.12 |
| Target hit | 2025-11-17 15:20:00 | 261.15 | 263.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-11-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:20:00 | 251.75 | 249.81 | 0.00 | ORB-long ORB[249.10,251.50] vol=2.2x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:25:00 | 253.26 | 250.95 | 0.00 | T1 1.5R @ 253.26 |
| Target hit | 2025-11-20 11:10:00 | 253.20 | 253.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2025-11-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:00:00 | 242.95 | 245.51 | 0.00 | ORB-short ORB[245.55,248.05] vol=2.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 11:20:00 | 241.70 | 245.06 | 0.00 | T1 1.5R @ 241.70 |
| Stop hit — per-position SL triggered | 2025-11-21 11:45:00 | 242.95 | 244.87 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 09:50:00 | 237.30 | 239.60 | 0.00 | ORB-short ORB[238.35,241.35] vol=1.6x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 10:00:00 | 235.24 | 239.02 | 0.00 | T1 1.5R @ 235.24 |
| Stop hit — per-position SL triggered | 2025-11-25 10:05:00 | 237.30 | 238.88 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-12-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 09:30:00 | 222.96 | 223.71 | 0.00 | ORB-short ORB[223.07,225.39] vol=1.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 223.61 | 223.45 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:40:00 | 213.15 | 211.57 | 0.00 | ORB-long ORB[209.50,212.70] vol=2.0x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 12:25:00 | 214.51 | 212.26 | 0.00 | T1 1.5R @ 214.51 |
| Target hit | 2025-12-22 15:20:00 | 214.14 | 212.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 230.82 | 230.31 | 0.00 | ORB-long ORB[228.00,230.47] vol=2.7x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-12-31 11:25:00 | 230.14 | 230.32 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 09:55:00 | 229.20 | 230.26 | 0.00 | ORB-short ORB[229.49,231.08] vol=2.7x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-01-02 10:25:00 | 229.88 | 230.14 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 208.89 | 209.79 | 0.00 | ORB-short ORB[209.07,211.65] vol=1.8x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:35:00 | 207.68 | 209.43 | 0.00 | T1 1.5R @ 207.68 |
| Stop hit — per-position SL triggered | 2026-01-20 10:20:00 | 208.89 | 208.78 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:05:00 | 205.03 | 206.45 | 0.00 | ORB-short ORB[205.60,208.60] vol=1.6x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:35:00 | 204.21 | 206.34 | 0.00 | T1 1.5R @ 204.21 |
| Target hit | 2026-01-23 15:20:00 | 201.40 | 204.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 225.88 | 222.15 | 0.00 | ORB-long ORB[217.90,221.00] vol=5.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-01-30 09:35:00 | 224.32 | 223.46 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 09:30:00 | 229.66 | 228.19 | 0.00 | ORB-long ORB[226.07,229.30] vol=2.4x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 09:55:00 | 231.68 | 229.06 | 0.00 | T1 1.5R @ 231.68 |
| Target hit | 2026-02-01 11:15:00 | 231.76 | 231.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 264.20 | 261.68 | 0.00 | ORB-long ORB[259.26,262.75] vol=1.8x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:35:00 | 266.05 | 263.40 | 0.00 | T1 1.5R @ 266.05 |
| Stop hit — per-position SL triggered | 2026-02-20 10:30:00 | 264.20 | 265.10 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 233.10 | 235.29 | 0.00 | ORB-short ORB[233.20,236.45] vol=2.8x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 12:20:00 | 231.50 | 233.94 | 0.00 | T1 1.5R @ 231.50 |
| Stop hit — per-position SL triggered | 2026-03-17 13:35:00 | 233.10 | 233.11 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 252.00 | 254.57 | 0.00 | ORB-short ORB[253.55,257.16] vol=1.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 253.12 | 254.39 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 289.82 | 293.50 | 0.00 | ORB-short ORB[292.46,296.71] vol=1.6x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-04-24 09:40:00 | 291.23 | 293.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-28 09:45:00 | 206.50 | 2025-05-28 09:50:00 | 205.54 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-05-29 09:35:00 | 204.92 | 2025-05-29 10:25:00 | 203.54 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-05-29 09:35:00 | 204.92 | 2025-05-29 14:55:00 | 204.92 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-02 09:30:00 | 202.90 | 2025-06-02 09:35:00 | 204.10 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-06-02 09:30:00 | 202.90 | 2025-06-02 09:50:00 | 204.12 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-06-03 09:30:00 | 205.36 | 2025-06-03 09:50:00 | 204.36 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-06-05 10:20:00 | 215.49 | 2025-06-05 10:25:00 | 214.24 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-06-26 09:30:00 | 222.91 | 2025-06-26 09:45:00 | 223.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-27 09:30:00 | 224.91 | 2025-06-27 09:35:00 | 224.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-07-01 09:35:00 | 225.99 | 2025-07-01 09:40:00 | 225.22 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-02 09:40:00 | 224.12 | 2025-07-02 09:50:00 | 224.92 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-04 10:45:00 | 221.97 | 2025-07-04 11:10:00 | 222.58 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-16 11:15:00 | 218.81 | 2025-07-16 11:50:00 | 219.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-18 10:15:00 | 219.46 | 2025-07-18 10:30:00 | 218.61 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-18 10:15:00 | 219.46 | 2025-07-18 10:50:00 | 219.46 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:35:00 | 216.89 | 2025-07-23 10:00:00 | 217.54 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-25 10:40:00 | 219.75 | 2025-07-25 12:05:00 | 218.80 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-07-25 10:40:00 | 219.75 | 2025-07-25 14:55:00 | 219.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 09:35:00 | 208.01 | 2025-08-06 09:45:00 | 207.05 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-08-06 09:35:00 | 208.01 | 2025-08-06 14:50:00 | 208.01 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-13 10:10:00 | 208.39 | 2025-08-13 10:30:00 | 209.88 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-08-13 10:10:00 | 208.39 | 2025-08-13 10:35:00 | 208.39 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 10:30:00 | 217.92 | 2025-08-22 10:50:00 | 217.21 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-26 09:35:00 | 210.66 | 2025-08-26 09:40:00 | 211.33 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-28 10:55:00 | 216.20 | 2025-08-28 11:00:00 | 215.36 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-01 09:45:00 | 214.90 | 2025-09-01 09:50:00 | 216.07 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-09-01 09:45:00 | 214.90 | 2025-09-01 10:05:00 | 214.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 11:10:00 | 220.04 | 2025-09-02 11:20:00 | 219.52 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-12 10:20:00 | 214.94 | 2025-09-12 10:30:00 | 214.19 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-15 09:40:00 | 216.77 | 2025-09-15 09:50:00 | 217.74 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-15 09:40:00 | 216.77 | 2025-09-15 10:05:00 | 216.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-16 09:50:00 | 216.05 | 2025-09-16 10:25:00 | 215.36 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-29 09:45:00 | 227.10 | 2025-09-29 10:10:00 | 225.99 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-09-30 09:50:00 | 222.33 | 2025-09-30 09:55:00 | 224.03 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-09-30 09:50:00 | 222.33 | 2025-09-30 10:05:00 | 222.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-01 10:20:00 | 224.64 | 2025-10-01 10:50:00 | 223.73 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-06 11:10:00 | 223.95 | 2025-10-06 13:15:00 | 224.53 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-07 11:10:00 | 222.34 | 2025-10-07 12:05:00 | 222.80 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-13 09:40:00 | 225.34 | 2025-10-13 09:45:00 | 227.27 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2025-10-13 09:40:00 | 225.34 | 2025-10-13 15:20:00 | 229.80 | TARGET_HIT | 0.50 | 1.98% |
| BUY | retest1 | 2025-10-14 09:35:00 | 233.99 | 2025-10-14 09:50:00 | 235.51 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-10-14 09:35:00 | 233.99 | 2025-10-14 10:15:00 | 235.49 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2025-10-17 09:30:00 | 230.94 | 2025-10-17 09:45:00 | 230.08 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-30 09:40:00 | 263.12 | 2025-10-30 10:40:00 | 264.40 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-10-31 10:10:00 | 266.13 | 2025-10-31 10:15:00 | 265.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-03 09:30:00 | 259.20 | 2025-11-03 09:35:00 | 260.46 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-11-14 11:15:00 | 264.70 | 2025-11-14 11:30:00 | 263.89 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-17 11:05:00 | 262.20 | 2025-11-17 15:20:00 | 261.15 | TARGET_HIT | 1.00 | 0.40% |
| BUY | retest1 | 2025-11-20 10:20:00 | 251.75 | 2025-11-20 10:25:00 | 253.26 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-11-20 10:20:00 | 251.75 | 2025-11-20 11:10:00 | 253.20 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-11-21 11:00:00 | 242.95 | 2025-11-21 11:20:00 | 241.70 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-11-21 11:00:00 | 242.95 | 2025-11-21 11:45:00 | 242.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-25 09:50:00 | 237.30 | 2025-11-25 10:00:00 | 235.24 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2025-11-25 09:50:00 | 237.30 | 2025-11-25 10:05:00 | 237.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-15 09:30:00 | 222.96 | 2025-12-15 10:15:00 | 223.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-22 10:40:00 | 213.15 | 2025-12-22 12:25:00 | 214.51 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-12-22 10:40:00 | 213.15 | 2025-12-22 15:20:00 | 214.14 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2025-12-31 11:00:00 | 230.82 | 2025-12-31 11:25:00 | 230.14 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-02 09:55:00 | 229.20 | 2026-01-02 10:25:00 | 229.88 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-20 09:30:00 | 208.89 | 2026-01-20 09:35:00 | 207.68 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-01-20 09:30:00 | 208.89 | 2026-01-20 10:20:00 | 208.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-23 11:05:00 | 205.03 | 2026-01-23 11:35:00 | 204.21 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-23 11:05:00 | 205.03 | 2026-01-23 15:20:00 | 201.40 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2026-01-30 09:30:00 | 225.88 | 2026-01-30 09:35:00 | 224.32 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2026-02-01 09:30:00 | 229.66 | 2026-02-01 09:55:00 | 231.68 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2026-02-01 09:30:00 | 229.66 | 2026-02-01 11:15:00 | 231.76 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2026-02-20 09:30:00 | 264.20 | 2026-02-20 09:35:00 | 266.05 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-02-20 09:30:00 | 264.20 | 2026-02-20 10:30:00 | 264.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 11:15:00 | 233.10 | 2026-03-17 12:20:00 | 231.50 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-17 11:15:00 | 233.10 | 2026-03-17 13:35:00 | 233.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:45:00 | 252.00 | 2026-04-16 09:50:00 | 253.12 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-24 09:35:00 | 289.82 | 2026-04-24 09:40:00 | 291.23 | STOP_HIT | 1.00 | -0.48% |
