# Manappuram Finance Ltd. (MANAPPURAM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-12-31 15:25:00 (30496 bars)
- **Last close:** 309.00
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
| ENTRY1 | 61 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 17 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 44
- **Target hits / Stop hits / Partials:** 17 / 44 / 34
- **Avg / median % per leg:** 0.33% / 0.31%
- **Sum % (uncompounded):** 31.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 30 | 61.2% | 12 | 19 | 18 | 0.50% | 24.4% |
| BUY @ 2nd Alert (retest1) | 49 | 30 | 61.2% | 12 | 19 | 18 | 0.50% | 24.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 21 | 45.7% | 5 | 25 | 16 | 0.15% | 6.8% |
| SELL @ 2nd Alert (retest1) | 46 | 21 | 45.7% | 5 | 25 | 16 | 0.15% | 6.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 95 | 51 | 53.7% | 17 | 44 | 34 | 0.33% | 31.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 178.65 | 177.68 | 0.00 | ORB-long ORB[176.50,178.40] vol=2.6x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 09:45:00 | 179.71 | 178.12 | 0.00 | T1 1.5R @ 179.71 |
| Target hit | 2024-05-15 15:10:00 | 179.45 | 179.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2024-05-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:40:00 | 170.45 | 170.97 | 0.00 | ORB-short ORB[170.65,172.35] vol=1.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:05:00 | 169.60 | 170.64 | 0.00 | T1 1.5R @ 169.60 |
| Target hit | 2024-05-30 15:20:00 | 168.15 | 168.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:20:00 | 168.10 | 169.28 | 0.00 | ORB-short ORB[168.75,171.10] vol=1.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-05-31 10:45:00 | 168.87 | 169.14 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:00:00 | 180.35 | 178.35 | 0.00 | ORB-long ORB[176.95,177.95] vol=3.4x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:05:00 | 181.39 | 179.21 | 0.00 | T1 1.5R @ 181.39 |
| Target hit | 2024-06-11 12:45:00 | 181.84 | 181.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2024-06-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:55:00 | 184.15 | 182.45 | 0.00 | ORB-long ORB[181.16,182.99] vol=1.5x ATR=0.72 |
| Target hit | 2024-06-12 15:20:00 | 184.58 | 183.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-06-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:40:00 | 188.00 | 185.63 | 0.00 | ORB-long ORB[184.70,186.40] vol=2.7x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:55:00 | 189.03 | 186.06 | 0.00 | T1 1.5R @ 189.03 |
| Target hit | 2024-06-18 15:20:00 | 191.49 | 189.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 199.38 | 197.82 | 0.00 | ORB-long ORB[195.84,197.47] vol=5.1x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:35:00 | 200.66 | 199.39 | 0.00 | T1 1.5R @ 200.66 |
| Target hit | 2024-06-27 15:20:00 | 211.12 | 209.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 212.21 | 210.68 | 0.00 | ORB-long ORB[209.60,211.60] vol=2.1x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-07-01 09:50:00 | 211.27 | 211.22 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:45:00 | 208.10 | 209.52 | 0.00 | ORB-short ORB[209.50,212.20] vol=1.6x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:40:00 | 206.99 | 209.02 | 0.00 | T1 1.5R @ 206.99 |
| Stop hit — per-position SL triggered | 2024-07-02 11:05:00 | 208.10 | 208.84 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:50:00 | 209.30 | 207.78 | 0.00 | ORB-long ORB[206.08,208.65] vol=1.8x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:05:00 | 210.06 | 208.19 | 0.00 | T1 1.5R @ 210.06 |
| Stop hit — per-position SL triggered | 2024-07-03 11:10:00 | 209.30 | 208.21 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:10:00 | 207.50 | 208.09 | 0.00 | ORB-short ORB[207.91,209.68] vol=2.1x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:05:00 | 206.56 | 207.92 | 0.00 | T1 1.5R @ 206.56 |
| Stop hit — per-position SL triggered | 2024-07-08 11:10:00 | 207.50 | 207.89 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:35:00 | 206.30 | 207.14 | 0.00 | ORB-short ORB[206.65,208.00] vol=3.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-07-09 12:00:00 | 206.87 | 206.40 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:25:00 | 214.70 | 213.40 | 0.00 | ORB-long ORB[213.05,214.40] vol=1.9x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:50:00 | 215.88 | 213.87 | 0.00 | T1 1.5R @ 215.88 |
| Stop hit — per-position SL triggered | 2024-07-12 11:20:00 | 214.70 | 214.26 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:50:00 | 207.95 | 206.56 | 0.00 | ORB-long ORB[204.81,207.68] vol=3.0x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-07-26 10:00:00 | 207.18 | 206.73 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:35:00 | 216.81 | 215.91 | 0.00 | ORB-long ORB[214.40,216.20] vol=4.1x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-08-01 09:40:00 | 216.17 | 216.00 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 11:05:00 | 203.58 | 202.25 | 0.00 | ORB-long ORB[200.93,202.76] vol=3.9x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 11:40:00 | 204.68 | 202.79 | 0.00 | T1 1.5R @ 204.68 |
| Target hit | 2024-08-09 15:20:00 | 204.97 | 204.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-08-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 09:35:00 | 204.50 | 204.02 | 0.00 | ORB-long ORB[201.84,204.24] vol=12.9x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:40:00 | 205.81 | 204.09 | 0.00 | T1 1.5R @ 205.81 |
| Target hit | 2024-08-12 15:20:00 | 210.17 | 206.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 203.15 | 201.74 | 0.00 | ORB-long ORB[200.70,202.99] vol=2.7x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:00:00 | 204.10 | 202.36 | 0.00 | T1 1.5R @ 204.10 |
| Target hit | 2024-08-20 15:20:00 | 208.25 | 206.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-08-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:35:00 | 212.44 | 213.76 | 0.00 | ORB-short ORB[212.61,215.60] vol=2.3x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-08-27 09:40:00 | 213.10 | 213.68 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:45:00 | 213.45 | 214.99 | 0.00 | ORB-short ORB[213.52,215.50] vol=2.0x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:05:00 | 212.43 | 214.72 | 0.00 | T1 1.5R @ 212.43 |
| Stop hit — per-position SL triggered | 2024-08-29 11:10:00 | 213.45 | 214.67 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:15:00 | 215.24 | 214.19 | 0.00 | ORB-long ORB[212.32,214.33] vol=1.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-09-03 10:20:00 | 214.56 | 214.23 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 11:10:00 | 211.39 | 210.44 | 0.00 | ORB-long ORB[209.45,210.99] vol=1.8x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:20:00 | 212.18 | 210.57 | 0.00 | T1 1.5R @ 212.18 |
| Stop hit — per-position SL triggered | 2024-09-05 11:25:00 | 211.39 | 210.59 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:50:00 | 207.75 | 209.21 | 0.00 | ORB-short ORB[209.30,210.57] vol=2.7x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 206.81 | 208.50 | 0.00 | T1 1.5R @ 206.81 |
| Target hit | 2024-09-06 11:45:00 | 207.36 | 207.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — SELL (started 2024-09-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 09:55:00 | 204.06 | 204.85 | 0.00 | ORB-short ORB[204.25,206.70] vol=1.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:15:00 | 203.12 | 204.42 | 0.00 | T1 1.5R @ 203.12 |
| Stop hit — per-position SL triggered | 2024-09-10 12:30:00 | 204.06 | 204.19 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:15:00 | 204.67 | 209.23 | 0.00 | ORB-short ORB[210.75,213.25] vol=1.5x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:25:00 | 203.27 | 208.85 | 0.00 | T1 1.5R @ 203.27 |
| Stop hit — per-position SL triggered | 2024-09-19 11:50:00 | 204.67 | 207.85 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:35:00 | 208.03 | 208.67 | 0.00 | ORB-short ORB[208.04,209.70] vol=2.0x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-09-24 09:40:00 | 208.58 | 208.66 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:45:00 | 206.23 | 204.31 | 0.00 | ORB-long ORB[202.51,204.29] vol=1.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-09-27 10:50:00 | 205.65 | 204.37 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 09:30:00 | 201.58 | 202.45 | 0.00 | ORB-short ORB[202.22,204.20] vol=3.0x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:40:00 | 200.72 | 202.19 | 0.00 | T1 1.5R @ 200.72 |
| Stop hit — per-position SL triggered | 2024-09-30 09:55:00 | 201.58 | 201.99 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:50:00 | 193.27 | 194.72 | 0.00 | ORB-short ORB[194.51,196.45] vol=2.9x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 10:00:00 | 192.25 | 194.16 | 0.00 | T1 1.5R @ 192.25 |
| Stop hit — per-position SL triggered | 2024-10-03 11:50:00 | 193.27 | 192.66 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 11:10:00 | 181.81 | 184.46 | 0.00 | ORB-short ORB[184.50,186.95] vol=1.9x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-10-15 11:25:00 | 182.34 | 184.20 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:05:00 | 181.80 | 182.82 | 0.00 | ORB-short ORB[182.70,183.59] vol=1.8x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:15:00 | 181.00 | 182.35 | 0.00 | T1 1.5R @ 181.00 |
| Target hit | 2024-10-16 13:10:00 | 180.61 | 180.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 156.40 | 155.56 | 0.00 | ORB-long ORB[154.44,156.26] vol=1.8x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 09:40:00 | 157.48 | 156.13 | 0.00 | T1 1.5R @ 157.48 |
| Target hit | 2024-11-19 11:55:00 | 157.63 | 157.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — BUY (started 2024-11-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:30:00 | 155.14 | 154.48 | 0.00 | ORB-long ORB[153.43,155.00] vol=2.1x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-11-25 09:40:00 | 154.46 | 154.55 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 166.86 | 168.27 | 0.00 | ORB-short ORB[167.80,169.40] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-12-06 10:10:00 | 167.64 | 167.62 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:10:00 | 173.80 | 175.97 | 0.00 | ORB-short ORB[175.80,178.15] vol=1.6x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:25:00 | 172.49 | 175.11 | 0.00 | T1 1.5R @ 172.49 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 173.80 | 174.71 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-01-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 09:30:00 | 188.48 | 189.71 | 0.00 | ORB-short ORB[188.55,190.95] vol=1.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-01-01 11:10:00 | 189.27 | 189.36 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-01-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 10:50:00 | 180.62 | 177.95 | 0.00 | ORB-long ORB[178.00,180.14] vol=1.7x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:55:00 | 182.23 | 178.40 | 0.00 | T1 1.5R @ 182.23 |
| Stop hit — per-position SL triggered | 2025-01-10 14:15:00 | 180.62 | 180.48 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-01-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:10:00 | 185.70 | 184.69 | 0.00 | ORB-long ORB[182.91,184.90] vol=1.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-01-17 10:45:00 | 185.08 | 185.02 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:15:00 | 190.69 | 192.92 | 0.00 | ORB-short ORB[192.29,194.95] vol=1.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:20:00 | 189.36 | 191.91 | 0.00 | T1 1.5R @ 189.36 |
| Stop hit — per-position SL triggered | 2025-01-27 11:50:00 | 190.69 | 190.34 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:55:00 | 199.35 | 200.36 | 0.00 | ORB-short ORB[200.05,202.30] vol=1.9x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-01-30 11:00:00 | 200.04 | 200.33 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-01-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 09:30:00 | 196.46 | 198.57 | 0.00 | ORB-short ORB[197.80,200.38] vol=1.6x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-01-31 09:45:00 | 197.50 | 197.75 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-02-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:45:00 | 198.30 | 196.23 | 0.00 | ORB-long ORB[194.48,197.00] vol=1.7x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-02-01 10:05:00 | 197.46 | 196.73 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:55:00 | 200.41 | 199.73 | 0.00 | ORB-long ORB[198.00,200.33] vol=1.6x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 11:40:00 | 201.48 | 199.92 | 0.00 | T1 1.5R @ 201.48 |
| Stop hit — per-position SL triggered | 2025-03-05 12:55:00 | 200.41 | 200.23 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-03-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:30:00 | 205.05 | 203.89 | 0.00 | ORB-long ORB[201.86,203.98] vol=5.1x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 10:35:00 | 206.12 | 204.91 | 0.00 | T1 1.5R @ 206.12 |
| Target hit | 2025-03-07 11:50:00 | 206.28 | 206.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — SELL (started 2025-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:10:00 | 198.55 | 200.84 | 0.00 | ORB-short ORB[200.46,202.64] vol=1.8x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 199.40 | 200.63 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:40:00 | 206.00 | 205.29 | 0.00 | ORB-long ORB[204.11,205.89] vol=1.9x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 11:50:00 | 207.07 | 205.89 | 0.00 | T1 1.5R @ 207.07 |
| Stop hit — per-position SL triggered | 2025-03-13 12:05:00 | 206.00 | 205.98 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-03-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 09:45:00 | 235.75 | 238.12 | 0.00 | ORB-short ORB[238.05,241.40] vol=2.8x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-03-25 09:55:00 | 236.95 | 237.84 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-03-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 10:45:00 | 231.39 | 231.93 | 0.00 | ORB-short ORB[231.90,235.25] vol=2.0x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-03-28 11:00:00 | 232.13 | 231.93 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 232.90 | 231.63 | 0.00 | ORB-long ORB[230.02,232.49] vol=3.6x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-04-02 09:45:00 | 232.17 | 231.98 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 11:15:00 | 225.86 | 227.04 | 0.00 | ORB-short ORB[226.72,229.85] vol=2.4x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-04-08 12:05:00 | 226.68 | 226.95 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-04-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:00:00 | 226.40 | 227.18 | 0.00 | ORB-short ORB[226.66,228.67] vol=1.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 10:35:00 | 225.22 | 226.22 | 0.00 | T1 1.5R @ 225.22 |
| Stop hit — per-position SL triggered | 2025-04-09 10:40:00 | 226.40 | 226.13 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-15 11:00:00 | 225.62 | 226.27 | 0.00 | ORB-short ORB[226.45,227.89] vol=1.8x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 11:10:00 | 225.12 | 226.11 | 0.00 | T1 1.5R @ 225.12 |
| Stop hit — per-position SL triggered | 2025-04-15 13:40:00 | 225.62 | 225.65 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:35:00 | 227.67 | 226.89 | 0.00 | ORB-long ORB[225.81,227.45] vol=1.7x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-04-16 09:40:00 | 227.10 | 226.94 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-04-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:20:00 | 232.16 | 230.54 | 0.00 | ORB-long ORB[229.00,231.90] vol=2.1x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:40:00 | 233.36 | 231.45 | 0.00 | T1 1.5R @ 233.36 |
| Target hit | 2025-04-21 14:30:00 | 233.04 | 233.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — SELL (started 2025-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:50:00 | 230.16 | 231.22 | 0.00 | ORB-short ORB[230.66,232.85] vol=3.1x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-04-23 10:00:00 | 230.90 | 231.15 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 227.44 | 228.10 | 0.00 | ORB-short ORB[228.00,229.49] vol=2.5x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 226.66 | 227.37 | 0.00 | T1 1.5R @ 226.66 |
| Target hit | 2025-04-25 12:00:00 | 225.16 | 225.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 228.52 | 227.22 | 0.00 | ORB-long ORB[225.51,227.48] vol=4.6x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 10:15:00 | 229.67 | 227.86 | 0.00 | T1 1.5R @ 229.67 |
| Target hit | 2025-04-28 15:20:00 | 231.61 | 230.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2025-05-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 09:35:00 | 232.36 | 231.53 | 0.00 | ORB-long ORB[230.10,232.30] vol=1.5x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 10:00:00 | 233.46 | 232.37 | 0.00 | T1 1.5R @ 233.46 |
| Stop hit — per-position SL triggered | 2025-05-02 10:35:00 | 232.36 | 232.45 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-05-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:20:00 | 232.50 | 231.62 | 0.00 | ORB-long ORB[230.40,231.88] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-05-05 10:55:00 | 231.99 | 231.91 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:15:00 | 230.80 | 228.12 | 0.00 | ORB-long ORB[226.71,228.64] vol=3.9x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-05-07 11:35:00 | 230.25 | 228.50 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:40:00 | 230.09 | 230.79 | 0.00 | ORB-short ORB[230.24,231.78] vol=2.0x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:10:00 | 229.38 | 230.47 | 0.00 | T1 1.5R @ 229.38 |
| Target hit | 2025-05-08 15:20:00 | 227.54 | 228.93 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:30:00 | 178.65 | 2024-05-15 09:45:00 | 179.71 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-05-15 09:30:00 | 178.65 | 2024-05-15 15:10:00 | 179.45 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-30 09:40:00 | 170.45 | 2024-05-30 10:05:00 | 169.60 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-05-30 09:40:00 | 170.45 | 2024-05-30 15:20:00 | 168.15 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2024-05-31 10:20:00 | 168.10 | 2024-05-31 10:45:00 | 168.87 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-06-11 10:00:00 | 180.35 | 2024-06-11 10:05:00 | 181.39 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-06-11 10:00:00 | 180.35 | 2024-06-11 12:45:00 | 181.84 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-06-12 10:55:00 | 184.15 | 2024-06-12 15:20:00 | 184.58 | TARGET_HIT | 1.00 | 0.23% |
| BUY | retest1 | 2024-06-18 10:40:00 | 188.00 | 2024-06-18 10:55:00 | 189.03 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-18 10:40:00 | 188.00 | 2024-06-18 15:20:00 | 191.49 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2024-06-27 09:30:00 | 199.38 | 2024-06-27 09:35:00 | 200.66 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-06-27 09:30:00 | 199.38 | 2024-06-27 15:20:00 | 211.12 | TARGET_HIT | 0.50 | 5.89% |
| BUY | retest1 | 2024-07-01 09:30:00 | 212.21 | 2024-07-01 09:50:00 | 211.27 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-02 09:45:00 | 208.10 | 2024-07-02 10:40:00 | 206.99 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-07-02 09:45:00 | 208.10 | 2024-07-02 11:05:00 | 208.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-03 10:50:00 | 209.30 | 2024-07-03 11:05:00 | 210.06 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-07-03 10:50:00 | 209.30 | 2024-07-03 11:10:00 | 209.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 10:10:00 | 207.50 | 2024-07-08 11:05:00 | 206.56 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-08 10:10:00 | 207.50 | 2024-07-08 11:10:00 | 207.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-09 09:35:00 | 206.30 | 2024-07-09 12:00:00 | 206.87 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-12 10:25:00 | 214.70 | 2024-07-12 10:50:00 | 215.88 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-07-12 10:25:00 | 214.70 | 2024-07-12 11:20:00 | 214.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 09:50:00 | 207.95 | 2024-07-26 10:00:00 | 207.18 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-01 09:35:00 | 216.81 | 2024-08-01 09:40:00 | 216.17 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-09 11:05:00 | 203.58 | 2024-08-09 11:40:00 | 204.68 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-09 11:05:00 | 203.58 | 2024-08-09 15:20:00 | 204.97 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2024-08-12 09:35:00 | 204.50 | 2024-08-12 09:40:00 | 205.81 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-08-12 09:35:00 | 204.50 | 2024-08-12 15:20:00 | 210.17 | TARGET_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2024-08-20 09:45:00 | 203.15 | 2024-08-20 11:00:00 | 204.10 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-08-20 09:45:00 | 203.15 | 2024-08-20 15:20:00 | 208.25 | TARGET_HIT | 0.50 | 2.51% |
| SELL | retest1 | 2024-08-27 09:35:00 | 212.44 | 2024-08-27 09:40:00 | 213.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-29 10:45:00 | 213.45 | 2024-08-29 11:05:00 | 212.43 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-08-29 10:45:00 | 213.45 | 2024-08-29 11:10:00 | 213.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 10:15:00 | 215.24 | 2024-09-03 10:20:00 | 214.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-05 11:10:00 | 211.39 | 2024-09-05 11:20:00 | 212.18 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-05 11:10:00 | 211.39 | 2024-09-05 11:25:00 | 211.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 09:50:00 | 207.75 | 2024-09-06 10:05:00 | 206.81 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-06 09:50:00 | 207.75 | 2024-09-06 11:45:00 | 207.36 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2024-09-10 09:55:00 | 204.06 | 2024-09-10 11:15:00 | 203.12 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-10 09:55:00 | 204.06 | 2024-09-10 12:30:00 | 204.06 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 11:15:00 | 204.67 | 2024-09-19 11:25:00 | 203.27 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-09-19 11:15:00 | 204.67 | 2024-09-19 11:50:00 | 204.67 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 09:35:00 | 208.03 | 2024-09-24 09:40:00 | 208.58 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-27 10:45:00 | 206.23 | 2024-09-27 10:50:00 | 205.65 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-30 09:30:00 | 201.58 | 2024-09-30 09:40:00 | 200.72 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-30 09:30:00 | 201.58 | 2024-09-30 09:55:00 | 201.58 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-03 09:50:00 | 193.27 | 2024-10-03 10:00:00 | 192.25 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-10-03 09:50:00 | 193.27 | 2024-10-03 11:50:00 | 193.27 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-15 11:10:00 | 181.81 | 2024-10-15 11:25:00 | 182.34 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-16 10:05:00 | 181.80 | 2024-10-16 10:15:00 | 181.00 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-16 10:05:00 | 181.80 | 2024-10-16 13:10:00 | 180.61 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2024-11-19 09:30:00 | 156.40 | 2024-11-19 09:40:00 | 157.48 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-11-19 09:30:00 | 156.40 | 2024-11-19 11:55:00 | 157.63 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2024-11-25 09:30:00 | 155.14 | 2024-11-25 09:40:00 | 154.46 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-12-06 09:30:00 | 166.86 | 2024-12-06 10:10:00 | 167.64 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-12-13 10:10:00 | 173.80 | 2024-12-13 10:25:00 | 172.49 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-12-13 10:10:00 | 173.80 | 2024-12-13 10:55:00 | 173.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-01 09:30:00 | 188.48 | 2025-01-01 11:10:00 | 189.27 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-01-10 10:50:00 | 180.62 | 2025-01-10 10:55:00 | 182.23 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2025-01-10 10:50:00 | 180.62 | 2025-01-10 14:15:00 | 180.62 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 10:10:00 | 185.70 | 2025-01-17 10:45:00 | 185.08 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-27 10:15:00 | 190.69 | 2025-01-27 10:20:00 | 189.36 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2025-01-27 10:15:00 | 190.69 | 2025-01-27 11:50:00 | 190.69 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-30 10:55:00 | 199.35 | 2025-01-30 11:00:00 | 200.04 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-31 09:30:00 | 196.46 | 2025-01-31 09:45:00 | 197.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-02-01 09:45:00 | 198.30 | 2025-02-01 10:05:00 | 197.46 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-05 10:55:00 | 200.41 | 2025-03-05 11:40:00 | 201.48 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-03-05 10:55:00 | 200.41 | 2025-03-05 12:55:00 | 200.41 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 10:30:00 | 205.05 | 2025-03-07 10:35:00 | 206.12 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-03-07 10:30:00 | 205.05 | 2025-03-07 11:50:00 | 206.28 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2025-03-12 10:10:00 | 198.55 | 2025-03-12 10:15:00 | 199.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-03-13 10:40:00 | 206.00 | 2025-03-13 11:50:00 | 207.07 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-03-13 10:40:00 | 206.00 | 2025-03-13 12:05:00 | 206.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-25 09:45:00 | 235.75 | 2025-03-25 09:55:00 | 236.95 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-03-28 10:45:00 | 231.39 | 2025-03-28 11:00:00 | 232.13 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-02 09:30:00 | 232.90 | 2025-04-02 09:45:00 | 232.17 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-08 11:15:00 | 225.86 | 2025-04-08 12:05:00 | 226.68 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-04-09 10:00:00 | 226.40 | 2025-04-09 10:35:00 | 225.22 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-04-09 10:00:00 | 226.40 | 2025-04-09 10:40:00 | 226.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-15 11:00:00 | 225.62 | 2025-04-15 11:10:00 | 225.12 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-04-15 11:00:00 | 225.62 | 2025-04-15 13:40:00 | 225.62 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-16 09:35:00 | 227.67 | 2025-04-16 09:40:00 | 227.10 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-04-21 10:20:00 | 232.16 | 2025-04-21 10:40:00 | 233.36 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-21 10:20:00 | 232.16 | 2025-04-21 14:30:00 | 233.04 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-04-23 09:50:00 | 230.16 | 2025-04-23 10:00:00 | 230.90 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-25 09:30:00 | 227.44 | 2025-04-25 09:35:00 | 226.66 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-04-25 09:30:00 | 227.44 | 2025-04-25 12:00:00 | 225.16 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2025-04-28 09:30:00 | 228.52 | 2025-04-28 10:15:00 | 229.67 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-28 09:30:00 | 228.52 | 2025-04-28 15:20:00 | 231.61 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2025-05-02 09:35:00 | 232.36 | 2025-05-02 10:00:00 | 233.46 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-02 09:35:00 | 232.36 | 2025-05-02 10:35:00 | 232.36 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 10:20:00 | 232.50 | 2025-05-05 10:55:00 | 231.99 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-05-07 11:15:00 | 230.80 | 2025-05-07 11:35:00 | 230.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-08 09:40:00 | 230.09 | 2025-05-08 10:10:00 | 229.38 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-05-08 09:40:00 | 230.09 | 2025-05-08 15:20:00 | 227.54 | TARGET_HIT | 0.50 | 1.11% |
