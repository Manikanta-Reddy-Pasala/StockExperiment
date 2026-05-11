# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-05-30 15:25:00 (38016 bars)
- **Last close:** 561.00
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
| ENTRY1 | 83 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 19 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 64
- **Target hits / Stop hits / Partials:** 19 / 64 / 35
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 24.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 23 | 44.2% | 9 | 29 | 14 | 0.17% | 8.9% |
| BUY @ 2nd Alert (retest1) | 52 | 23 | 44.2% | 9 | 29 | 14 | 0.17% | 8.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 31 | 47.0% | 10 | 35 | 21 | 0.23% | 15.5% |
| SELL @ 2nd Alert (retest1) | 66 | 31 | 47.0% | 10 | 35 | 21 | 0.23% | 15.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 118 | 54 | 45.8% | 19 | 64 | 35 | 0.21% | 24.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:45:00 | 141.05 | 139.67 | 0.00 | ORB-long ORB[139.00,140.65] vol=1.6x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 14:30:00 | 142.25 | 140.94 | 0.00 | T1 1.5R @ 142.25 |
| Target hit | 2023-05-15 15:20:00 | 142.85 | 141.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2023-05-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 10:35:00 | 147.40 | 146.24 | 0.00 | ORB-long ORB[144.95,146.50] vol=2.7x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 10:45:00 | 148.22 | 146.79 | 0.00 | T1 1.5R @ 148.22 |
| Target hit | 2023-05-18 12:20:00 | 148.00 | 148.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2023-05-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 10:20:00 | 139.70 | 140.53 | 0.00 | ORB-short ORB[140.15,141.10] vol=3.2x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-05-25 10:45:00 | 140.27 | 140.49 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 10:55:00 | 148.90 | 149.54 | 0.00 | ORB-short ORB[149.45,151.40] vol=3.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-05-30 11:05:00 | 149.42 | 149.53 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:30:00 | 153.00 | 151.99 | 0.00 | ORB-long ORB[150.35,152.05] vol=7.1x ATR=0.43 |
| Stop hit — per-position SL triggered | 2023-05-31 10:45:00 | 152.57 | 152.10 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-01 11:05:00 | 150.95 | 152.28 | 0.00 | ORB-short ORB[151.80,153.00] vol=3.8x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-06-01 11:15:00 | 151.31 | 152.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:30:00 | 153.65 | 153.16 | 0.00 | ORB-long ORB[152.15,153.45] vol=1.9x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 09:40:00 | 154.60 | 153.83 | 0.00 | T1 1.5R @ 154.60 |
| Stop hit — per-position SL triggered | 2023-06-02 09:50:00 | 153.65 | 153.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:30:00 | 157.05 | 158.47 | 0.00 | ORB-short ORB[157.30,159.00] vol=1.5x ATR=0.79 |
| Stop hit — per-position SL triggered | 2023-06-06 10:50:00 | 157.84 | 158.27 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:00:00 | 163.20 | 161.26 | 0.00 | ORB-long ORB[159.30,161.15] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-06-07 10:15:00 | 162.40 | 162.04 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 10:45:00 | 164.85 | 163.22 | 0.00 | ORB-long ORB[162.00,163.90] vol=2.2x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 10:50:00 | 166.03 | 164.62 | 0.00 | T1 1.5R @ 166.03 |
| Target hit | 2023-06-09 12:45:00 | 165.00 | 165.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2023-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:35:00 | 169.75 | 169.40 | 0.00 | ORB-long ORB[167.85,169.70] vol=3.7x ATR=0.72 |
| Stop hit — per-position SL triggered | 2023-06-13 09:50:00 | 169.03 | 169.39 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:50:00 | 172.00 | 170.50 | 0.00 | ORB-long ORB[169.50,170.80] vol=6.2x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-06-14 11:00:00 | 171.38 | 170.61 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:30:00 | 169.15 | 167.93 | 0.00 | ORB-long ORB[167.00,168.00] vol=1.5x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-06-22 09:40:00 | 168.44 | 168.32 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:40:00 | 166.55 | 164.96 | 0.00 | ORB-long ORB[163.70,165.00] vol=2.2x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-06-26 10:00:00 | 165.63 | 165.36 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 10:20:00 | 182.50 | 181.28 | 0.00 | ORB-long ORB[179.55,181.80] vol=1.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-07-07 10:25:00 | 181.89 | 181.32 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 09:40:00 | 177.60 | 178.72 | 0.00 | ORB-short ORB[178.50,180.55] vol=1.7x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 10:55:00 | 176.30 | 177.40 | 0.00 | T1 1.5R @ 176.30 |
| Stop hit — per-position SL triggered | 2023-07-10 15:05:00 | 177.60 | 176.95 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:50:00 | 179.85 | 178.96 | 0.00 | ORB-long ORB[177.20,179.45] vol=2.2x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 10:05:00 | 180.90 | 180.10 | 0.00 | T1 1.5R @ 180.90 |
| Target hit | 2023-07-11 12:15:00 | 183.50 | 183.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2023-07-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 09:45:00 | 181.05 | 182.07 | 0.00 | ORB-short ORB[181.35,183.70] vol=1.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2023-07-14 10:00:00 | 182.02 | 181.82 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:35:00 | 193.35 | 192.71 | 0.00 | ORB-long ORB[192.00,193.00] vol=1.7x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-07-25 10:25:00 | 192.69 | 192.94 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-26 09:35:00 | 191.65 | 193.16 | 0.00 | ORB-short ORB[192.05,194.00] vol=3.4x ATR=0.68 |
| Stop hit — per-position SL triggered | 2023-07-26 09:40:00 | 192.33 | 192.95 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 10:05:00 | 200.45 | 202.04 | 0.00 | ORB-short ORB[202.00,205.00] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2023-07-28 15:10:00 | 201.66 | 200.97 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-31 10:40:00 | 200.00 | 200.75 | 0.00 | ORB-short ORB[201.05,202.50] vol=2.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 11:15:00 | 198.76 | 200.17 | 0.00 | T1 1.5R @ 198.76 |
| Target hit | 2023-07-31 12:25:00 | 199.40 | 199.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — SELL (started 2023-08-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:55:00 | 200.85 | 202.21 | 0.00 | ORB-short ORB[201.65,203.35] vol=2.3x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-08-01 11:20:00 | 201.48 | 202.02 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 09:30:00 | 190.95 | 193.28 | 0.00 | ORB-short ORB[193.10,195.90] vol=3.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2023-08-09 09:35:00 | 191.99 | 192.95 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 09:55:00 | 192.40 | 193.51 | 0.00 | ORB-short ORB[193.15,194.90] vol=1.5x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 10:35:00 | 191.01 | 192.53 | 0.00 | T1 1.5R @ 191.01 |
| Target hit | 2023-08-10 15:20:00 | 188.55 | 190.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2023-08-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:55:00 | 198.30 | 199.69 | 0.00 | ORB-short ORB[200.40,202.35] vol=1.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-08-18 10:20:00 | 199.28 | 199.31 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 09:45:00 | 200.85 | 199.26 | 0.00 | ORB-long ORB[197.00,199.95] vol=2.2x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-21 14:00:00 | 202.42 | 200.34 | 0.00 | T1 1.5R @ 202.42 |
| Target hit | 2023-08-21 15:20:00 | 202.20 | 200.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2023-09-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 11:05:00 | 216.95 | 217.41 | 0.00 | ORB-short ORB[218.25,220.90] vol=1.9x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-09-01 11:15:00 | 217.76 | 217.42 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-09-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:35:00 | 222.40 | 220.89 | 0.00 | ORB-long ORB[219.50,221.70] vol=1.8x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-09-08 09:50:00 | 221.46 | 221.08 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 10:05:00 | 216.75 | 218.79 | 0.00 | ORB-short ORB[217.90,220.15] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-09-21 10:20:00 | 217.83 | 218.63 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:30:00 | 214.50 | 215.11 | 0.00 | ORB-short ORB[215.25,218.35] vol=5.1x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 09:55:00 | 212.95 | 213.83 | 0.00 | T1 1.5R @ 212.95 |
| Target hit | 2023-09-22 10:25:00 | 214.25 | 213.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — SELL (started 2023-09-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 11:10:00 | 211.55 | 213.64 | 0.00 | ORB-short ORB[213.80,215.80] vol=2.5x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 11:50:00 | 210.57 | 213.17 | 0.00 | T1 1.5R @ 210.57 |
| Stop hit — per-position SL triggered | 2023-09-25 12:05:00 | 211.55 | 212.75 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 10:00:00 | 216.00 | 215.02 | 0.00 | ORB-long ORB[213.65,215.65] vol=1.7x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 11:20:00 | 217.19 | 215.94 | 0.00 | T1 1.5R @ 217.19 |
| Target hit | 2023-09-27 15:20:00 | 222.25 | 219.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2023-09-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 11:05:00 | 220.00 | 221.31 | 0.00 | ORB-short ORB[221.40,224.20] vol=3.4x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 12:00:00 | 218.76 | 220.94 | 0.00 | T1 1.5R @ 218.76 |
| Stop hit — per-position SL triggered | 2023-09-28 14:50:00 | 220.00 | 220.07 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:40:00 | 236.00 | 235.21 | 0.00 | ORB-long ORB[233.85,235.80] vol=1.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-10-11 09:50:00 | 234.92 | 234.86 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 11:05:00 | 237.00 | 239.05 | 0.00 | ORB-short ORB[238.95,241.65] vol=2.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-10-17 11:20:00 | 237.65 | 238.84 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 11:00:00 | 236.65 | 239.39 | 0.00 | ORB-short ORB[239.05,241.00] vol=3.2x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:15:00 | 235.49 | 238.71 | 0.00 | T1 1.5R @ 235.49 |
| Stop hit — per-position SL triggered | 2023-10-18 11:20:00 | 236.65 | 238.69 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-10-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 09:50:00 | 240.95 | 239.10 | 0.00 | ORB-long ORB[236.60,239.50] vol=1.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2023-10-19 10:20:00 | 239.81 | 239.77 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-11-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 09:30:00 | 250.40 | 248.08 | 0.00 | ORB-long ORB[245.75,249.00] vol=2.7x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-11-09 09:35:00 | 249.32 | 248.35 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-11-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 09:40:00 | 250.30 | 249.07 | 0.00 | ORB-long ORB[246.30,249.35] vol=7.3x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-11-10 09:45:00 | 249.38 | 249.09 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-11-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:40:00 | 252.45 | 249.95 | 0.00 | ORB-long ORB[248.25,251.40] vol=2.1x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 09:50:00 | 253.97 | 251.67 | 0.00 | T1 1.5R @ 253.97 |
| Stop hit — per-position SL triggered | 2023-11-15 10:00:00 | 252.45 | 251.92 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-11-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 09:40:00 | 254.10 | 252.57 | 0.00 | ORB-long ORB[250.30,253.00] vol=3.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-11-16 10:10:00 | 253.18 | 252.98 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 10:15:00 | 255.90 | 254.06 | 0.00 | ORB-long ORB[251.75,254.80] vol=2.9x ATR=0.93 |
| Stop hit — per-position SL triggered | 2023-11-17 10:20:00 | 254.97 | 254.15 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 11:15:00 | 255.95 | 258.32 | 0.00 | ORB-short ORB[257.85,261.00] vol=5.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 11:25:00 | 254.74 | 257.89 | 0.00 | T1 1.5R @ 254.74 |
| Target hit | 2023-11-20 15:20:00 | 253.55 | 256.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2023-11-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 11:10:00 | 251.15 | 253.23 | 0.00 | ORB-short ORB[253.50,255.80] vol=4.4x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 11:15:00 | 250.08 | 252.40 | 0.00 | T1 1.5R @ 250.08 |
| Stop hit — per-position SL triggered | 2023-11-21 11:20:00 | 251.15 | 252.24 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-11-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 10:35:00 | 253.50 | 255.37 | 0.00 | ORB-short ORB[254.40,257.50] vol=2.1x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-11-23 10:40:00 | 254.35 | 255.24 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-11-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:05:00 | 252.25 | 253.54 | 0.00 | ORB-short ORB[252.75,254.50] vol=1.8x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 10:15:00 | 251.04 | 253.28 | 0.00 | T1 1.5R @ 251.04 |
| Stop hit — per-position SL triggered | 2023-11-24 10:45:00 | 252.25 | 253.09 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-11-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:45:00 | 259.90 | 257.72 | 0.00 | ORB-long ORB[255.55,258.75] vol=1.9x ATR=1.20 |
| Stop hit — per-position SL triggered | 2023-11-29 09:50:00 | 258.70 | 257.82 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 09:35:00 | 270.00 | 268.92 | 0.00 | ORB-long ORB[266.15,269.95] vol=1.9x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 09:55:00 | 271.84 | 270.42 | 0.00 | T1 1.5R @ 271.84 |
| Target hit | 2023-12-05 10:05:00 | 270.55 | 270.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2023-12-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 10:10:00 | 271.60 | 270.53 | 0.00 | ORB-long ORB[267.50,271.40] vol=2.2x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 10:25:00 | 273.17 | 271.30 | 0.00 | T1 1.5R @ 273.17 |
| Target hit | 2023-12-07 15:20:00 | 277.40 | 275.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2023-12-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 10:05:00 | 281.25 | 279.77 | 0.00 | ORB-long ORB[276.80,280.00] vol=1.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:25:00 | 283.15 | 280.42 | 0.00 | T1 1.5R @ 283.15 |
| Stop hit — per-position SL triggered | 2023-12-08 11:00:00 | 281.25 | 281.04 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-12-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:55:00 | 285.70 | 287.01 | 0.00 | ORB-short ORB[286.70,289.90] vol=1.5x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-12-12 11:20:00 | 286.82 | 286.87 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 09:35:00 | 287.00 | 285.15 | 0.00 | ORB-long ORB[282.75,285.70] vol=2.0x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 10:10:00 | 289.09 | 287.05 | 0.00 | T1 1.5R @ 289.09 |
| Target hit | 2023-12-13 12:20:00 | 288.10 | 288.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — SELL (started 2023-12-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 10:00:00 | 295.00 | 296.88 | 0.00 | ORB-short ORB[295.65,298.50] vol=1.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2023-12-28 10:15:00 | 295.99 | 296.58 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-01-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 10:30:00 | 298.50 | 297.09 | 0.00 | ORB-long ORB[295.45,297.90] vol=1.8x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-01-01 11:10:00 | 297.27 | 297.16 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:30:00 | 307.90 | 305.36 | 0.00 | ORB-long ORB[302.40,306.40] vol=4.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-01-02 09:40:00 | 306.48 | 305.93 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:55:00 | 303.80 | 305.69 | 0.00 | ORB-short ORB[304.15,308.35] vol=1.8x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-01-03 11:05:00 | 305.43 | 305.31 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-01-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:10:00 | 323.00 | 320.86 | 0.00 | ORB-long ORB[317.20,321.70] vol=1.7x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-01-09 10:25:00 | 321.44 | 321.22 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-01-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:35:00 | 324.40 | 322.98 | 0.00 | ORB-long ORB[321.35,324.00] vol=1.6x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 09:50:00 | 326.03 | 323.94 | 0.00 | T1 1.5R @ 326.03 |
| Stop hit — per-position SL triggered | 2024-01-12 11:00:00 | 324.40 | 324.94 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-01-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:45:00 | 311.25 | 316.93 | 0.00 | ORB-short ORB[316.25,320.95] vol=1.8x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:55:00 | 308.72 | 313.96 | 0.00 | T1 1.5R @ 308.72 |
| Stop hit — per-position SL triggered | 2024-01-18 10:00:00 | 311.25 | 313.89 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-01-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 09:40:00 | 318.65 | 321.05 | 0.00 | ORB-short ORB[321.00,324.00] vol=3.2x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 10:55:00 | 316.42 | 320.05 | 0.00 | T1 1.5R @ 316.42 |
| Stop hit — per-position SL triggered | 2024-01-19 11:20:00 | 318.65 | 319.40 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 11:05:00 | 315.00 | 315.93 | 0.00 | ORB-short ORB[316.75,320.90] vol=1.8x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 11:40:00 | 312.81 | 315.67 | 0.00 | T1 1.5R @ 312.81 |
| Stop hit — per-position SL triggered | 2024-01-23 13:10:00 | 315.00 | 314.64 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 11:05:00 | 313.25 | 315.18 | 0.00 | ORB-short ORB[314.55,317.90] vol=2.3x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-01-25 11:35:00 | 314.40 | 314.76 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 11:10:00 | 313.00 | 315.46 | 0.00 | ORB-short ORB[316.00,319.00] vol=4.7x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-01-29 11:15:00 | 314.02 | 315.34 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-01-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 09:40:00 | 316.00 | 314.37 | 0.00 | ORB-long ORB[312.85,314.95] vol=2.1x ATR=1.24 |
| Stop hit — per-position SL triggered | 2024-01-30 09:50:00 | 314.76 | 314.45 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-31 11:05:00 | 314.10 | 316.21 | 0.00 | ORB-short ORB[316.00,319.10] vol=1.7x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-01-31 11:20:00 | 315.09 | 316.14 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-02-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 09:50:00 | 329.80 | 327.46 | 0.00 | ORB-long ORB[323.50,327.90] vol=2.5x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 10:00:00 | 332.14 | 328.66 | 0.00 | T1 1.5R @ 332.14 |
| Stop hit — per-position SL triggered | 2024-02-05 11:40:00 | 329.80 | 330.65 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 09:35:00 | 329.20 | 332.52 | 0.00 | ORB-short ORB[331.30,336.00] vol=2.9x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 12:40:00 | 325.57 | 329.93 | 0.00 | T1 1.5R @ 325.57 |
| Target hit | 2024-02-07 15:20:00 | 323.30 | 328.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2024-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:50:00 | 345.25 | 347.01 | 0.00 | ORB-short ORB[346.50,349.50] vol=9.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-02-20 11:05:00 | 346.29 | 346.92 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-02-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:35:00 | 344.55 | 347.21 | 0.00 | ORB-short ORB[345.90,349.80] vol=1.5x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 13:25:00 | 342.20 | 346.05 | 0.00 | T1 1.5R @ 342.20 |
| Target hit | 2024-02-21 15:20:00 | 340.80 | 343.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — SELL (started 2024-02-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:00:00 | 345.90 | 348.72 | 0.00 | ORB-short ORB[349.05,351.50] vol=1.6x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:25:00 | 344.36 | 347.67 | 0.00 | T1 1.5R @ 344.36 |
| Target hit | 2024-02-28 15:20:00 | 339.10 | 342.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 349.00 | 351.66 | 0.00 | ORB-short ORB[351.00,354.90] vol=3.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-03-06 09:35:00 | 350.44 | 351.11 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-03-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 10:30:00 | 343.75 | 344.09 | 0.00 | ORB-short ORB[344.40,348.70] vol=5.2x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 10:45:00 | 341.63 | 343.94 | 0.00 | T1 1.5R @ 341.63 |
| Target hit | 2024-03-11 15:20:00 | 337.20 | 340.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2024-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 09:35:00 | 290.45 | 290.57 | 0.00 | ORB-short ORB[291.10,293.00] vol=2.3x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:00:00 | 287.60 | 290.21 | 0.00 | T1 1.5R @ 287.60 |
| Target hit | 2024-03-20 12:35:00 | 287.60 | 286.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 75 — BUY (started 2024-03-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:50:00 | 302.40 | 298.47 | 0.00 | ORB-long ORB[294.00,298.45] vol=2.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-03-21 12:05:00 | 300.89 | 299.76 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 329.05 | 330.93 | 0.00 | ORB-short ORB[329.70,333.90] vol=1.8x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 10:40:00 | 326.81 | 329.97 | 0.00 | T1 1.5R @ 326.81 |
| Target hit | 2024-04-04 15:20:00 | 326.70 | 326.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2024-04-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 10:30:00 | 331.05 | 327.13 | 0.00 | ORB-long ORB[324.05,328.85] vol=2.8x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-04-05 10:50:00 | 329.38 | 327.45 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:30:00 | 359.00 | 356.84 | 0.00 | ORB-long ORB[354.00,358.00] vol=3.2x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-04-23 09:35:00 | 357.41 | 356.93 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-04-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 09:55:00 | 368.95 | 366.79 | 0.00 | ORB-long ORB[364.20,368.00] vol=4.0x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-04-26 10:00:00 | 366.79 | 366.83 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 09:35:00 | 361.50 | 363.95 | 0.00 | ORB-short ORB[362.75,366.80] vol=2.6x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 09:45:00 | 359.23 | 363.00 | 0.00 | T1 1.5R @ 359.23 |
| Stop hit — per-position SL triggered | 2024-04-29 10:00:00 | 361.50 | 362.31 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-05-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 09:50:00 | 357.70 | 360.37 | 0.00 | ORB-short ORB[358.85,364.00] vol=2.1x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-05-02 10:35:00 | 359.30 | 359.93 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-05-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:00:00 | 359.30 | 360.49 | 0.00 | ORB-short ORB[360.00,362.70] vol=2.3x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 10:05:00 | 357.68 | 359.98 | 0.00 | T1 1.5R @ 357.68 |
| Stop hit — per-position SL triggered | 2024-05-03 10:15:00 | 359.30 | 359.79 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-05-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-10 09:35:00 | 362.25 | 366.17 | 0.00 | ORB-short ORB[365.90,369.40] vol=2.2x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-05-10 09:40:00 | 364.21 | 365.79 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:45:00 | 141.05 | 2023-05-15 14:30:00 | 142.25 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2023-05-15 09:45:00 | 141.05 | 2023-05-15 15:20:00 | 142.85 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2023-05-18 10:35:00 | 147.40 | 2023-05-18 10:45:00 | 148.22 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-05-18 10:35:00 | 147.40 | 2023-05-18 12:20:00 | 148.00 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2023-05-25 10:20:00 | 139.70 | 2023-05-25 10:45:00 | 140.27 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-05-30 10:55:00 | 148.90 | 2023-05-30 11:05:00 | 149.42 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-05-31 10:30:00 | 153.00 | 2023-05-31 10:45:00 | 152.57 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-06-01 11:05:00 | 150.95 | 2023-06-01 11:15:00 | 151.31 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-06-02 09:30:00 | 153.65 | 2023-06-02 09:40:00 | 154.60 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2023-06-02 09:30:00 | 153.65 | 2023-06-02 09:50:00 | 153.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-06 10:30:00 | 157.05 | 2023-06-06 10:50:00 | 157.84 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2023-06-07 10:00:00 | 163.20 | 2023-06-07 10:15:00 | 162.40 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2023-06-09 10:45:00 | 164.85 | 2023-06-09 10:50:00 | 166.03 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2023-06-09 10:45:00 | 164.85 | 2023-06-09 12:45:00 | 165.00 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2023-06-13 09:35:00 | 169.75 | 2023-06-13 09:50:00 | 169.03 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-06-14 10:50:00 | 172.00 | 2023-06-14 11:00:00 | 171.38 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-06-22 09:30:00 | 169.15 | 2023-06-22 09:40:00 | 168.44 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-06-26 09:40:00 | 166.55 | 2023-06-26 10:00:00 | 165.63 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2023-07-07 10:20:00 | 182.50 | 2023-07-07 10:25:00 | 181.89 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-07-10 09:40:00 | 177.60 | 2023-07-10 10:55:00 | 176.30 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2023-07-10 09:40:00 | 177.60 | 2023-07-10 15:05:00 | 177.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-11 09:50:00 | 179.85 | 2023-07-11 10:05:00 | 180.90 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-07-11 09:50:00 | 179.85 | 2023-07-11 12:15:00 | 183.50 | TARGET_HIT | 0.50 | 2.03% |
| SELL | retest1 | 2023-07-14 09:45:00 | 181.05 | 2023-07-14 10:00:00 | 182.02 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2023-07-25 09:35:00 | 193.35 | 2023-07-25 10:25:00 | 192.69 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-26 09:35:00 | 191.65 | 2023-07-26 09:40:00 | 192.33 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-07-28 10:05:00 | 200.45 | 2023-07-28 15:10:00 | 201.66 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2023-07-31 10:40:00 | 200.00 | 2023-07-31 11:15:00 | 198.76 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2023-07-31 10:40:00 | 200.00 | 2023-07-31 12:25:00 | 199.40 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2023-08-01 10:55:00 | 200.85 | 2023-08-01 11:20:00 | 201.48 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-08-09 09:30:00 | 190.95 | 2023-08-09 09:35:00 | 191.99 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2023-08-10 09:55:00 | 192.40 | 2023-08-10 10:35:00 | 191.01 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2023-08-10 09:55:00 | 192.40 | 2023-08-10 15:20:00 | 188.55 | TARGET_HIT | 0.50 | 2.00% |
| SELL | retest1 | 2023-08-18 09:55:00 | 198.30 | 2023-08-18 10:20:00 | 199.28 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2023-08-21 09:45:00 | 200.85 | 2023-08-21 14:00:00 | 202.42 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2023-08-21 09:45:00 | 200.85 | 2023-08-21 15:20:00 | 202.20 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2023-09-01 11:05:00 | 216.95 | 2023-09-01 11:15:00 | 217.76 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-09-08 09:35:00 | 222.40 | 2023-09-08 09:50:00 | 221.46 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-09-21 10:05:00 | 216.75 | 2023-09-21 10:20:00 | 217.83 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-09-22 09:30:00 | 214.50 | 2023-09-22 09:55:00 | 212.95 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2023-09-22 09:30:00 | 214.50 | 2023-09-22 10:25:00 | 214.25 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2023-09-25 11:10:00 | 211.55 | 2023-09-25 11:50:00 | 210.57 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-09-25 11:10:00 | 211.55 | 2023-09-25 12:05:00 | 211.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-27 10:00:00 | 216.00 | 2023-09-27 11:20:00 | 217.19 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2023-09-27 10:00:00 | 216.00 | 2023-09-27 15:20:00 | 222.25 | TARGET_HIT | 0.50 | 2.89% |
| SELL | retest1 | 2023-09-28 11:05:00 | 220.00 | 2023-09-28 12:00:00 | 218.76 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-09-28 11:05:00 | 220.00 | 2023-09-28 14:50:00 | 220.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-11 09:40:00 | 236.00 | 2023-10-11 09:50:00 | 234.92 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-10-17 11:05:00 | 237.00 | 2023-10-17 11:20:00 | 237.65 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-10-18 11:00:00 | 236.65 | 2023-10-18 11:15:00 | 235.49 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-10-18 11:00:00 | 236.65 | 2023-10-18 11:20:00 | 236.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-19 09:50:00 | 240.95 | 2023-10-19 10:20:00 | 239.81 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2023-11-09 09:30:00 | 250.40 | 2023-11-09 09:35:00 | 249.32 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-11-10 09:40:00 | 250.30 | 2023-11-10 09:45:00 | 249.38 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-11-15 09:40:00 | 252.45 | 2023-11-15 09:50:00 | 253.97 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2023-11-15 09:40:00 | 252.45 | 2023-11-15 10:00:00 | 252.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-16 09:40:00 | 254.10 | 2023-11-16 10:10:00 | 253.18 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-11-17 10:15:00 | 255.90 | 2023-11-17 10:20:00 | 254.97 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-11-20 11:15:00 | 255.95 | 2023-11-20 11:25:00 | 254.74 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-11-20 11:15:00 | 255.95 | 2023-11-20 15:20:00 | 253.55 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2023-11-21 11:10:00 | 251.15 | 2023-11-21 11:15:00 | 250.08 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-11-21 11:10:00 | 251.15 | 2023-11-21 11:20:00 | 251.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-23 10:35:00 | 253.50 | 2023-11-23 10:40:00 | 254.35 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-11-24 10:05:00 | 252.25 | 2023-11-24 10:15:00 | 251.04 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-11-24 10:05:00 | 252.25 | 2023-11-24 10:45:00 | 252.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 09:45:00 | 259.90 | 2023-11-29 09:50:00 | 258.70 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2023-12-05 09:35:00 | 270.00 | 2023-12-05 09:55:00 | 271.84 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2023-12-05 09:35:00 | 270.00 | 2023-12-05 10:05:00 | 270.55 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2023-12-07 10:10:00 | 271.60 | 2023-12-07 10:25:00 | 273.17 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-12-07 10:10:00 | 271.60 | 2023-12-07 15:20:00 | 277.40 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2023-12-08 10:05:00 | 281.25 | 2023-12-08 10:25:00 | 283.15 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2023-12-08 10:05:00 | 281.25 | 2023-12-08 11:00:00 | 281.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-12 10:55:00 | 285.70 | 2023-12-12 11:20:00 | 286.82 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-12-13 09:35:00 | 287.00 | 2023-12-13 10:10:00 | 289.09 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2023-12-13 09:35:00 | 287.00 | 2023-12-13 12:20:00 | 288.10 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2023-12-28 10:00:00 | 295.00 | 2023-12-28 10:15:00 | 295.99 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-01-01 10:30:00 | 298.50 | 2024-01-01 11:10:00 | 297.27 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-01-02 09:30:00 | 307.90 | 2024-01-02 09:40:00 | 306.48 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-01-03 09:55:00 | 303.80 | 2024-01-03 11:05:00 | 305.43 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-01-09 10:10:00 | 323.00 | 2024-01-09 10:25:00 | 321.44 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-01-12 09:35:00 | 324.40 | 2024-01-12 09:50:00 | 326.03 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-01-12 09:35:00 | 324.40 | 2024-01-12 11:00:00 | 324.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-18 09:45:00 | 311.25 | 2024-01-18 09:55:00 | 308.72 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2024-01-18 09:45:00 | 311.25 | 2024-01-18 10:00:00 | 311.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-19 09:40:00 | 318.65 | 2024-01-19 10:55:00 | 316.42 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-01-19 09:40:00 | 318.65 | 2024-01-19 11:20:00 | 318.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-23 11:05:00 | 315.00 | 2024-01-23 11:40:00 | 312.81 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-01-23 11:05:00 | 315.00 | 2024-01-23 13:10:00 | 315.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-25 11:05:00 | 313.25 | 2024-01-25 11:35:00 | 314.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-01-29 11:10:00 | 313.00 | 2024-01-29 11:15:00 | 314.02 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-01-30 09:40:00 | 316.00 | 2024-01-30 09:50:00 | 314.76 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-01-31 11:05:00 | 314.10 | 2024-01-31 11:20:00 | 315.09 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-02-05 09:50:00 | 329.80 | 2024-02-05 10:00:00 | 332.14 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-02-05 09:50:00 | 329.80 | 2024-02-05 11:40:00 | 329.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-07 09:35:00 | 329.20 | 2024-02-07 12:40:00 | 325.57 | PARTIAL | 0.50 | 1.10% |
| SELL | retest1 | 2024-02-07 09:35:00 | 329.20 | 2024-02-07 15:20:00 | 323.30 | TARGET_HIT | 0.50 | 1.79% |
| SELL | retest1 | 2024-02-20 10:50:00 | 345.25 | 2024-02-20 11:05:00 | 346.29 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-02-21 10:35:00 | 344.55 | 2024-02-21 13:25:00 | 342.20 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-02-21 10:35:00 | 344.55 | 2024-02-21 15:20:00 | 340.80 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2024-02-28 10:00:00 | 345.90 | 2024-02-28 10:25:00 | 344.36 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-02-28 10:00:00 | 345.90 | 2024-02-28 15:20:00 | 339.10 | TARGET_HIT | 0.50 | 1.97% |
| SELL | retest1 | 2024-03-06 09:30:00 | 349.00 | 2024-03-06 09:35:00 | 350.44 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-03-11 10:30:00 | 343.75 | 2024-03-11 10:45:00 | 341.63 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-03-11 10:30:00 | 343.75 | 2024-03-11 15:20:00 | 337.20 | TARGET_HIT | 0.50 | 1.91% |
| SELL | retest1 | 2024-03-20 09:35:00 | 290.45 | 2024-03-20 10:00:00 | 287.60 | PARTIAL | 0.50 | 0.98% |
| SELL | retest1 | 2024-03-20 09:35:00 | 290.45 | 2024-03-20 12:35:00 | 287.60 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2024-03-21 10:50:00 | 302.40 | 2024-03-21 12:05:00 | 300.89 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-04-04 09:50:00 | 329.05 | 2024-04-04 10:40:00 | 326.81 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-04-04 09:50:00 | 329.05 | 2024-04-04 15:20:00 | 326.70 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2024-04-05 10:30:00 | 331.05 | 2024-04-05 10:50:00 | 329.38 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-04-23 09:30:00 | 359.00 | 2024-04-23 09:35:00 | 357.41 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-04-26 09:55:00 | 368.95 | 2024-04-26 10:00:00 | 366.79 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-04-29 09:35:00 | 361.50 | 2024-04-29 09:45:00 | 359.23 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-04-29 09:35:00 | 361.50 | 2024-04-29 10:00:00 | 361.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-02 09:50:00 | 357.70 | 2024-05-02 10:35:00 | 359.30 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-05-03 10:00:00 | 359.30 | 2024-05-03 10:05:00 | 357.68 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-03 10:00:00 | 359.30 | 2024-05-03 10:15:00 | 359.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-10 09:35:00 | 362.25 | 2024-05-10 09:40:00 | 364.21 | STOP_HIT | 1.00 | -0.54% |
