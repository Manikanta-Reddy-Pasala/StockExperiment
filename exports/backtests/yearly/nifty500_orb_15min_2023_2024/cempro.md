# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (50533 bars)
- **Last close:** 955.20
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
| ENTRY1 | 59 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 9 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 50
- **Target hits / Stop hits / Partials:** 9 / 50 / 19
- **Avg / median % per leg:** 0.17% / -0.29%
- **Sum % (uncompounded):** 13.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 10 | 27.8% | 3 | 26 | 7 | -0.04% | -1.3% |
| BUY @ 2nd Alert (retest1) | 36 | 10 | 27.8% | 3 | 26 | 7 | -0.04% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 18 | 42.9% | 6 | 24 | 12 | 0.35% | 14.5% |
| SELL @ 2nd Alert (retest1) | 42 | 18 | 42.9% | 6 | 24 | 12 | 0.35% | 14.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 78 | 28 | 35.9% | 9 | 50 | 19 | 0.17% | 13.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:40:00 | 133.00 | 130.99 | 0.00 | ORB-long ORB[128.10,129.60] vol=2.0x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 10:55:00 | 134.31 | 132.53 | 0.00 | T1 1.5R @ 134.31 |
| Target hit | 2023-05-12 13:40:00 | 133.35 | 133.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2023-05-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 09:50:00 | 149.35 | 148.19 | 0.00 | ORB-long ORB[147.15,148.20] vol=2.8x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-30 10:05:00 | 150.36 | 149.15 | 0.00 | T1 1.5R @ 150.36 |
| Target hit | 2023-05-30 10:45:00 | 152.55 | 152.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2023-06-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:55:00 | 167.00 | 166.02 | 0.00 | ORB-long ORB[163.70,166.20] vol=1.5x ATR=0.86 |
| Stop hit — per-position SL triggered | 2023-06-06 10:10:00 | 166.14 | 166.10 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-07 09:55:00 | 165.50 | 166.33 | 0.00 | ORB-short ORB[165.80,167.80] vol=1.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-06-07 11:25:00 | 166.21 | 165.92 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 09:50:00 | 163.00 | 163.72 | 0.00 | ORB-short ORB[163.55,165.45] vol=2.8x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-06-15 10:05:00 | 163.57 | 163.69 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:40:00 | 170.35 | 168.58 | 0.00 | ORB-long ORB[167.10,169.60] vol=2.0x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-06-19 09:45:00 | 169.55 | 168.68 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 10:50:00 | 165.30 | 166.80 | 0.00 | ORB-short ORB[167.20,168.90] vol=4.9x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 11:20:00 | 164.29 | 166.06 | 0.00 | T1 1.5R @ 164.29 |
| Target hit | 2023-06-22 15:20:00 | 163.20 | 164.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2023-06-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:55:00 | 165.00 | 164.26 | 0.00 | ORB-long ORB[163.35,164.80] vol=3.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-06-27 10:00:00 | 164.43 | 164.07 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:35:00 | 163.70 | 162.89 | 0.00 | ORB-long ORB[160.50,162.95] vol=2.1x ATR=0.55 |
| Stop hit — per-position SL triggered | 2023-07-05 09:40:00 | 163.15 | 162.97 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-07-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:30:00 | 167.45 | 166.32 | 0.00 | ORB-long ORB[165.15,166.95] vol=2.1x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-07-07 10:00:00 | 166.60 | 166.59 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:35:00 | 167.10 | 166.03 | 0.00 | ORB-long ORB[164.55,166.55] vol=2.1x ATR=0.84 |
| Stop hit — per-position SL triggered | 2023-07-11 10:00:00 | 166.26 | 166.26 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 11:15:00 | 176.45 | 177.92 | 0.00 | ORB-short ORB[177.60,180.25] vol=2.5x ATR=0.59 |
| Stop hit — per-position SL triggered | 2023-07-19 11:20:00 | 177.04 | 177.88 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:30:00 | 186.90 | 186.24 | 0.00 | ORB-long ORB[184.35,185.80] vol=14.1x ATR=0.97 |
| Stop hit — per-position SL triggered | 2023-07-31 09:35:00 | 185.93 | 186.23 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-08-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 09:45:00 | 187.25 | 186.20 | 0.00 | ORB-long ORB[184.70,186.80] vol=3.8x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 09:50:00 | 188.49 | 186.90 | 0.00 | T1 1.5R @ 188.49 |
| Stop hit — per-position SL triggered | 2023-08-02 10:10:00 | 187.25 | 187.05 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-08-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 09:55:00 | 183.30 | 181.55 | 0.00 | ORB-long ORB[179.45,181.80] vol=3.0x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-08-04 10:10:00 | 182.38 | 181.78 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-08-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:30:00 | 206.20 | 204.05 | 0.00 | ORB-long ORB[201.15,204.00] vol=2.3x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 09:35:00 | 207.83 | 205.29 | 0.00 | T1 1.5R @ 207.83 |
| Stop hit — per-position SL triggered | 2023-08-17 09:45:00 | 206.20 | 205.61 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-08-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 09:55:00 | 206.80 | 205.96 | 0.00 | ORB-long ORB[204.65,206.70] vol=2.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-08-18 10:35:00 | 205.93 | 206.09 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-08-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 10:40:00 | 206.45 | 209.01 | 0.00 | ORB-short ORB[208.10,210.85] vol=3.6x ATR=0.95 |
| Stop hit — per-position SL triggered | 2023-08-23 10:45:00 | 207.40 | 208.70 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-08-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:30:00 | 211.55 | 210.44 | 0.00 | ORB-long ORB[207.30,210.40] vol=5.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-08-24 09:40:00 | 210.43 | 210.50 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:40:00 | 214.75 | 212.72 | 0.00 | ORB-long ORB[210.90,214.10] vol=1.7x ATR=1.03 |
| Stop hit — per-position SL triggered | 2023-08-28 09:45:00 | 213.72 | 212.95 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 10:45:00 | 212.95 | 211.02 | 0.00 | ORB-long ORB[210.00,212.35] vol=2.3x ATR=0.93 |
| Stop hit — per-position SL triggered | 2023-08-29 10:50:00 | 212.02 | 211.46 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:35:00 | 213.35 | 211.74 | 0.00 | ORB-long ORB[210.00,211.95] vol=2.7x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 10:20:00 | 214.77 | 213.11 | 0.00 | T1 1.5R @ 214.77 |
| Stop hit — per-position SL triggered | 2023-08-30 10:45:00 | 213.35 | 213.32 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 10:50:00 | 212.90 | 214.49 | 0.00 | ORB-short ORB[213.15,216.00] vol=1.8x ATR=0.72 |
| Stop hit — per-position SL triggered | 2023-08-31 13:30:00 | 213.62 | 213.94 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-09-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 09:50:00 | 217.60 | 219.14 | 0.00 | ORB-short ORB[218.05,220.40] vol=2.2x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 10:00:00 | 215.80 | 218.71 | 0.00 | T1 1.5R @ 215.80 |
| Stop hit — per-position SL triggered | 2023-09-21 10:10:00 | 217.60 | 218.55 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 09:30:00 | 217.45 | 215.91 | 0.00 | ORB-long ORB[213.70,216.70] vol=2.2x ATR=0.89 |
| Stop hit — per-position SL triggered | 2023-09-26 09:35:00 | 216.56 | 216.01 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-09-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 10:30:00 | 217.15 | 215.83 | 0.00 | ORB-long ORB[214.40,216.80] vol=1.7x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 10:40:00 | 218.46 | 216.38 | 0.00 | T1 1.5R @ 218.46 |
| Target hit | 2023-09-27 15:20:00 | 220.35 | 219.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2023-09-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 10:55:00 | 218.55 | 219.19 | 0.00 | ORB-short ORB[219.15,222.35] vol=2.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2023-09-28 12:10:00 | 219.32 | 219.01 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-10-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 10:35:00 | 214.05 | 215.47 | 0.00 | ORB-short ORB[214.75,217.20] vol=3.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-10-12 10:40:00 | 214.67 | 215.44 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 09:35:00 | 214.35 | 215.21 | 0.00 | ORB-short ORB[214.40,217.60] vol=1.7x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 10:10:00 | 213.31 | 214.20 | 0.00 | T1 1.5R @ 213.31 |
| Target hit | 2023-10-16 10:35:00 | 214.15 | 213.72 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2023-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:30:00 | 219.30 | 217.64 | 0.00 | ORB-long ORB[216.70,218.40] vol=2.4x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-10-17 09:35:00 | 218.45 | 217.95 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-10-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 09:30:00 | 217.85 | 218.78 | 0.00 | ORB-short ORB[218.05,220.75] vol=2.0x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 09:35:00 | 216.35 | 218.29 | 0.00 | T1 1.5R @ 216.35 |
| Stop hit — per-position SL triggered | 2023-10-19 10:05:00 | 217.85 | 217.81 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-10-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 10:35:00 | 217.95 | 218.87 | 0.00 | ORB-short ORB[218.35,220.30] vol=2.1x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-10-20 10:50:00 | 218.59 | 218.85 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 11:15:00 | 202.90 | 205.13 | 0.00 | ORB-short ORB[205.00,207.60] vol=2.2x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-10-31 11:40:00 | 203.77 | 204.47 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-11-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 10:05:00 | 217.20 | 214.95 | 0.00 | ORB-long ORB[212.55,215.75] vol=3.3x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-11-07 10:30:00 | 216.24 | 215.58 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-11-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:35:00 | 276.90 | 274.16 | 0.00 | ORB-long ORB[271.55,274.45] vol=6.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-11-23 09:40:00 | 275.44 | 274.29 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:30:00 | 271.50 | 273.29 | 0.00 | ORB-short ORB[272.50,275.35] vol=1.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2023-11-24 09:40:00 | 272.39 | 273.02 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-11-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:45:00 | 268.50 | 270.34 | 0.00 | ORB-short ORB[269.75,273.20] vol=2.0x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 09:55:00 | 266.95 | 269.64 | 0.00 | T1 1.5R @ 266.95 |
| Stop hit — per-position SL triggered | 2023-11-30 10:05:00 | 268.50 | 269.35 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-12-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 10:45:00 | 285.35 | 283.49 | 0.00 | ORB-long ORB[281.70,283.90] vol=4.4x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-12-11 12:00:00 | 284.37 | 284.32 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-12-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 09:40:00 | 292.85 | 290.74 | 0.00 | ORB-long ORB[287.20,291.20] vol=2.8x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-12-13 09:55:00 | 291.66 | 291.32 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-12-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 10:35:00 | 296.50 | 295.65 | 0.00 | ORB-long ORB[293.10,295.90] vol=1.7x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 10:40:00 | 298.22 | 296.07 | 0.00 | T1 1.5R @ 298.22 |
| Stop hit — per-position SL triggered | 2023-12-15 10:50:00 | 296.50 | 296.11 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 11:15:00 | 294.40 | 295.77 | 0.00 | ORB-short ORB[295.25,299.00] vol=3.4x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 11:30:00 | 293.09 | 295.29 | 0.00 | T1 1.5R @ 293.09 |
| Target hit | 2023-12-20 15:20:00 | 268.30 | 278.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2023-12-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 10:30:00 | 282.80 | 285.46 | 0.00 | ORB-short ORB[285.55,289.50] vol=1.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2023-12-28 10:40:00 | 284.19 | 285.20 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-12-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 10:50:00 | 288.55 | 291.27 | 0.00 | ORB-short ORB[289.50,293.45] vol=2.2x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 12:30:00 | 286.66 | 290.28 | 0.00 | T1 1.5R @ 286.66 |
| Target hit | 2023-12-29 15:20:00 | 284.30 | 288.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 286.35 | 289.01 | 0.00 | ORB-short ORB[287.55,290.85] vol=1.7x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-01-02 10:10:00 | 287.98 | 288.80 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-01-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 10:45:00 | 288.15 | 285.93 | 0.00 | ORB-long ORB[283.20,286.70] vol=1.9x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-01-03 11:30:00 | 286.96 | 286.63 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-01-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 11:00:00 | 296.05 | 297.06 | 0.00 | ORB-short ORB[297.00,299.95] vol=1.8x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-01-09 11:05:00 | 297.01 | 297.00 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-01-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 10:30:00 | 305.75 | 306.89 | 0.00 | ORB-short ORB[306.40,309.95] vol=1.5x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 11:20:00 | 304.08 | 306.36 | 0.00 | T1 1.5R @ 304.08 |
| Target hit | 2024-01-12 15:20:00 | 303.00 | 304.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2024-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:50:00 | 314.75 | 315.70 | 0.00 | ORB-short ORB[315.00,317.85] vol=2.2x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-01-20 12:30:00 | 316.25 | 316.29 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-01-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 09:30:00 | 319.30 | 320.70 | 0.00 | ORB-short ORB[319.80,322.45] vol=2.2x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-01-25 09:35:00 | 320.82 | 320.78 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-01-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-31 09:55:00 | 313.40 | 315.89 | 0.00 | ORB-short ORB[314.15,317.90] vol=2.2x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-01-31 10:05:00 | 315.06 | 315.76 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-02-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 09:40:00 | 321.90 | 323.44 | 0.00 | ORB-short ORB[322.20,326.80] vol=2.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 11:10:00 | 318.93 | 322.33 | 0.00 | T1 1.5R @ 318.93 |
| Stop hit — per-position SL triggered | 2024-02-01 11:55:00 | 321.90 | 322.00 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-04-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:35:00 | 347.50 | 344.56 | 0.00 | ORB-long ORB[340.85,344.35] vol=2.2x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-04-03 09:45:00 | 345.97 | 345.08 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:30:00 | 338.90 | 340.82 | 0.00 | ORB-short ORB[339.10,343.80] vol=2.1x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 09:45:00 | 336.67 | 339.30 | 0.00 | T1 1.5R @ 336.67 |
| Target hit | 2024-04-04 15:20:00 | 333.45 | 334.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2024-04-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:30:00 | 330.00 | 332.76 | 0.00 | ORB-short ORB[333.00,337.45] vol=4.2x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-05 09:35:00 | 327.54 | 330.84 | 0.00 | T1 1.5R @ 327.54 |
| Stop hit — per-position SL triggered | 2024-04-05 09:40:00 | 330.00 | 330.69 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 10:15:00 | 362.85 | 364.79 | 0.00 | ORB-short ORB[363.30,368.65] vol=1.7x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 10:40:00 | 361.07 | 364.36 | 0.00 | T1 1.5R @ 361.07 |
| Stop hit — per-position SL triggered | 2024-04-23 10:45:00 | 362.85 | 364.31 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:35:00 | 387.00 | 381.71 | 0.00 | ORB-long ORB[375.20,379.80] vol=2.8x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-04-25 09:45:00 | 384.40 | 383.60 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 10:15:00 | 372.00 | 375.76 | 0.00 | ORB-short ORB[375.65,379.75] vol=1.6x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-04-26 10:25:00 | 373.59 | 375.62 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 09:45:00 | 389.50 | 384.99 | 0.00 | ORB-long ORB[381.20,386.50] vol=3.3x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-05-07 09:50:00 | 387.29 | 385.45 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-05-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:40:00 | 374.65 | 377.92 | 0.00 | ORB-short ORB[378.90,382.90] vol=4.2x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-05-09 09:45:00 | 376.26 | 377.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:40:00 | 133.00 | 2023-05-12 10:55:00 | 134.31 | PARTIAL | 0.50 | 0.98% |
| BUY | retest1 | 2023-05-12 10:40:00 | 133.00 | 2023-05-12 13:40:00 | 133.35 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2023-05-30 09:50:00 | 149.35 | 2023-05-30 10:05:00 | 150.36 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2023-05-30 09:50:00 | 149.35 | 2023-05-30 10:45:00 | 152.55 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2023-06-06 09:55:00 | 167.00 | 2023-06-06 10:10:00 | 166.14 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2023-06-07 09:55:00 | 165.50 | 2023-06-07 11:25:00 | 166.21 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-06-15 09:50:00 | 163.00 | 2023-06-15 10:05:00 | 163.57 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-06-19 09:40:00 | 170.35 | 2023-06-19 09:45:00 | 169.55 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2023-06-22 10:50:00 | 165.30 | 2023-06-22 11:20:00 | 164.29 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2023-06-22 10:50:00 | 165.30 | 2023-06-22 15:20:00 | 163.20 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2023-06-27 09:55:00 | 165.00 | 2023-06-27 10:00:00 | 164.43 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-07-05 09:35:00 | 163.70 | 2023-07-05 09:40:00 | 163.15 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-07-07 09:30:00 | 167.45 | 2023-07-07 10:00:00 | 166.60 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2023-07-11 09:35:00 | 167.10 | 2023-07-11 10:00:00 | 166.26 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-07-19 11:15:00 | 176.45 | 2023-07-19 11:20:00 | 177.04 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-07-31 09:30:00 | 186.90 | 2023-07-31 09:35:00 | 185.93 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2023-08-02 09:45:00 | 187.25 | 2023-08-02 09:50:00 | 188.49 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2023-08-02 09:45:00 | 187.25 | 2023-08-02 10:10:00 | 187.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-04 09:55:00 | 183.30 | 2023-08-04 10:10:00 | 182.38 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2023-08-17 09:30:00 | 206.20 | 2023-08-17 09:35:00 | 207.83 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2023-08-17 09:30:00 | 206.20 | 2023-08-17 09:45:00 | 206.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-18 09:55:00 | 206.80 | 2023-08-18 10:35:00 | 205.93 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-08-23 10:40:00 | 206.45 | 2023-08-23 10:45:00 | 207.40 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2023-08-24 09:30:00 | 211.55 | 2023-08-24 09:40:00 | 210.43 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2023-08-28 09:40:00 | 214.75 | 2023-08-28 09:45:00 | 213.72 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2023-08-29 10:45:00 | 212.95 | 2023-08-29 10:50:00 | 212.02 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-08-30 09:35:00 | 213.35 | 2023-08-30 10:20:00 | 214.77 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2023-08-30 09:35:00 | 213.35 | 2023-08-30 10:45:00 | 213.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-31 10:50:00 | 212.90 | 2023-08-31 13:30:00 | 213.62 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-09-21 09:50:00 | 217.60 | 2023-09-21 10:00:00 | 215.80 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2023-09-21 09:50:00 | 217.60 | 2023-09-21 10:10:00 | 217.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-26 09:30:00 | 217.45 | 2023-09-26 09:35:00 | 216.56 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-09-27 10:30:00 | 217.15 | 2023-09-27 10:40:00 | 218.46 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2023-09-27 10:30:00 | 217.15 | 2023-09-27 15:20:00 | 220.35 | TARGET_HIT | 0.50 | 1.47% |
| SELL | retest1 | 2023-09-28 10:55:00 | 218.55 | 2023-09-28 12:10:00 | 219.32 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-10-12 10:35:00 | 214.05 | 2023-10-12 10:40:00 | 214.67 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-16 09:35:00 | 214.35 | 2023-10-16 10:10:00 | 213.31 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-10-16 09:35:00 | 214.35 | 2023-10-16 10:35:00 | 214.15 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2023-10-17 09:30:00 | 219.30 | 2023-10-17 09:35:00 | 218.45 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-10-19 09:30:00 | 217.85 | 2023-10-19 09:35:00 | 216.35 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2023-10-19 09:30:00 | 217.85 | 2023-10-19 10:05:00 | 217.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-20 10:35:00 | 217.95 | 2023-10-20 10:50:00 | 218.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-31 11:15:00 | 202.90 | 2023-10-31 11:40:00 | 203.77 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-11-07 10:05:00 | 217.20 | 2023-11-07 10:30:00 | 216.24 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-11-23 09:35:00 | 276.90 | 2023-11-23 09:40:00 | 275.44 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2023-11-24 09:30:00 | 271.50 | 2023-11-24 09:40:00 | 272.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-11-30 09:45:00 | 268.50 | 2023-11-30 09:55:00 | 266.95 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2023-11-30 09:45:00 | 268.50 | 2023-11-30 10:05:00 | 268.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-11 10:45:00 | 285.35 | 2023-12-11 12:00:00 | 284.37 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-12-13 09:40:00 | 292.85 | 2023-12-13 09:55:00 | 291.66 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-12-15 10:35:00 | 296.50 | 2023-12-15 10:40:00 | 298.22 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-12-15 10:35:00 | 296.50 | 2023-12-15 10:50:00 | 296.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-20 11:15:00 | 294.40 | 2023-12-20 11:30:00 | 293.09 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-12-20 11:15:00 | 294.40 | 2023-12-20 15:20:00 | 268.30 | TARGET_HIT | 0.50 | 8.87% |
| SELL | retest1 | 2023-12-28 10:30:00 | 282.80 | 2023-12-28 10:40:00 | 284.19 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2023-12-29 10:50:00 | 288.55 | 2023-12-29 12:30:00 | 286.66 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2023-12-29 10:50:00 | 288.55 | 2023-12-29 15:20:00 | 284.30 | TARGET_HIT | 0.50 | 1.47% |
| SELL | retest1 | 2024-01-02 09:55:00 | 286.35 | 2024-01-02 10:10:00 | 287.98 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-01-03 10:45:00 | 288.15 | 2024-01-03 11:30:00 | 286.96 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-01-09 11:00:00 | 296.05 | 2024-01-09 11:05:00 | 297.01 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-01-12 10:30:00 | 305.75 | 2024-01-12 11:20:00 | 304.08 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-01-12 10:30:00 | 305.75 | 2024-01-12 15:20:00 | 303.00 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2024-01-20 10:50:00 | 314.75 | 2024-01-20 12:30:00 | 316.25 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-01-25 09:30:00 | 319.30 | 2024-01-25 09:35:00 | 320.82 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-01-31 09:55:00 | 313.40 | 2024-01-31 10:05:00 | 315.06 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-02-01 09:40:00 | 321.90 | 2024-02-01 11:10:00 | 318.93 | PARTIAL | 0.50 | 0.92% |
| SELL | retest1 | 2024-02-01 09:40:00 | 321.90 | 2024-02-01 11:55:00 | 321.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-03 09:35:00 | 347.50 | 2024-04-03 09:45:00 | 345.97 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-04-04 09:30:00 | 338.90 | 2024-04-04 09:45:00 | 336.67 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-04-04 09:30:00 | 338.90 | 2024-04-04 15:20:00 | 333.45 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2024-04-05 09:30:00 | 330.00 | 2024-04-05 09:35:00 | 327.54 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-04-05 09:30:00 | 330.00 | 2024-04-05 09:40:00 | 330.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-23 10:15:00 | 362.85 | 2024-04-23 10:40:00 | 361.07 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-04-23 10:15:00 | 362.85 | 2024-04-23 10:45:00 | 362.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-25 09:35:00 | 387.00 | 2024-04-25 09:45:00 | 384.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2024-04-26 10:15:00 | 372.00 | 2024-04-26 10:25:00 | 373.59 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-07 09:45:00 | 389.50 | 2024-05-07 09:50:00 | 387.29 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-05-09 09:40:00 | 374.65 | 2024-05-09 09:45:00 | 376.26 | STOP_HIT | 1.00 | -0.43% |
