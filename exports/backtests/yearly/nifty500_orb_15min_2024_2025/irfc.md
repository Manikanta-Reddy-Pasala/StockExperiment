# Indian Railway Finance Corporation Ltd. (IRFC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 106.02
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
| ENTRY1 | 43 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 12 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 31
- **Target hits / Stop hits / Partials:** 12 / 31 / 18
- **Avg / median % per leg:** 0.29% / 0.00%
- **Sum % (uncompounded):** 17.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 14 | 43.8% | 5 | 18 | 9 | 0.33% | 10.7% |
| BUY @ 2nd Alert (retest1) | 32 | 14 | 43.8% | 5 | 18 | 9 | 0.33% | 10.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 29 | 16 | 55.2% | 7 | 13 | 9 | 0.24% | 7.0% |
| SELL @ 2nd Alert (retest1) | 29 | 16 | 55.2% | 7 | 13 | 9 | 0.24% | 7.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 61 | 30 | 49.2% | 12 | 31 | 18 | 0.29% | 17.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:10:00 | 158.10 | 156.94 | 0.00 | ORB-long ORB[155.60,157.35] vol=3.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-05-16 10:20:00 | 157.39 | 157.12 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:40:00 | 176.45 | 178.47 | 0.00 | ORB-short ORB[177.85,179.85] vol=1.7x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:00:00 | 175.11 | 177.26 | 0.00 | T1 1.5R @ 175.11 |
| Target hit | 2024-05-31 11:55:00 | 175.75 | 175.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2024-06-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:40:00 | 178.67 | 177.47 | 0.00 | ORB-long ORB[176.35,177.83] vol=3.4x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-06-12 09:45:00 | 178.03 | 177.55 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:50:00 | 175.99 | 174.68 | 0.00 | ORB-long ORB[173.20,175.80] vol=2.5x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-06-14 10:05:00 | 175.40 | 175.02 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 177.60 | 176.50 | 0.00 | ORB-long ORB[175.45,176.95] vol=3.7x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 09:35:00 | 178.44 | 176.95 | 0.00 | T1 1.5R @ 178.44 |
| Stop hit — per-position SL triggered | 2024-06-18 09:40:00 | 177.60 | 177.04 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 175.65 | 176.84 | 0.00 | ORB-short ORB[176.85,178.60] vol=2.3x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 175.99 | 176.81 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:35:00 | 177.51 | 176.86 | 0.00 | ORB-long ORB[175.50,177.40] vol=2.3x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-06-27 09:45:00 | 176.98 | 176.93 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:45:00 | 187.00 | 186.06 | 0.00 | ORB-long ORB[184.67,186.87] vol=2.6x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 09:55:00 | 188.29 | 186.58 | 0.00 | T1 1.5R @ 188.29 |
| Target hit | 2024-07-29 15:20:00 | 197.10 | 190.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2024-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:30:00 | 192.95 | 194.25 | 0.00 | ORB-short ORB[193.82,195.65] vol=1.7x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:35:00 | 192.18 | 194.14 | 0.00 | T1 1.5R @ 192.18 |
| Target hit | 2024-08-01 15:20:00 | 189.98 | 191.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 182.06 | 181.21 | 0.00 | ORB-long ORB[180.33,182.03] vol=1.8x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-08-16 09:35:00 | 181.37 | 181.24 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 181.55 | 180.86 | 0.00 | ORB-long ORB[179.95,181.35] vol=1.6x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-08-22 09:40:00 | 181.08 | 180.92 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 182.00 | 182.66 | 0.00 | ORB-short ORB[182.06,184.20] vol=1.9x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:55:00 | 181.24 | 182.37 | 0.00 | T1 1.5R @ 181.24 |
| Stop hit — per-position SL triggered | 2024-08-26 12:25:00 | 182.00 | 181.80 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:55:00 | 182.36 | 181.78 | 0.00 | ORB-long ORB[180.66,182.20] vol=1.8x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-08-28 10:05:00 | 181.83 | 181.80 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:30:00 | 179.83 | 180.54 | 0.00 | ORB-short ORB[180.27,181.78] vol=1.6x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:40:00 | 179.19 | 180.44 | 0.00 | T1 1.5R @ 179.19 |
| Stop hit — per-position SL triggered | 2024-08-29 10:45:00 | 179.83 | 180.41 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 178.89 | 179.71 | 0.00 | ORB-short ORB[179.50,180.85] vol=2.0x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-08-30 09:40:00 | 179.39 | 179.62 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:10:00 | 174.82 | 175.64 | 0.00 | ORB-short ORB[175.27,176.85] vol=1.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 10:15:00 | 174.26 | 175.45 | 0.00 | T1 1.5R @ 174.26 |
| Target hit | 2024-09-05 15:20:00 | 173.24 | 174.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2024-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:35:00 | 171.59 | 172.31 | 0.00 | ORB-short ORB[171.61,174.17] vol=2.2x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:45:00 | 170.87 | 171.99 | 0.00 | T1 1.5R @ 170.87 |
| Target hit | 2024-09-06 11:05:00 | 170.51 | 170.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2024-09-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:55:00 | 164.85 | 165.92 | 0.00 | ORB-short ORB[165.85,167.45] vol=1.6x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-09-16 10:20:00 | 165.30 | 165.70 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 157.02 | 157.52 | 0.00 | ORB-short ORB[157.11,158.70] vol=1.8x ATR=0.47 |
| Target hit | 2024-09-25 15:20:00 | 156.96 | 157.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:45:00 | 154.92 | 155.68 | 0.00 | ORB-short ORB[155.56,156.90] vol=2.0x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-09-26 10:00:00 | 155.32 | 155.60 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:15:00 | 158.13 | 157.52 | 0.00 | ORB-long ORB[156.85,158.00] vol=1.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-09-27 11:05:00 | 157.69 | 157.72 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:40:00 | 151.93 | 151.07 | 0.00 | ORB-long ORB[150.40,151.79] vol=1.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 09:50:00 | 152.67 | 151.64 | 0.00 | T1 1.5R @ 152.67 |
| Target hit | 2024-10-11 10:25:00 | 153.17 | 153.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2024-10-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:00:00 | 152.40 | 151.81 | 0.00 | ORB-long ORB[150.63,152.20] vol=2.2x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-10-16 10:05:00 | 151.86 | 151.83 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 136.96 | 137.94 | 0.00 | ORB-short ORB[137.41,139.28] vol=1.8x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:40:00 | 136.16 | 137.54 | 0.00 | T1 1.5R @ 136.16 |
| Target hit | 2024-10-25 15:20:00 | 134.61 | 135.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:40:00 | 148.56 | 147.13 | 0.00 | ORB-long ORB[145.84,147.14] vol=3.2x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 09:45:00 | 149.37 | 148.11 | 0.00 | T1 1.5R @ 149.37 |
| Target hit | 2024-11-27 11:45:00 | 149.59 | 149.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2024-12-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 09:50:00 | 148.10 | 149.05 | 0.00 | ORB-short ORB[148.11,150.21] vol=2.0x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 11:30:00 | 147.24 | 148.48 | 0.00 | T1 1.5R @ 147.24 |
| Target hit | 2024-12-02 15:20:00 | 147.20 | 148.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 152.91 | 151.71 | 0.00 | ORB-long ORB[150.55,151.55] vol=5.2x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-12-06 09:50:00 | 152.38 | 152.16 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-12-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:55:00 | 160.39 | 158.45 | 0.00 | ORB-long ORB[156.55,158.45] vol=5.6x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:00:00 | 161.38 | 159.38 | 0.00 | T1 1.5R @ 161.38 |
| Stop hit — per-position SL triggered | 2024-12-11 10:05:00 | 160.39 | 159.58 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 162.99 | 165.13 | 0.00 | ORB-short ORB[164.50,166.90] vol=1.6x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-12-12 10:00:00 | 163.80 | 164.69 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 09:55:00 | 154.25 | 155.44 | 0.00 | ORB-short ORB[154.34,156.29] vol=2.0x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-12-18 10:05:00 | 154.98 | 155.36 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 146.73 | 147.69 | 0.00 | ORB-short ORB[147.30,148.60] vol=2.1x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-12-24 09:35:00 | 147.23 | 147.59 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:45:00 | 146.41 | 148.15 | 0.00 | ORB-short ORB[147.75,149.50] vol=3.0x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-12-26 09:55:00 | 147.08 | 147.89 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:50:00 | 147.25 | 145.91 | 0.00 | ORB-long ORB[144.80,146.22] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 146.71 | 146.45 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 151.39 | 150.86 | 0.00 | ORB-long ORB[150.00,151.35] vol=2.0x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:35:00 | 152.06 | 151.06 | 0.00 | T1 1.5R @ 152.06 |
| Stop hit — per-position SL triggered | 2025-01-02 11:15:00 | 151.39 | 151.54 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-01-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:55:00 | 142.52 | 144.71 | 0.00 | ORB-short ORB[144.60,146.60] vol=1.6x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 11:25:00 | 141.45 | 143.62 | 0.00 | T1 1.5R @ 141.45 |
| Stop hit — per-position SL triggered | 2025-01-21 11:35:00 | 142.52 | 143.57 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:30:00 | 140.95 | 139.82 | 0.00 | ORB-long ORB[138.45,140.20] vol=1.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-01-29 10:20:00 | 140.19 | 140.50 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:35:00 | 144.18 | 142.95 | 0.00 | ORB-long ORB[141.85,143.25] vol=2.5x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-01-30 09:40:00 | 143.72 | 143.14 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 136.94 | 137.76 | 0.00 | ORB-short ORB[137.15,138.70] vol=1.7x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-02-06 09:40:00 | 137.33 | 137.67 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-03-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:05:00 | 121.20 | 120.02 | 0.00 | ORB-long ORB[118.83,120.20] vol=2.4x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-03-18 10:10:00 | 120.83 | 120.08 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:40:00 | 124.06 | 123.48 | 0.00 | ORB-long ORB[122.25,123.87] vol=1.6x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 09:50:00 | 124.75 | 123.71 | 0.00 | T1 1.5R @ 124.75 |
| Target hit | 2025-03-19 15:20:00 | 128.20 | 126.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-04-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:25:00 | 127.12 | 126.36 | 0.00 | ORB-long ORB[125.00,126.34] vol=1.8x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 11:35:00 | 127.63 | 126.64 | 0.00 | T1 1.5R @ 127.63 |
| Stop hit — per-position SL triggered | 2025-04-15 11:50:00 | 127.12 | 126.67 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:50:00 | 130.54 | 129.67 | 0.00 | ORB-long ORB[128.58,130.34] vol=1.9x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:00:00 | 131.12 | 130.00 | 0.00 | T1 1.5R @ 131.12 |
| Target hit | 2025-04-21 15:20:00 | 131.44 | 130.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 131.97 | 132.87 | 0.00 | ORB-short ORB[132.60,133.56] vol=2.7x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 132.38 | 132.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:10:00 | 158.10 | 2024-05-16 10:20:00 | 157.39 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-05-31 09:40:00 | 176.45 | 2024-05-31 10:00:00 | 175.11 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-05-31 09:40:00 | 176.45 | 2024-05-31 11:55:00 | 175.75 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-12 09:40:00 | 178.67 | 2024-06-12 09:45:00 | 178.03 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-14 09:50:00 | 175.99 | 2024-06-14 10:05:00 | 175.40 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-18 09:30:00 | 177.60 | 2024-06-18 09:35:00 | 178.44 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-18 09:30:00 | 177.60 | 2024-06-18 09:40:00 | 177.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 11:15:00 | 175.65 | 2024-06-25 11:20:00 | 175.99 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-06-27 09:35:00 | 177.51 | 2024-06-27 09:45:00 | 176.98 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-29 09:45:00 | 187.00 | 2024-07-29 09:55:00 | 188.29 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-07-29 09:45:00 | 187.00 | 2024-07-29 15:20:00 | 197.10 | TARGET_HIT | 0.50 | 5.40% |
| SELL | retest1 | 2024-08-01 10:30:00 | 192.95 | 2024-08-01 10:35:00 | 192.18 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-08-01 10:30:00 | 192.95 | 2024-08-01 15:20:00 | 189.98 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2024-08-16 09:30:00 | 182.06 | 2024-08-16 09:35:00 | 181.37 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-22 09:30:00 | 181.55 | 2024-08-22 09:40:00 | 181.08 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-26 09:30:00 | 182.00 | 2024-08-26 09:55:00 | 181.24 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-08-26 09:30:00 | 182.00 | 2024-08-26 12:25:00 | 182.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-28 09:55:00 | 182.36 | 2024-08-28 10:05:00 | 181.83 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-29 10:30:00 | 179.83 | 2024-08-29 10:40:00 | 179.19 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-29 10:30:00 | 179.83 | 2024-08-29 10:45:00 | 179.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 09:30:00 | 178.89 | 2024-08-30 09:40:00 | 179.39 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-05 10:10:00 | 174.82 | 2024-09-05 10:15:00 | 174.26 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-09-05 10:10:00 | 174.82 | 2024-09-05 15:20:00 | 173.24 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2024-09-06 09:35:00 | 171.59 | 2024-09-06 09:45:00 | 170.87 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-06 09:35:00 | 171.59 | 2024-09-06 11:05:00 | 170.51 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2024-09-16 09:55:00 | 164.85 | 2024-09-16 10:20:00 | 165.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-25 09:30:00 | 157.02 | 2024-09-25 15:20:00 | 156.96 | TARGET_HIT | 1.00 | 0.04% |
| SELL | retest1 | 2024-09-26 09:45:00 | 154.92 | 2024-09-26 10:00:00 | 155.32 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-27 10:15:00 | 158.13 | 2024-09-27 11:05:00 | 157.69 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-11 09:40:00 | 151.93 | 2024-10-11 09:50:00 | 152.67 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-10-11 09:40:00 | 151.93 | 2024-10-11 10:25:00 | 153.17 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2024-10-16 10:00:00 | 152.40 | 2024-10-16 10:05:00 | 151.86 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-25 09:30:00 | 136.96 | 2024-10-25 09:40:00 | 136.16 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-10-25 09:30:00 | 136.96 | 2024-10-25 15:20:00 | 134.61 | TARGET_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2024-11-27 09:40:00 | 148.56 | 2024-11-27 09:45:00 | 149.37 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-11-27 09:40:00 | 148.56 | 2024-11-27 11:45:00 | 149.59 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2024-12-02 09:50:00 | 148.10 | 2024-12-02 11:30:00 | 147.24 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-12-02 09:50:00 | 148.10 | 2024-12-02 15:20:00 | 147.20 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-12-06 09:35:00 | 152.91 | 2024-12-06 09:50:00 | 152.38 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-11 09:55:00 | 160.39 | 2024-12-11 10:00:00 | 161.38 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-12-11 09:55:00 | 160.39 | 2024-12-11 10:05:00 | 160.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 09:35:00 | 162.99 | 2024-12-12 10:00:00 | 163.80 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-12-18 09:55:00 | 154.25 | 2024-12-18 10:05:00 | 154.98 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-12-24 09:30:00 | 146.73 | 2024-12-24 09:35:00 | 147.23 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-26 09:45:00 | 146.41 | 2024-12-26 09:55:00 | 147.08 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-12-30 09:50:00 | 147.25 | 2024-12-30 10:05:00 | 146.71 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-02 09:30:00 | 151.39 | 2025-01-02 09:35:00 | 152.06 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-02 09:30:00 | 151.39 | 2025-01-02 11:15:00 | 151.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 09:55:00 | 142.52 | 2025-01-21 11:25:00 | 141.45 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2025-01-21 09:55:00 | 142.52 | 2025-01-21 11:35:00 | 142.52 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-29 09:30:00 | 140.95 | 2025-01-29 10:20:00 | 140.19 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-01-30 09:35:00 | 144.18 | 2025-01-30 09:40:00 | 143.72 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-06 09:30:00 | 136.94 | 2025-02-06 09:40:00 | 137.33 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-18 10:05:00 | 121.20 | 2025-03-18 10:10:00 | 120.83 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-19 09:40:00 | 124.06 | 2025-03-19 09:50:00 | 124.75 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-03-19 09:40:00 | 124.06 | 2025-03-19 15:20:00 | 128.20 | TARGET_HIT | 0.50 | 3.34% |
| BUY | retest1 | 2025-04-15 10:25:00 | 127.12 | 2025-04-15 11:35:00 | 127.63 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-04-15 10:25:00 | 127.12 | 2025-04-15 11:50:00 | 127.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:50:00 | 130.54 | 2025-04-21 10:00:00 | 131.12 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-04-21 09:50:00 | 130.54 | 2025-04-21 15:20:00 | 131.44 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-04-23 09:30:00 | 131.97 | 2025-04-23 09:45:00 | 132.38 | STOP_HIT | 1.00 | -0.31% |
