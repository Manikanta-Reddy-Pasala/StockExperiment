# Indian Energy Exchange Ltd. (IEX)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 134.07
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
| ENTRY1 | 84 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 21 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 63
- **Target hits / Stop hits / Partials:** 21 / 63 / 38
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 29.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 31 | 44.9% | 10 | 38 | 21 | 0.24% | 16.6% |
| BUY @ 2nd Alert (retest1) | 69 | 31 | 44.9% | 10 | 38 | 21 | 0.24% | 16.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 28 | 52.8% | 11 | 25 | 17 | 0.23% | 12.4% |
| SELL @ 2nd Alert (retest1) | 53 | 28 | 52.8% | 11 | 25 | 17 | 0.23% | 12.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 122 | 59 | 48.4% | 21 | 63 | 38 | 0.24% | 29.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:30:00 | 145.25 | 145.77 | 0.00 | ORB-short ORB[145.30,146.30] vol=1.6x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 09:50:00 | 144.53 | 145.47 | 0.00 | T1 1.5R @ 144.53 |
| Target hit | 2024-05-14 10:40:00 | 145.20 | 145.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2024-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:40:00 | 145.15 | 146.33 | 0.00 | ORB-short ORB[145.50,147.15] vol=4.4x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-05-15 09:45:00 | 145.79 | 146.21 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 159.80 | 160.83 | 0.00 | ORB-short ORB[160.30,162.00] vol=1.9x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:35:00 | 158.84 | 160.59 | 0.00 | T1 1.5R @ 158.84 |
| Stop hit — per-position SL triggered | 2024-05-28 09:40:00 | 159.80 | 160.49 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:15:00 | 157.65 | 156.60 | 0.00 | ORB-long ORB[154.65,157.00] vol=1.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-05-29 11:40:00 | 156.98 | 156.83 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:00:00 | 154.90 | 155.39 | 0.00 | ORB-short ORB[155.00,156.35] vol=1.7x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-05-30 10:15:00 | 155.42 | 155.33 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:40:00 | 153.75 | 154.62 | 0.00 | ORB-short ORB[154.05,155.65] vol=1.7x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:15:00 | 152.79 | 154.16 | 0.00 | T1 1.5R @ 152.79 |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 153.75 | 153.91 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:45:00 | 170.75 | 169.37 | 0.00 | ORB-long ORB[168.46,169.98] vol=2.0x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-06-12 10:05:00 | 170.05 | 169.86 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 11:15:00 | 183.10 | 181.75 | 0.00 | ORB-long ORB[180.77,182.70] vol=5.3x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:25:00 | 184.07 | 182.04 | 0.00 | T1 1.5R @ 184.07 |
| Target hit | 2024-06-18 15:20:00 | 185.75 | 184.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:55:00 | 181.64 | 179.88 | 0.00 | ORB-long ORB[179.00,180.99] vol=2.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-06-21 10:40:00 | 180.95 | 180.66 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:05:00 | 182.19 | 180.61 | 0.00 | ORB-long ORB[179.50,181.50] vol=1.8x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 10:30:00 | 183.37 | 181.21 | 0.00 | T1 1.5R @ 183.37 |
| Stop hit — per-position SL triggered | 2024-06-24 10:55:00 | 182.19 | 181.40 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:05:00 | 181.20 | 183.28 | 0.00 | ORB-short ORB[182.50,184.95] vol=1.5x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:00:00 | 180.29 | 182.52 | 0.00 | T1 1.5R @ 180.29 |
| Target hit | 2024-06-25 15:20:00 | 179.50 | 180.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-06-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:05:00 | 176.98 | 178.84 | 0.00 | ORB-short ORB[179.02,180.62] vol=2.1x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 177.58 | 178.47 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 183.10 | 182.14 | 0.00 | ORB-long ORB[181.00,182.27] vol=4.3x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:35:00 | 183.93 | 183.22 | 0.00 | T1 1.5R @ 183.93 |
| Target hit | 2024-07-01 15:20:00 | 189.30 | 187.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:45:00 | 184.57 | 183.21 | 0.00 | ORB-long ORB[182.36,184.10] vol=2.4x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:50:00 | 185.38 | 183.58 | 0.00 | T1 1.5R @ 185.38 |
| Stop hit — per-position SL triggered | 2024-07-05 11:35:00 | 184.57 | 183.90 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:35:00 | 183.10 | 184.41 | 0.00 | ORB-short ORB[183.74,185.49] vol=2.1x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 10:35:00 | 182.17 | 183.89 | 0.00 | T1 1.5R @ 182.17 |
| Target hit | 2024-07-08 15:20:00 | 181.42 | 182.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:45:00 | 184.95 | 183.57 | 0.00 | ORB-long ORB[181.65,184.30] vol=3.5x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-07-09 10:05:00 | 184.23 | 183.95 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:35:00 | 178.33 | 177.46 | 0.00 | ORB-long ORB[176.75,178.31] vol=2.5x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:40:00 | 179.08 | 177.88 | 0.00 | T1 1.5R @ 179.08 |
| Stop hit — per-position SL triggered | 2024-07-12 09:55:00 | 178.33 | 178.12 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 174.06 | 175.52 | 0.00 | ORB-short ORB[174.75,177.09] vol=1.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 174.76 | 175.04 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:45:00 | 171.32 | 172.09 | 0.00 | ORB-short ORB[171.50,173.38] vol=3.4x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-07-19 11:30:00 | 171.95 | 171.50 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 09:35:00 | 169.23 | 167.84 | 0.00 | ORB-long ORB[166.40,168.60] vol=1.6x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 10:00:00 | 170.39 | 169.10 | 0.00 | T1 1.5R @ 170.39 |
| Stop hit — per-position SL triggered | 2024-07-22 10:05:00 | 169.23 | 169.22 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:40:00 | 170.69 | 169.69 | 0.00 | ORB-long ORB[168.21,169.84] vol=2.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 09:50:00 | 171.94 | 170.04 | 0.00 | T1 1.5R @ 171.94 |
| Target hit | 2024-07-24 11:25:00 | 172.30 | 172.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — SELL (started 2024-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:55:00 | 185.30 | 186.98 | 0.00 | ORB-short ORB[186.37,188.60] vol=1.6x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-08-14 12:50:00 | 186.18 | 186.47 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:35:00 | 191.05 | 189.73 | 0.00 | ORB-long ORB[187.00,189.85] vol=3.6x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-08-16 09:40:00 | 190.29 | 189.89 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 197.23 | 196.19 | 0.00 | ORB-long ORB[194.10,196.67] vol=3.5x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-08-21 09:50:00 | 196.65 | 196.26 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 198.74 | 197.44 | 0.00 | ORB-long ORB[195.61,197.89] vol=3.9x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 09:55:00 | 199.73 | 198.71 | 0.00 | T1 1.5R @ 199.73 |
| Target hit | 2024-08-22 10:20:00 | 199.00 | 199.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2024-08-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:55:00 | 187.42 | 188.61 | 0.00 | ORB-short ORB[188.57,190.96] vol=1.8x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:50:00 | 186.60 | 188.32 | 0.00 | T1 1.5R @ 186.60 |
| Stop hit — per-position SL triggered | 2024-08-26 12:25:00 | 187.42 | 188.10 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 191.80 | 190.58 | 0.00 | ORB-long ORB[189.45,191.57] vol=2.5x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:50:00 | 192.81 | 191.43 | 0.00 | T1 1.5R @ 192.81 |
| Target hit | 2024-08-27 15:20:00 | 195.66 | 193.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:40:00 | 198.96 | 197.13 | 0.00 | ORB-long ORB[194.75,197.40] vol=4.0x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 09:55:00 | 200.28 | 198.12 | 0.00 | T1 1.5R @ 200.28 |
| Target hit | 2024-08-28 14:25:00 | 202.60 | 202.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2024-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:35:00 | 203.70 | 202.63 | 0.00 | ORB-long ORB[201.70,203.50] vol=1.9x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-08-29 09:45:00 | 202.93 | 202.80 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:10:00 | 215.11 | 213.72 | 0.00 | ORB-long ORB[212.60,214.80] vol=2.0x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-09-10 10:30:00 | 214.28 | 213.97 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:45:00 | 219.21 | 217.83 | 0.00 | ORB-long ORB[216.06,218.25] vol=1.9x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-09-13 10:55:00 | 218.69 | 217.91 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:40:00 | 229.09 | 227.84 | 0.00 | ORB-long ORB[226.01,228.68] vol=2.3x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-09-20 09:50:00 | 228.10 | 228.00 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:40:00 | 203.66 | 204.91 | 0.00 | ORB-short ORB[204.28,206.38] vol=2.8x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-10-01 11:25:00 | 204.35 | 204.70 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:55:00 | 207.70 | 206.28 | 0.00 | ORB-long ORB[204.19,206.50] vol=1.7x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 206.86 | 206.72 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 09:30:00 | 203.60 | 204.60 | 0.00 | ORB-short ORB[204.50,205.90] vol=1.8x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-10-10 09:35:00 | 204.23 | 204.47 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 203.52 | 203.26 | 0.00 | ORB-long ORB[202.19,203.50] vol=3.8x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:25:00 | 204.41 | 203.51 | 0.00 | T1 1.5R @ 204.41 |
| Stop hit — per-position SL triggered | 2024-10-11 10:45:00 | 203.52 | 203.66 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 202.40 | 203.21 | 0.00 | ORB-short ORB[202.76,204.16] vol=1.9x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:45:00 | 201.57 | 202.83 | 0.00 | T1 1.5R @ 201.57 |
| Target hit | 2024-10-14 15:05:00 | 196.60 | 196.60 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2024-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:10:00 | 190.40 | 191.77 | 0.00 | ORB-short ORB[192.00,194.80] vol=1.8x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:55:00 | 189.42 | 191.01 | 0.00 | T1 1.5R @ 189.42 |
| Stop hit — per-position SL triggered | 2024-10-17 12:55:00 | 190.40 | 190.62 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:25:00 | 184.55 | 185.90 | 0.00 | ORB-short ORB[185.32,187.82] vol=2.0x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:20:00 | 183.16 | 185.33 | 0.00 | T1 1.5R @ 183.16 |
| Target hit | 2024-10-22 15:20:00 | 179.46 | 182.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2024-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:35:00 | 180.29 | 181.46 | 0.00 | ORB-short ORB[181.02,183.12] vol=1.8x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:00:00 | 179.08 | 180.66 | 0.00 | T1 1.5R @ 179.08 |
| Target hit | 2024-10-29 12:15:00 | 179.40 | 179.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — BUY (started 2024-10-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:25:00 | 180.58 | 179.87 | 0.00 | ORB-long ORB[177.85,180.00] vol=1.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-10-30 10:30:00 | 180.09 | 179.89 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-11-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 11:00:00 | 175.40 | 176.79 | 0.00 | ORB-short ORB[176.80,178.90] vol=1.5x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 12:05:00 | 174.53 | 176.45 | 0.00 | T1 1.5R @ 174.53 |
| Target hit | 2024-11-07 15:20:00 | 173.84 | 175.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2024-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:45:00 | 174.20 | 173.24 | 0.00 | ORB-long ORB[171.86,174.01] vol=1.9x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 173.57 | 173.27 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:45:00 | 171.36 | 169.80 | 0.00 | ORB-long ORB[168.64,170.75] vol=1.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 170.61 | 170.34 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 11:10:00 | 164.11 | 164.94 | 0.00 | ORB-short ORB[164.27,166.27] vol=2.6x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:20:00 | 163.12 | 164.44 | 0.00 | T1 1.5R @ 163.12 |
| Target hit | 2024-11-13 15:20:00 | 163.26 | 163.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 164.77 | 163.54 | 0.00 | ORB-long ORB[161.76,164.08] vol=1.8x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-11-19 09:45:00 | 164.17 | 163.82 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 10:00:00 | 159.89 | 160.51 | 0.00 | ORB-short ORB[160.18,162.50] vol=1.8x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-11-21 10:05:00 | 160.61 | 160.53 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 11:00:00 | 166.91 | 166.00 | 0.00 | ORB-long ORB[165.31,166.80] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-11-25 11:10:00 | 166.52 | 166.05 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:35:00 | 167.92 | 167.14 | 0.00 | ORB-long ORB[166.33,167.77] vol=2.5x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 09:45:00 | 168.61 | 167.58 | 0.00 | T1 1.5R @ 168.61 |
| Target hit | 2024-11-27 15:20:00 | 172.13 | 169.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2024-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 174.35 | 173.67 | 0.00 | ORB-long ORB[171.70,173.15] vol=1.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-11-28 10:35:00 | 173.73 | 173.71 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:40:00 | 180.25 | 179.01 | 0.00 | ORB-long ORB[177.74,179.40] vol=2.8x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-12-03 09:55:00 | 179.69 | 179.34 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 11:05:00 | 179.37 | 177.97 | 0.00 | ORB-long ORB[177.35,179.16] vol=3.8x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-12-05 11:25:00 | 178.78 | 178.28 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:40:00 | 179.11 | 178.47 | 0.00 | ORB-long ORB[177.70,178.97] vol=1.6x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:50:00 | 179.81 | 179.16 | 0.00 | T1 1.5R @ 179.81 |
| Stop hit — per-position SL triggered | 2024-12-06 10:05:00 | 179.11 | 179.30 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:30:00 | 186.75 | 186.08 | 0.00 | ORB-long ORB[184.16,186.10] vol=4.7x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-12-10 09:35:00 | 186.14 | 186.11 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:35:00 | 188.83 | 187.62 | 0.00 | ORB-long ORB[185.86,188.23] vol=3.3x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:50:00 | 189.65 | 188.09 | 0.00 | T1 1.5R @ 189.65 |
| Target hit | 2024-12-11 14:30:00 | 190.31 | 190.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2024-12-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:45:00 | 187.36 | 188.35 | 0.00 | ORB-short ORB[187.85,189.85] vol=1.9x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-12-16 11:00:00 | 187.78 | 188.29 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 188.10 | 188.61 | 0.00 | ORB-short ORB[188.40,189.65] vol=2.4x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-12-17 10:10:00 | 188.57 | 188.45 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-12-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:40:00 | 180.25 | 178.13 | 0.00 | ORB-long ORB[175.60,177.05] vol=2.4x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:55:00 | 181.09 | 178.46 | 0.00 | T1 1.5R @ 181.09 |
| Stop hit — per-position SL triggered | 2024-12-24 11:50:00 | 180.25 | 179.23 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:40:00 | 178.28 | 177.23 | 0.00 | ORB-long ORB[176.65,177.95] vol=2.6x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:10:00 | 179.11 | 177.63 | 0.00 | T1 1.5R @ 179.11 |
| Stop hit — per-position SL triggered | 2024-12-26 12:15:00 | 178.28 | 178.08 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:40:00 | 180.27 | 179.05 | 0.00 | ORB-long ORB[178.00,179.45] vol=1.8x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-12-31 11:35:00 | 179.58 | 179.41 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:45:00 | 178.89 | 179.67 | 0.00 | ORB-short ORB[179.23,181.13] vol=1.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:10:00 | 178.33 | 179.30 | 0.00 | T1 1.5R @ 178.33 |
| Target hit | 2025-01-02 11:50:00 | 178.63 | 178.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — SELL (started 2025-01-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:40:00 | 170.82 | 171.23 | 0.00 | ORB-short ORB[170.86,172.91] vol=2.1x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:45:00 | 169.87 | 171.11 | 0.00 | T1 1.5R @ 169.87 |
| Target hit | 2025-01-10 10:20:00 | 170.79 | 170.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — BUY (started 2025-01-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 10:45:00 | 165.80 | 164.46 | 0.00 | ORB-long ORB[163.24,165.00] vol=2.0x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 11:00:00 | 166.79 | 164.94 | 0.00 | T1 1.5R @ 166.79 |
| Stop hit — per-position SL triggered | 2025-01-14 12:15:00 | 165.80 | 165.43 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 11:05:00 | 168.64 | 166.86 | 0.00 | ORB-long ORB[165.53,167.93] vol=1.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 12:30:00 | 169.38 | 167.50 | 0.00 | T1 1.5R @ 169.38 |
| Stop hit — per-position SL triggered | 2025-01-15 13:15:00 | 168.64 | 167.66 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:35:00 | 171.23 | 170.08 | 0.00 | ORB-long ORB[168.20,170.53] vol=1.7x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 09:40:00 | 172.05 | 170.47 | 0.00 | T1 1.5R @ 172.05 |
| Stop hit — per-position SL triggered | 2025-01-16 09:45:00 | 171.23 | 170.54 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:55:00 | 173.29 | 172.08 | 0.00 | ORB-long ORB[170.00,172.15] vol=2.5x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-01-17 10:50:00 | 172.82 | 172.62 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:05:00 | 171.13 | 172.94 | 0.00 | ORB-short ORB[172.10,174.25] vol=1.5x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:20:00 | 170.33 | 172.35 | 0.00 | T1 1.5R @ 170.33 |
| Stop hit — per-position SL triggered | 2025-01-21 11:10:00 | 171.13 | 171.27 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-01-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:45:00 | 167.08 | 168.05 | 0.00 | ORB-short ORB[167.75,170.10] vol=1.9x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-01-22 09:55:00 | 167.74 | 167.99 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:30:00 | 162.40 | 163.73 | 0.00 | ORB-short ORB[163.13,165.37] vol=1.5x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:40:00 | 161.31 | 163.04 | 0.00 | T1 1.5R @ 161.31 |
| Stop hit — per-position SL triggered | 2025-01-28 10:00:00 | 162.40 | 162.60 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:50:00 | 176.00 | 176.85 | 0.00 | ORB-short ORB[177.01,179.44] vol=1.8x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 11:30:00 | 175.13 | 176.70 | 0.00 | T1 1.5R @ 175.13 |
| Target hit | 2025-01-30 15:05:00 | 175.42 | 175.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:15:00 | 176.75 | 175.28 | 0.00 | ORB-long ORB[173.20,174.99] vol=2.4x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-01-31 11:30:00 | 176.23 | 175.37 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-02-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 09:35:00 | 173.88 | 174.41 | 0.00 | ORB-short ORB[174.06,175.55] vol=1.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-02-01 10:50:00 | 174.45 | 174.04 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-02-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:35:00 | 169.96 | 171.43 | 0.00 | ORB-short ORB[170.51,172.94] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-02-04 11:00:00 | 170.52 | 171.18 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:00:00 | 155.50 | 154.56 | 0.00 | ORB-long ORB[152.25,154.50] vol=2.9x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 154.86 | 154.68 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 11:00:00 | 165.39 | 163.92 | 0.00 | ORB-long ORB[162.33,163.92] vol=2.2x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 14:00:00 | 166.01 | 164.88 | 0.00 | T1 1.5R @ 166.01 |
| Target hit | 2025-03-19 15:20:00 | 167.75 | 166.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2025-03-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:45:00 | 179.55 | 178.16 | 0.00 | ORB-long ORB[177.08,179.00] vol=1.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-03-25 10:00:00 | 178.74 | 178.42 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-04-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-02 09:50:00 | 174.91 | 175.85 | 0.00 | ORB-short ORB[175.10,176.97] vol=1.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-04-02 10:00:00 | 175.51 | 175.81 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 11:00:00 | 174.30 | 174.89 | 0.00 | ORB-short ORB[174.68,177.05] vol=2.1x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 174.89 | 174.87 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:45:00 | 186.40 | 185.33 | 0.00 | ORB-long ORB[183.50,185.64] vol=2.3x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-04-16 10:20:00 | 185.79 | 185.65 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:05:00 | 185.78 | 186.79 | 0.00 | ORB-short ORB[185.95,188.00] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-04-17 10:15:00 | 186.41 | 186.74 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:55:00 | 189.92 | 188.92 | 0.00 | ORB-long ORB[187.26,188.63] vol=1.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-04-21 10:10:00 | 189.42 | 189.07 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 188.89 | 190.12 | 0.00 | ORB-short ORB[189.26,191.90] vol=1.8x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-04-23 09:35:00 | 189.49 | 190.03 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 09:45:00 | 190.52 | 191.40 | 0.00 | ORB-short ORB[190.76,192.00] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-04-24 09:50:00 | 191.13 | 191.39 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:45:00 | 193.20 | 191.56 | 0.00 | ORB-long ORB[190.25,191.99] vol=1.7x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 10:15:00 | 194.25 | 192.58 | 0.00 | T1 1.5R @ 194.25 |
| Target hit | 2025-05-05 15:20:00 | 195.24 | 194.71 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 09:30:00 | 145.25 | 2024-05-14 09:50:00 | 144.53 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-14 09:30:00 | 145.25 | 2024-05-14 10:40:00 | 145.20 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2024-05-15 09:40:00 | 145.15 | 2024-05-15 09:45:00 | 145.79 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-05-28 09:30:00 | 159.80 | 2024-05-28 09:35:00 | 158.84 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-28 09:30:00 | 159.80 | 2024-05-28 09:40:00 | 159.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-29 10:15:00 | 157.65 | 2024-05-29 11:40:00 | 156.98 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-05-30 10:00:00 | 154.90 | 2024-05-30 10:15:00 | 155.42 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-31 09:40:00 | 153.75 | 2024-05-31 10:15:00 | 152.79 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-05-31 09:40:00 | 153.75 | 2024-05-31 11:15:00 | 153.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 09:45:00 | 170.75 | 2024-06-12 10:05:00 | 170.05 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-06-18 11:15:00 | 183.10 | 2024-06-18 11:25:00 | 184.07 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-18 11:15:00 | 183.10 | 2024-06-18 15:20:00 | 185.75 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2024-06-21 09:55:00 | 181.64 | 2024-06-21 10:40:00 | 180.95 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-24 10:05:00 | 182.19 | 2024-06-24 10:30:00 | 183.37 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-06-24 10:05:00 | 182.19 | 2024-06-24 10:55:00 | 182.19 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 10:05:00 | 181.20 | 2024-06-25 11:00:00 | 180.29 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-06-25 10:05:00 | 181.20 | 2024-06-25 15:20:00 | 179.50 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-06-26 10:05:00 | 176.98 | 2024-06-26 10:15:00 | 177.58 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-01 09:30:00 | 183.10 | 2024-07-01 09:35:00 | 183.93 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-01 09:30:00 | 183.10 | 2024-07-01 15:20:00 | 189.30 | TARGET_HIT | 0.50 | 3.39% |
| BUY | retest1 | 2024-07-05 10:45:00 | 184.57 | 2024-07-05 10:50:00 | 185.38 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-07-05 10:45:00 | 184.57 | 2024-07-05 11:35:00 | 184.57 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 09:35:00 | 183.10 | 2024-07-08 10:35:00 | 182.17 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-08 09:35:00 | 183.10 | 2024-07-08 15:20:00 | 181.42 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2024-07-09 09:45:00 | 184.95 | 2024-07-09 10:05:00 | 184.23 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-12 09:35:00 | 178.33 | 2024-07-12 09:40:00 | 179.08 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-12 09:35:00 | 178.33 | 2024-07-12 09:55:00 | 178.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 09:30:00 | 174.06 | 2024-07-18 09:40:00 | 174.76 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-19 09:45:00 | 171.32 | 2024-07-19 11:30:00 | 171.95 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-22 09:35:00 | 169.23 | 2024-07-22 10:00:00 | 170.39 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-07-22 09:35:00 | 169.23 | 2024-07-22 10:05:00 | 169.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 09:40:00 | 170.69 | 2024-07-24 09:50:00 | 171.94 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-07-24 09:40:00 | 170.69 | 2024-07-24 11:25:00 | 172.30 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-08-14 10:55:00 | 185.30 | 2024-08-14 12:50:00 | 186.18 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-16 09:35:00 | 191.05 | 2024-08-16 09:40:00 | 190.29 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-21 09:45:00 | 197.23 | 2024-08-21 09:50:00 | 196.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-22 09:30:00 | 198.74 | 2024-08-22 09:55:00 | 199.73 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-08-22 09:30:00 | 198.74 | 2024-08-22 10:20:00 | 199.00 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-08-26 10:55:00 | 187.42 | 2024-08-26 11:50:00 | 186.60 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-26 10:55:00 | 187.42 | 2024-08-26 12:25:00 | 187.42 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-27 09:30:00 | 191.80 | 2024-08-27 10:50:00 | 192.81 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-08-27 09:30:00 | 191.80 | 2024-08-27 15:20:00 | 195.66 | TARGET_HIT | 0.50 | 2.01% |
| BUY | retest1 | 2024-08-28 09:40:00 | 198.96 | 2024-08-28 09:55:00 | 200.28 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-08-28 09:40:00 | 198.96 | 2024-08-28 14:25:00 | 202.60 | TARGET_HIT | 0.50 | 1.83% |
| BUY | retest1 | 2024-08-29 09:35:00 | 203.70 | 2024-08-29 09:45:00 | 202.93 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-10 10:10:00 | 215.11 | 2024-09-10 10:30:00 | 214.28 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-13 10:45:00 | 219.21 | 2024-09-13 10:55:00 | 218.69 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-20 09:40:00 | 229.09 | 2024-09-20 09:50:00 | 228.10 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-01 10:40:00 | 203.66 | 2024-10-01 11:25:00 | 204.35 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-03 09:55:00 | 207.70 | 2024-10-03 10:15:00 | 206.86 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-10 09:30:00 | 203.60 | 2024-10-10 09:35:00 | 204.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-11 09:35:00 | 203.52 | 2024-10-11 10:25:00 | 204.41 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-10-11 09:35:00 | 203.52 | 2024-10-11 10:45:00 | 203.52 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-14 09:30:00 | 202.40 | 2024-10-14 09:45:00 | 201.57 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-14 09:30:00 | 202.40 | 2024-10-14 15:05:00 | 196.60 | TARGET_HIT | 0.50 | 2.87% |
| SELL | retest1 | 2024-10-17 11:10:00 | 190.40 | 2024-10-17 11:55:00 | 189.42 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-17 11:10:00 | 190.40 | 2024-10-17 12:55:00 | 190.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 10:25:00 | 184.55 | 2024-10-22 11:20:00 | 183.16 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-10-22 10:25:00 | 184.55 | 2024-10-22 15:20:00 | 179.46 | TARGET_HIT | 0.50 | 2.76% |
| SELL | retest1 | 2024-10-29 09:35:00 | 180.29 | 2024-10-29 10:00:00 | 179.08 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-10-29 09:35:00 | 180.29 | 2024-10-29 12:15:00 | 179.40 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-10-30 10:25:00 | 180.58 | 2024-10-30 10:30:00 | 180.09 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-11-07 11:00:00 | 175.40 | 2024-11-07 12:05:00 | 174.53 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-07 11:00:00 | 175.40 | 2024-11-07 15:20:00 | 173.84 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2024-11-08 09:45:00 | 174.20 | 2024-11-08 09:50:00 | 173.57 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-11 09:45:00 | 171.36 | 2024-11-11 10:15:00 | 170.61 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-11-13 11:10:00 | 164.11 | 2024-11-13 14:20:00 | 163.12 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-11-13 11:10:00 | 164.11 | 2024-11-13 15:20:00 | 163.26 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2024-11-19 09:30:00 | 164.77 | 2024-11-19 09:45:00 | 164.17 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-11-21 10:00:00 | 159.89 | 2024-11-21 10:05:00 | 160.61 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-11-25 11:00:00 | 166.91 | 2024-11-25 11:10:00 | 166.52 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-27 09:35:00 | 167.92 | 2024-11-27 09:45:00 | 168.61 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-11-27 09:35:00 | 167.92 | 2024-11-27 15:20:00 | 172.13 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2024-11-28 10:30:00 | 174.35 | 2024-11-28 10:35:00 | 173.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-03 09:40:00 | 180.25 | 2024-12-03 09:55:00 | 179.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-05 11:05:00 | 179.37 | 2024-12-05 11:25:00 | 178.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-06 09:40:00 | 179.11 | 2024-12-06 09:50:00 | 179.81 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-12-06 09:40:00 | 179.11 | 2024-12-06 10:05:00 | 179.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 09:30:00 | 186.75 | 2024-12-10 09:35:00 | 186.14 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-11 10:35:00 | 188.83 | 2024-12-11 10:50:00 | 189.65 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-11 10:35:00 | 188.83 | 2024-12-11 14:30:00 | 190.31 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2024-12-16 10:45:00 | 187.36 | 2024-12-16 11:00:00 | 187.78 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-17 09:35:00 | 188.10 | 2024-12-17 10:10:00 | 188.57 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-24 10:40:00 | 180.25 | 2024-12-24 10:55:00 | 181.09 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-24 10:40:00 | 180.25 | 2024-12-24 11:50:00 | 180.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-26 10:40:00 | 178.28 | 2024-12-26 11:10:00 | 179.11 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-26 10:40:00 | 178.28 | 2024-12-26 12:15:00 | 178.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-31 10:40:00 | 180.27 | 2024-12-31 11:35:00 | 179.58 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-02 09:45:00 | 178.89 | 2025-01-02 10:10:00 | 178.33 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-01-02 09:45:00 | 178.89 | 2025-01-02 11:50:00 | 178.63 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-01-10 09:40:00 | 170.82 | 2025-01-10 09:45:00 | 169.87 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-01-10 09:40:00 | 170.82 | 2025-01-10 10:20:00 | 170.79 | TARGET_HIT | 0.50 | 0.02% |
| BUY | retest1 | 2025-01-14 10:45:00 | 165.80 | 2025-01-14 11:00:00 | 166.79 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-14 10:45:00 | 165.80 | 2025-01-14 12:15:00 | 165.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-15 11:05:00 | 168.64 | 2025-01-15 12:30:00 | 169.38 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-15 11:05:00 | 168.64 | 2025-01-15 13:15:00 | 168.64 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-16 09:35:00 | 171.23 | 2025-01-16 09:40:00 | 172.05 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-16 09:35:00 | 171.23 | 2025-01-16 09:45:00 | 171.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 09:55:00 | 173.29 | 2025-01-17 10:50:00 | 172.82 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-21 10:05:00 | 171.13 | 2025-01-21 10:20:00 | 170.33 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-21 10:05:00 | 171.13 | 2025-01-21 11:10:00 | 171.13 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-22 09:45:00 | 167.08 | 2025-01-22 09:55:00 | 167.74 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-28 09:30:00 | 162.40 | 2025-01-28 09:40:00 | 161.31 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-01-28 09:30:00 | 162.40 | 2025-01-28 10:00:00 | 162.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-30 10:50:00 | 176.00 | 2025-01-30 11:30:00 | 175.13 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-01-30 10:50:00 | 176.00 | 2025-01-30 15:05:00 | 175.42 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2025-01-31 11:15:00 | 176.75 | 2025-01-31 11:30:00 | 176.23 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-01 09:35:00 | 173.88 | 2025-02-01 10:50:00 | 174.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-04 10:35:00 | 169.96 | 2025-02-04 11:00:00 | 170.52 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-05 10:00:00 | 155.50 | 2025-03-05 10:15:00 | 154.86 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-03-19 11:00:00 | 165.39 | 2025-03-19 14:00:00 | 166.01 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-03-19 11:00:00 | 165.39 | 2025-03-19 15:20:00 | 167.75 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2025-03-25 09:45:00 | 179.55 | 2025-03-25 10:00:00 | 178.74 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-04-02 09:50:00 | 174.91 | 2025-04-02 10:00:00 | 175.51 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-09 11:00:00 | 174.30 | 2025-04-09 11:15:00 | 174.89 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-16 09:45:00 | 186.40 | 2025-04-16 10:20:00 | 185.79 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-17 10:05:00 | 185.78 | 2025-04-17 10:15:00 | 186.41 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-21 09:55:00 | 189.92 | 2025-04-21 10:10:00 | 189.42 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-23 09:30:00 | 188.89 | 2025-04-23 09:35:00 | 189.49 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-24 09:45:00 | 190.52 | 2025-04-24 09:50:00 | 191.13 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-05 09:45:00 | 193.20 | 2025-05-05 10:15:00 | 194.25 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-05-05 09:45:00 | 193.20 | 2025-05-05 15:20:00 | 195.24 | TARGET_HIT | 0.50 | 1.06% |
