# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-12-01 15:25:00 (28921 bars)
- **Last close:** 264.25
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
| ENTRY1 | 53 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 46
- **Target hits / Stop hits / Partials:** 7 / 46 / 19
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 12.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 15 | 39.5% | 4 | 23 | 11 | 0.25% | 9.6% |
| BUY @ 2nd Alert (retest1) | 38 | 15 | 39.5% | 4 | 23 | 11 | 0.25% | 9.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 34 | 11 | 32.4% | 3 | 23 | 8 | 0.08% | 2.6% |
| SELL @ 2nd Alert (retest1) | 34 | 11 | 32.4% | 3 | 23 | 8 | 0.08% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 72 | 26 | 36.1% | 7 | 46 | 19 | 0.17% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:55:00 | 190.10 | 191.05 | 0.00 | ORB-short ORB[190.50,192.30] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-05-16 10:25:00 | 190.88 | 190.82 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:10:00 | 192.85 | 191.22 | 0.00 | ORB-long ORB[189.70,191.65] vol=5.3x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-05-17 11:25:00 | 192.22 | 191.66 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:30:00 | 198.00 | 197.17 | 0.00 | ORB-long ORB[195.60,197.90] vol=2.0x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-05-18 09:40:00 | 197.21 | 197.27 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:40:00 | 204.10 | 201.57 | 0.00 | ORB-long ORB[199.55,201.70] vol=2.4x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-05-21 09:45:00 | 203.03 | 201.72 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 11:10:00 | 190.95 | 193.74 | 0.00 | ORB-short ORB[192.40,195.25] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-05-23 15:20:00 | 191.15 | 192.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-05-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 09:45:00 | 193.40 | 192.76 | 0.00 | ORB-long ORB[191.15,193.30] vol=1.9x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-05-24 09:50:00 | 192.73 | 192.78 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 191.80 | 193.44 | 0.00 | ORB-short ORB[193.60,195.15] vol=2.4x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:00:00 | 190.46 | 192.83 | 0.00 | T1 1.5R @ 190.46 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 191.80 | 192.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:45:00 | 186.31 | 184.00 | 0.00 | ORB-long ORB[181.55,184.29] vol=2.4x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-06-11 10:55:00 | 185.65 | 184.16 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:50:00 | 185.36 | 186.29 | 0.00 | ORB-short ORB[185.50,187.55] vol=2.2x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-06-13 09:55:00 | 186.04 | 186.26 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:40:00 | 191.60 | 190.14 | 0.00 | ORB-long ORB[188.65,191.29] vol=2.3x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:50:00 | 192.96 | 190.69 | 0.00 | T1 1.5R @ 192.96 |
| Target hit | 2024-06-14 14:50:00 | 192.05 | 192.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-06-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:35:00 | 189.25 | 186.43 | 0.00 | ORB-long ORB[183.22,185.80] vol=1.9x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:55:00 | 190.29 | 187.25 | 0.00 | T1 1.5R @ 190.29 |
| Target hit | 2024-06-20 15:20:00 | 191.48 | 189.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-06-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:45:00 | 189.43 | 190.71 | 0.00 | ORB-short ORB[190.50,192.25] vol=1.6x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:10:00 | 188.60 | 190.48 | 0.00 | T1 1.5R @ 188.60 |
| Target hit | 2024-06-25 15:00:00 | 188.87 | 188.60 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2024-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:35:00 | 190.65 | 188.87 | 0.00 | ORB-long ORB[187.41,189.33] vol=3.6x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 10:20:00 | 191.81 | 189.99 | 0.00 | T1 1.5R @ 191.81 |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 190.65 | 190.32 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:45:00 | 191.17 | 191.71 | 0.00 | ORB-short ORB[191.32,192.90] vol=2.1x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-07-02 09:55:00 | 191.71 | 191.69 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:35:00 | 192.12 | 193.01 | 0.00 | ORB-short ORB[192.60,194.35] vol=1.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-07-03 10:55:00 | 192.69 | 192.94 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:35:00 | 197.00 | 195.51 | 0.00 | ORB-long ORB[193.69,196.34] vol=2.0x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 11:00:00 | 197.93 | 196.20 | 0.00 | T1 1.5R @ 197.93 |
| Stop hit — per-position SL triggered | 2024-07-05 11:10:00 | 197.00 | 196.31 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:15:00 | 201.76 | 200.14 | 0.00 | ORB-long ORB[198.80,201.00] vol=2.5x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-07-08 11:10:00 | 200.89 | 200.82 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:40:00 | 182.70 | 184.05 | 0.00 | ORB-short ORB[183.05,185.11] vol=1.8x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:05:00 | 181.42 | 183.34 | 0.00 | T1 1.5R @ 181.42 |
| Stop hit — per-position SL triggered | 2024-07-25 10:10:00 | 182.70 | 183.30 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 11:15:00 | 192.75 | 191.46 | 0.00 | ORB-long ORB[189.63,192.26] vol=5.1x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-07-29 12:15:00 | 192.13 | 191.95 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 193.00 | 191.65 | 0.00 | ORB-long ORB[190.20,192.60] vol=2.1x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:35:00 | 193.97 | 192.21 | 0.00 | T1 1.5R @ 193.97 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 193.00 | 192.41 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:35:00 | 185.60 | 187.59 | 0.00 | ORB-short ORB[186.64,189.43] vol=2.7x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-08-02 10:10:00 | 186.82 | 186.74 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 11:10:00 | 174.52 | 175.07 | 0.00 | ORB-short ORB[174.69,176.84] vol=1.8x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 12:50:00 | 173.60 | 174.99 | 0.00 | T1 1.5R @ 173.60 |
| Stop hit — per-position SL triggered | 2024-08-09 13:30:00 | 174.52 | 174.91 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:55:00 | 167.44 | 170.75 | 0.00 | ORB-short ORB[170.35,172.86] vol=2.9x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-08-14 11:00:00 | 168.46 | 170.54 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 11:00:00 | 170.90 | 171.33 | 0.00 | ORB-short ORB[170.95,172.50] vol=1.7x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-08-22 11:25:00 | 171.36 | 171.30 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:35:00 | 179.98 | 181.81 | 0.00 | ORB-short ORB[181.26,183.34] vol=2.2x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-08-29 11:05:00 | 180.59 | 181.30 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 10:35:00 | 174.23 | 175.23 | 0.00 | ORB-short ORB[174.72,176.70] vol=1.9x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-09-04 11:35:00 | 174.77 | 175.07 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:05:00 | 173.25 | 174.48 | 0.00 | ORB-short ORB[174.58,176.49] vol=1.7x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-09-06 10:20:00 | 173.81 | 174.23 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:30:00 | 174.75 | 173.47 | 0.00 | ORB-long ORB[171.75,173.90] vol=3.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-09-10 09:55:00 | 174.08 | 174.28 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:00:00 | 183.05 | 185.04 | 0.00 | ORB-short ORB[185.40,187.45] vol=1.9x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:10:00 | 181.91 | 184.69 | 0.00 | T1 1.5R @ 181.91 |
| Target hit | 2024-09-19 12:30:00 | 181.25 | 180.68 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — SELL (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 11:15:00 | 181.52 | 181.96 | 0.00 | ORB-short ORB[181.67,183.54] vol=2.5x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:30:00 | 180.82 | 181.90 | 0.00 | T1 1.5R @ 180.82 |
| Stop hit — per-position SL triggered | 2024-09-23 12:20:00 | 181.52 | 181.63 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:50:00 | 194.88 | 193.66 | 0.00 | ORB-long ORB[192.64,194.70] vol=2.0x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:00:00 | 195.98 | 193.87 | 0.00 | T1 1.5R @ 195.98 |
| Target hit | 2024-09-26 15:20:00 | 203.00 | 199.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2024-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:30:00 | 219.29 | 217.49 | 0.00 | ORB-long ORB[215.90,218.35] vol=1.6x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:45:00 | 220.96 | 218.53 | 0.00 | T1 1.5R @ 220.96 |
| Stop hit — per-position SL triggered | 2024-10-16 10:20:00 | 219.29 | 218.87 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 09:40:00 | 242.01 | 243.67 | 0.00 | ORB-short ORB[242.61,246.14] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-11-29 09:45:00 | 243.22 | 243.59 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:15:00 | 246.07 | 248.31 | 0.00 | ORB-short ORB[247.19,250.24] vol=1.6x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-12-04 10:50:00 | 247.05 | 247.88 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:40:00 | 253.70 | 251.61 | 0.00 | ORB-long ORB[248.20,251.70] vol=2.9x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-12-11 09:55:00 | 252.80 | 252.35 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 11:00:00 | 214.97 | 216.82 | 0.00 | ORB-short ORB[216.13,218.51] vol=2.2x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:30:00 | 213.94 | 216.24 | 0.00 | T1 1.5R @ 213.94 |
| Stop hit — per-position SL triggered | 2024-12-24 12:40:00 | 214.97 | 215.83 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 216.15 | 216.89 | 0.00 | ORB-short ORB[216.55,218.90] vol=2.0x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-12-27 10:10:00 | 216.83 | 216.61 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:00:00 | 213.49 | 211.94 | 0.00 | ORB-long ORB[210.11,212.17] vol=2.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-01-01 11:35:00 | 212.87 | 212.37 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:40:00 | 217.70 | 216.89 | 0.00 | ORB-long ORB[215.20,217.20] vol=3.7x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-01-03 09:55:00 | 217.03 | 217.00 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-01-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:30:00 | 209.40 | 207.01 | 0.00 | ORB-long ORB[205.16,207.79] vol=1.6x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:30:00 | 210.59 | 207.98 | 0.00 | T1 1.5R @ 210.59 |
| Stop hit — per-position SL triggered | 2025-01-20 11:35:00 | 209.40 | 208.04 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:55:00 | 206.30 | 204.49 | 0.00 | ORB-long ORB[202.40,204.26] vol=2.0x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:00:00 | 207.47 | 204.96 | 0.00 | T1 1.5R @ 207.47 |
| Stop hit — per-position SL triggered | 2025-01-23 10:50:00 | 206.30 | 206.00 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-01-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:30:00 | 198.41 | 198.97 | 0.00 | ORB-short ORB[198.45,200.79] vol=3.4x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-01-27 09:35:00 | 199.33 | 199.02 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:45:00 | 195.61 | 194.23 | 0.00 | ORB-long ORB[192.49,194.30] vol=1.7x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-01-31 10:10:00 | 194.89 | 194.57 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-02-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:40:00 | 200.47 | 198.99 | 0.00 | ORB-long ORB[197.00,199.54] vol=1.5x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 11:15:00 | 202.01 | 199.73 | 0.00 | T1 1.5R @ 202.01 |
| Stop hit — per-position SL triggered | 2025-02-07 11:35:00 | 200.47 | 199.96 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:40:00 | 182.05 | 181.07 | 0.00 | ORB-long ORB[179.10,181.65] vol=1.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-03-05 10:00:00 | 181.19 | 181.22 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 10:50:00 | 188.70 | 190.34 | 0.00 | ORB-short ORB[189.82,192.66] vol=2.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-03-13 11:05:00 | 189.40 | 190.11 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-03-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:05:00 | 186.09 | 187.91 | 0.00 | ORB-short ORB[187.38,189.48] vol=1.6x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 186.78 | 187.77 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:40:00 | 153.98 | 152.10 | 0.00 | ORB-long ORB[150.80,152.49] vol=2.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-04-17 10:50:00 | 153.39 | 152.21 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:40:00 | 156.36 | 155.30 | 0.00 | ORB-long ORB[153.73,155.73] vol=1.5x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:00:00 | 157.10 | 155.51 | 0.00 | T1 1.5R @ 157.10 |
| Target hit | 2025-04-21 15:20:00 | 161.58 | 158.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2025-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:50:00 | 163.10 | 162.14 | 0.00 | ORB-long ORB[161.00,162.68] vol=1.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-04-24 09:55:00 | 162.56 | 162.20 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 162.14 | 162.80 | 0.00 | ORB-short ORB[162.36,163.60] vol=2.3x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 161.38 | 162.52 | 0.00 | T1 1.5R @ 161.38 |
| Target hit | 2025-04-25 15:20:00 | 156.01 | 158.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:45:00 | 159.72 | 160.73 | 0.00 | ORB-short ORB[160.08,161.75] vol=1.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-04-29 09:55:00 | 160.32 | 160.59 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 11:15:00 | 159.21 | 158.39 | 0.00 | ORB-long ORB[157.14,159.18] vol=2.4x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-05-08 11:55:00 | 158.76 | 158.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:55:00 | 190.10 | 2024-05-16 10:25:00 | 190.88 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-05-17 11:10:00 | 192.85 | 2024-05-17 11:25:00 | 192.22 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-18 09:30:00 | 198.00 | 2024-05-18 09:40:00 | 197.21 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-05-21 09:40:00 | 204.10 | 2024-05-21 09:45:00 | 203.03 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-05-23 11:10:00 | 190.95 | 2024-05-23 15:20:00 | 191.15 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest1 | 2024-05-24 09:45:00 | 193.40 | 2024-05-24 09:50:00 | 192.73 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-27 09:45:00 | 191.80 | 2024-05-27 10:00:00 | 190.46 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-05-27 09:45:00 | 191.80 | 2024-05-27 10:05:00 | 191.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 10:45:00 | 186.31 | 2024-06-11 10:55:00 | 185.65 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-13 09:50:00 | 185.36 | 2024-06-13 09:55:00 | 186.04 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-14 09:40:00 | 191.60 | 2024-06-14 09:50:00 | 192.96 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-06-14 09:40:00 | 191.60 | 2024-06-14 14:50:00 | 192.05 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-06-20 10:35:00 | 189.25 | 2024-06-20 10:55:00 | 190.29 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-20 10:35:00 | 189.25 | 2024-06-20 15:20:00 | 191.48 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2024-06-25 10:45:00 | 189.43 | 2024-06-25 11:10:00 | 188.60 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-06-25 10:45:00 | 189.43 | 2024-06-25 15:00:00 | 188.87 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2024-07-01 09:35:00 | 190.65 | 2024-07-01 10:20:00 | 191.81 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-01 09:35:00 | 190.65 | 2024-07-01 11:15:00 | 190.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:45:00 | 191.17 | 2024-07-02 09:55:00 | 191.71 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-03 10:35:00 | 192.12 | 2024-07-03 10:55:00 | 192.69 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-05 10:35:00 | 197.00 | 2024-07-05 11:00:00 | 197.93 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-05 10:35:00 | 197.00 | 2024-07-05 11:10:00 | 197.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 10:15:00 | 201.76 | 2024-07-08 11:10:00 | 200.89 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-07-25 09:40:00 | 182.70 | 2024-07-25 10:05:00 | 181.42 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-07-25 09:40:00 | 182.70 | 2024-07-25 10:10:00 | 182.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-29 11:15:00 | 192.75 | 2024-07-29 12:15:00 | 192.13 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-31 09:30:00 | 193.00 | 2024-07-31 09:35:00 | 193.97 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-31 09:30:00 | 193.00 | 2024-07-31 09:45:00 | 193.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-02 09:35:00 | 185.60 | 2024-08-02 10:10:00 | 186.82 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2024-08-09 11:10:00 | 174.52 | 2024-08-09 12:50:00 | 173.60 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-08-09 11:10:00 | 174.52 | 2024-08-09 13:30:00 | 174.52 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 10:55:00 | 167.44 | 2024-08-14 11:00:00 | 168.46 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2024-08-22 11:00:00 | 170.90 | 2024-08-22 11:25:00 | 171.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-29 10:35:00 | 179.98 | 2024-08-29 11:05:00 | 180.59 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-04 10:35:00 | 174.23 | 2024-09-04 11:35:00 | 174.77 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-06 10:05:00 | 173.25 | 2024-09-06 10:20:00 | 173.81 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-10 09:30:00 | 174.75 | 2024-09-10 09:55:00 | 174.08 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-19 10:00:00 | 183.05 | 2024-09-19 10:10:00 | 181.91 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-09-19 10:00:00 | 183.05 | 2024-09-19 12:30:00 | 181.25 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2024-09-23 11:15:00 | 181.52 | 2024-09-23 11:30:00 | 180.82 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-09-23 11:15:00 | 181.52 | 2024-09-23 12:20:00 | 181.52 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 10:50:00 | 194.88 | 2024-09-26 11:00:00 | 195.98 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-09-26 10:50:00 | 194.88 | 2024-09-26 15:20:00 | 203.00 | TARGET_HIT | 0.50 | 4.17% |
| BUY | retest1 | 2024-10-16 09:30:00 | 219.29 | 2024-10-16 09:45:00 | 220.96 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-10-16 09:30:00 | 219.29 | 2024-10-16 10:20:00 | 219.29 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 09:40:00 | 242.01 | 2024-11-29 09:45:00 | 243.22 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-12-04 10:15:00 | 246.07 | 2024-12-04 10:50:00 | 247.05 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-12-11 09:40:00 | 253.70 | 2024-12-11 09:55:00 | 252.80 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-24 11:00:00 | 214.97 | 2024-12-24 11:30:00 | 213.94 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-24 11:00:00 | 214.97 | 2024-12-24 12:40:00 | 214.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 09:30:00 | 216.15 | 2024-12-27 10:10:00 | 216.83 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-01 11:00:00 | 213.49 | 2025-01-01 11:35:00 | 212.87 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-03 09:40:00 | 217.70 | 2025-01-03 09:55:00 | 217.03 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-20 10:30:00 | 209.40 | 2025-01-20 11:30:00 | 210.59 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-01-20 10:30:00 | 209.40 | 2025-01-20 11:35:00 | 209.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 09:55:00 | 206.30 | 2025-01-23 10:00:00 | 207.47 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-01-23 09:55:00 | 206.30 | 2025-01-23 10:50:00 | 206.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 09:30:00 | 198.41 | 2025-01-27 09:35:00 | 199.33 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-01-31 09:45:00 | 195.61 | 2025-01-31 10:10:00 | 194.89 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-02-07 10:40:00 | 200.47 | 2025-02-07 11:15:00 | 202.01 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2025-02-07 10:40:00 | 200.47 | 2025-02-07 11:35:00 | 200.47 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 09:40:00 | 182.05 | 2025-03-05 10:00:00 | 181.19 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-03-13 10:50:00 | 188.70 | 2025-03-13 11:05:00 | 189.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-03-20 10:05:00 | 186.09 | 2025-03-20 10:15:00 | 186.78 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-17 10:40:00 | 153.98 | 2025-04-17 10:50:00 | 153.39 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-04-21 10:40:00 | 156.36 | 2025-04-21 11:00:00 | 157.10 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-21 10:40:00 | 156.36 | 2025-04-21 15:20:00 | 161.58 | TARGET_HIT | 0.50 | 3.34% |
| BUY | retest1 | 2025-04-24 09:50:00 | 163.10 | 2025-04-24 09:55:00 | 162.56 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-25 09:30:00 | 162.14 | 2025-04-25 09:35:00 | 161.38 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-04-25 09:30:00 | 162.14 | 2025-04-25 15:20:00 | 156.01 | TARGET_HIT | 0.50 | 3.78% |
| SELL | retest1 | 2025-04-29 09:45:00 | 159.72 | 2025-04-29 09:55:00 | 160.32 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-05-08 11:15:00 | 159.21 | 2025-05-08 11:55:00 | 158.76 | STOP_HIT | 1.00 | -0.28% |
