# NTPC (NTPC)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55356 bars)
- **Last close:** 402.10
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
| ENTRY1 | 96 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 16 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 134 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 80
- **Target hits / Stop hits / Partials:** 16 / 80 / 38
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 12.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 33 | 44.0% | 12 | 42 | 21 | 0.13% | 9.4% |
| BUY @ 2nd Alert (retest1) | 75 | 33 | 44.0% | 12 | 42 | 21 | 0.13% | 9.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 59 | 21 | 35.6% | 4 | 38 | 17 | 0.05% | 2.7% |
| SELL @ 2nd Alert (retest1) | 59 | 21 | 35.6% | 4 | 38 | 17 | 0.05% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 134 | 54 | 40.3% | 16 | 80 | 38 | 0.09% | 12.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 11:05:00 | 174.95 | 174.78 | 0.00 | ORB-long ORB[173.45,174.90] vol=1.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-05-24 11:45:00 | 174.66 | 174.80 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 11:15:00 | 176.90 | 175.89 | 0.00 | ORB-long ORB[174.85,176.40] vol=2.1x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 11:35:00 | 177.41 | 176.13 | 0.00 | T1 1.5R @ 177.41 |
| Stop hit — per-position SL triggered | 2023-05-29 12:05:00 | 176.90 | 176.37 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 11:15:00 | 173.95 | 174.63 | 0.00 | ORB-short ORB[174.50,177.00] vol=2.5x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:45:00 | 173.44 | 174.44 | 0.00 | T1 1.5R @ 173.44 |
| Stop hit — per-position SL triggered | 2023-05-31 13:35:00 | 173.95 | 173.82 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-01 11:15:00 | 174.00 | 173.57 | 0.00 | ORB-long ORB[171.85,173.60] vol=1.6x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-06-01 12:10:00 | 173.68 | 173.69 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 11:05:00 | 174.00 | 174.92 | 0.00 | ORB-short ORB[174.50,175.90] vol=1.8x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 11:25:00 | 173.48 | 174.58 | 0.00 | T1 1.5R @ 173.48 |
| Stop hit — per-position SL triggered | 2023-06-02 11:45:00 | 174.00 | 174.48 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 11:05:00 | 175.00 | 175.47 | 0.00 | ORB-short ORB[175.25,176.20] vol=2.9x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-06-05 11:20:00 | 175.30 | 175.38 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:20:00 | 184.60 | 183.77 | 0.00 | ORB-long ORB[182.65,183.80] vol=5.8x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 10:50:00 | 185.27 | 184.17 | 0.00 | T1 1.5R @ 185.27 |
| Target hit | 2023-06-12 15:20:00 | 185.55 | 185.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2023-06-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 10:25:00 | 185.80 | 186.11 | 0.00 | ORB-short ORB[185.85,186.65] vol=2.3x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 11:25:00 | 185.26 | 185.97 | 0.00 | T1 1.5R @ 185.26 |
| Stop hit — per-position SL triggered | 2023-06-13 11:45:00 | 185.80 | 185.94 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:30:00 | 187.05 | 186.21 | 0.00 | ORB-long ORB[185.35,186.55] vol=4.5x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 11:05:00 | 187.62 | 186.62 | 0.00 | T1 1.5R @ 187.62 |
| Target hit | 2023-06-14 14:45:00 | 187.35 | 187.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2023-06-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 09:45:00 | 185.95 | 186.70 | 0.00 | ORB-short ORB[186.50,187.95] vol=1.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2023-06-22 10:25:00 | 186.39 | 186.40 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-23 09:30:00 | 186.00 | 184.92 | 0.00 | ORB-long ORB[183.75,185.40] vol=1.7x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 09:50:00 | 186.82 | 185.67 | 0.00 | T1 1.5R @ 186.82 |
| Target hit | 2023-06-23 13:20:00 | 186.50 | 186.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 10:15:00 | 186.25 | 185.89 | 0.00 | ORB-long ORB[185.30,186.20] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-06-28 10:20:00 | 185.96 | 185.95 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 10:00:00 | 188.90 | 189.74 | 0.00 | ORB-short ORB[189.10,190.80] vol=1.6x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-07-03 10:20:00 | 189.35 | 189.44 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 10:15:00 | 194.60 | 194.07 | 0.00 | ORB-long ORB[192.75,194.20] vol=2.5x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 10:25:00 | 195.16 | 194.25 | 0.00 | T1 1.5R @ 195.16 |
| Target hit | 2023-07-06 13:30:00 | 195.80 | 195.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2023-07-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 10:55:00 | 194.70 | 193.83 | 0.00 | ORB-long ORB[192.75,193.90] vol=1.9x ATR=0.38 |
| Stop hit — per-position SL triggered | 2023-07-12 11:10:00 | 194.32 | 193.90 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:50:00 | 190.20 | 190.54 | 0.00 | ORB-short ORB[190.75,191.80] vol=1.6x ATR=0.34 |
| Stop hit — per-position SL triggered | 2023-07-13 11:20:00 | 190.54 | 190.48 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 11:00:00 | 188.60 | 187.43 | 0.00 | ORB-long ORB[187.25,188.45] vol=2.5x ATR=0.43 |
| Stop hit — per-position SL triggered | 2023-07-17 11:15:00 | 188.17 | 187.92 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:55:00 | 188.10 | 187.30 | 0.00 | ORB-long ORB[186.30,187.65] vol=1.6x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 10:15:00 | 188.72 | 187.67 | 0.00 | T1 1.5R @ 188.72 |
| Stop hit — per-position SL triggered | 2023-07-18 10:20:00 | 188.10 | 187.71 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 11:15:00 | 192.15 | 192.98 | 0.00 | ORB-short ORB[192.65,194.15] vol=1.5x ATR=0.31 |
| Stop hit — per-position SL triggered | 2023-07-20 11:25:00 | 192.46 | 192.89 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 11:10:00 | 194.35 | 195.17 | 0.00 | ORB-short ORB[194.55,197.00] vol=2.5x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 11:40:00 | 193.82 | 194.95 | 0.00 | T1 1.5R @ 193.82 |
| Stop hit — per-position SL triggered | 2023-07-24 12:10:00 | 194.35 | 194.64 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:45:00 | 198.50 | 197.32 | 0.00 | ORB-long ORB[195.90,197.40] vol=3.4x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 09:55:00 | 199.25 | 197.82 | 0.00 | T1 1.5R @ 199.25 |
| Stop hit — per-position SL triggered | 2023-07-25 10:00:00 | 198.50 | 197.89 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 10:15:00 | 205.80 | 201.87 | 0.00 | ORB-long ORB[199.50,201.90] vol=2.3x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 10:20:00 | 206.93 | 203.52 | 0.00 | T1 1.5R @ 206.93 |
| Target hit | 2023-07-28 15:20:00 | 210.15 | 207.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2023-08-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 10:55:00 | 220.65 | 219.93 | 0.00 | ORB-long ORB[218.05,220.25] vol=1.9x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-08-07 11:25:00 | 220.16 | 220.12 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 09:35:00 | 218.30 | 219.35 | 0.00 | ORB-short ORB[218.50,220.30] vol=1.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-08-08 09:40:00 | 218.79 | 219.26 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:55:00 | 217.00 | 218.00 | 0.00 | ORB-short ORB[217.85,219.70] vol=2.0x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-08-09 11:20:00 | 217.62 | 217.84 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 11:00:00 | 218.00 | 219.36 | 0.00 | ORB-short ORB[218.30,220.00] vol=2.0x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-08-10 12:40:00 | 218.54 | 218.90 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 10:45:00 | 214.30 | 213.46 | 0.00 | ORB-long ORB[212.05,213.10] vol=1.6x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-08-16 10:50:00 | 213.88 | 213.50 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:15:00 | 216.45 | 217.97 | 0.00 | ORB-short ORB[217.85,219.80] vol=1.9x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 11:55:00 | 215.79 | 217.71 | 0.00 | T1 1.5R @ 215.79 |
| Stop hit — per-position SL triggered | 2023-08-17 12:40:00 | 216.45 | 217.47 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 11:10:00 | 215.65 | 215.83 | 0.00 | ORB-short ORB[215.75,216.65] vol=4.5x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 12:10:00 | 215.14 | 215.67 | 0.00 | T1 1.5R @ 215.14 |
| Target hit | 2023-08-18 14:00:00 | 215.00 | 214.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2023-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 09:40:00 | 218.10 | 217.30 | 0.00 | ORB-long ORB[215.70,217.65] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-08-21 09:50:00 | 217.47 | 217.37 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:30:00 | 220.80 | 220.34 | 0.00 | ORB-long ORB[218.85,220.75] vol=2.3x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 10:05:00 | 221.46 | 220.80 | 0.00 | T1 1.5R @ 221.46 |
| Stop hit — per-position SL triggered | 2023-08-22 10:40:00 | 220.80 | 220.97 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:30:00 | 219.45 | 219.88 | 0.00 | ORB-short ORB[219.65,221.85] vol=1.9x ATR=0.44 |
| Stop hit — per-position SL triggered | 2023-08-25 10:40:00 | 219.89 | 219.83 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 09:30:00 | 220.40 | 219.86 | 0.00 | ORB-long ORB[218.25,220.20] vol=2.3x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 12:30:00 | 221.03 | 220.43 | 0.00 | T1 1.5R @ 221.03 |
| Target hit | 2023-08-29 15:20:00 | 221.25 | 220.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2023-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 11:15:00 | 225.25 | 220.81 | 0.00 | ORB-long ORB[216.15,217.85] vol=6.6x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 11:35:00 | 226.37 | 221.74 | 0.00 | T1 1.5R @ 226.37 |
| Target hit | 2023-09-01 15:20:00 | 231.20 | 226.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2023-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:50:00 | 234.50 | 232.60 | 0.00 | ORB-long ORB[231.25,233.70] vol=2.0x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-09-04 10:55:00 | 233.60 | 232.68 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 10:40:00 | 233.80 | 234.97 | 0.00 | ORB-short ORB[235.30,236.90] vol=2.2x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 12:20:00 | 232.86 | 234.49 | 0.00 | T1 1.5R @ 232.86 |
| Stop hit — per-position SL triggered | 2023-09-05 14:35:00 | 233.80 | 233.86 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 09:30:00 | 233.15 | 233.68 | 0.00 | ORB-short ORB[233.40,235.85] vol=2.9x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 09:55:00 | 232.37 | 233.41 | 0.00 | T1 1.5R @ 232.37 |
| Target hit | 2023-09-06 12:15:00 | 230.90 | 230.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2023-09-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 10:35:00 | 230.00 | 230.55 | 0.00 | ORB-short ORB[230.30,233.25] vol=1.7x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-09-07 10:45:00 | 230.52 | 230.53 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-09-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 10:30:00 | 239.35 | 239.68 | 0.00 | ORB-short ORB[241.60,244.95] vol=6.0x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-09-12 10:35:00 | 240.57 | 239.69 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 10:55:00 | 240.70 | 238.24 | 0.00 | ORB-long ORB[236.05,238.30] vol=2.3x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-09-18 11:05:00 | 239.99 | 238.51 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 09:40:00 | 242.15 | 240.66 | 0.00 | ORB-long ORB[239.05,240.40] vol=2.3x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 10:15:00 | 243.15 | 241.92 | 0.00 | T1 1.5R @ 243.15 |
| Stop hit — per-position SL triggered | 2023-09-26 10:25:00 | 242.15 | 242.00 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 11:05:00 | 242.65 | 244.93 | 0.00 | ORB-short ORB[243.85,246.70] vol=1.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2023-10-03 11:25:00 | 243.40 | 244.67 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-10-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:45:00 | 232.80 | 234.32 | 0.00 | ORB-short ORB[235.15,237.40] vol=2.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-10-05 11:25:00 | 233.46 | 233.96 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 10:00:00 | 234.00 | 232.75 | 0.00 | ORB-long ORB[230.50,233.85] vol=1.6x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-10-09 12:15:00 | 233.20 | 233.55 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 09:50:00 | 234.70 | 235.42 | 0.00 | ORB-short ORB[234.90,236.85] vol=1.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 10:10:00 | 233.76 | 235.05 | 0.00 | T1 1.5R @ 233.76 |
| Stop hit — per-position SL triggered | 2023-10-10 10:15:00 | 234.70 | 235.01 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:35:00 | 238.60 | 237.38 | 0.00 | ORB-long ORB[236.00,237.70] vol=2.2x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-10-11 09:55:00 | 237.97 | 238.02 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 10:40:00 | 242.10 | 240.84 | 0.00 | ORB-long ORB[238.60,241.40] vol=2.1x ATR=0.59 |
| Stop hit — per-position SL triggered | 2023-10-12 10:55:00 | 241.51 | 240.92 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-10-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 09:55:00 | 243.15 | 241.37 | 0.00 | ORB-long ORB[239.60,241.25] vol=2.1x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-10-13 10:00:00 | 242.54 | 241.52 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 10:10:00 | 243.85 | 244.75 | 0.00 | ORB-short ORB[244.15,245.25] vol=3.3x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-10-17 10:15:00 | 244.34 | 244.54 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:45:00 | 244.40 | 245.10 | 0.00 | ORB-short ORB[244.80,246.80] vol=1.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:15:00 | 243.82 | 244.84 | 0.00 | T1 1.5R @ 243.82 |
| Stop hit — per-position SL triggered | 2023-10-18 11:35:00 | 244.40 | 244.75 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-10-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:55:00 | 239.20 | 239.38 | 0.00 | ORB-short ORB[239.50,240.90] vol=3.3x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 11:25:00 | 238.17 | 239.29 | 0.00 | T1 1.5R @ 238.17 |
| Target hit | 2023-10-23 15:20:00 | 235.40 | 237.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2023-10-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 09:35:00 | 236.70 | 235.13 | 0.00 | ORB-long ORB[232.00,234.45] vol=2.0x ATR=1.05 |
| Stop hit — per-position SL triggered | 2023-10-27 10:10:00 | 235.65 | 235.81 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:20:00 | 236.90 | 236.22 | 0.00 | ORB-long ORB[234.55,236.45] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-11-06 10:45:00 | 236.51 | 236.37 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 11:05:00 | 238.35 | 236.82 | 0.00 | ORB-long ORB[235.20,236.95] vol=1.5x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-11-07 11:20:00 | 237.89 | 237.04 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 10:55:00 | 237.00 | 238.34 | 0.00 | ORB-short ORB[238.80,240.80] vol=1.9x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 11:50:00 | 236.47 | 237.96 | 0.00 | T1 1.5R @ 236.47 |
| Stop hit — per-position SL triggered | 2023-11-08 12:20:00 | 237.00 | 237.78 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 11:10:00 | 238.25 | 237.19 | 0.00 | ORB-long ORB[236.00,237.50] vol=1.9x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:45:00 | 239.26 | 237.35 | 0.00 | T1 1.5R @ 239.26 |
| Stop hit — per-position SL triggered | 2023-11-09 13:00:00 | 238.25 | 237.92 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-11-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 09:45:00 | 238.85 | 237.75 | 0.00 | ORB-long ORB[235.90,237.85] vol=1.8x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 10:35:00 | 239.70 | 238.63 | 0.00 | T1 1.5R @ 239.70 |
| Target hit | 2023-11-10 15:20:00 | 242.75 | 240.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2023-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 09:40:00 | 247.45 | 245.78 | 0.00 | ORB-long ORB[243.30,246.30] vol=1.8x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 09:45:00 | 248.70 | 246.93 | 0.00 | T1 1.5R @ 248.70 |
| Target hit | 2023-11-13 11:30:00 | 248.25 | 248.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2023-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 09:35:00 | 255.15 | 253.85 | 0.00 | ORB-long ORB[250.95,254.20] vol=3.2x ATR=0.74 |
| Stop hit — per-position SL triggered | 2023-11-20 09:40:00 | 254.41 | 253.88 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-11-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 10:45:00 | 248.20 | 250.51 | 0.00 | ORB-short ORB[250.10,252.80] vol=1.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2023-11-21 10:55:00 | 248.78 | 250.33 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-11-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 10:55:00 | 252.45 | 251.31 | 0.00 | ORB-long ORB[250.10,251.15] vol=2.0x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-11-22 11:35:00 | 251.97 | 251.56 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-11-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:40:00 | 255.40 | 254.52 | 0.00 | ORB-long ORB[253.60,254.85] vol=2.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2023-11-23 09:45:00 | 254.73 | 254.58 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 09:30:00 | 256.35 | 254.90 | 0.00 | ORB-long ORB[253.00,254.85] vol=3.3x ATR=0.78 |
| Stop hit — per-position SL triggered | 2023-11-24 09:40:00 | 255.57 | 255.39 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 10:20:00 | 288.85 | 287.79 | 0.00 | ORB-long ORB[284.65,287.35] vol=1.7x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:35:00 | 290.72 | 288.39 | 0.00 | T1 1.5R @ 290.72 |
| Stop hit — per-position SL triggered | 2023-12-08 11:00:00 | 288.85 | 288.62 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 10:10:00 | 299.05 | 296.96 | 0.00 | ORB-long ORB[295.65,298.40] vol=1.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-12-15 10:15:00 | 298.24 | 297.07 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:05:00 | 303.05 | 304.08 | 0.00 | ORB-short ORB[303.20,306.05] vol=1.6x ATR=0.93 |
| Stop hit — per-position SL triggered | 2023-12-19 10:10:00 | 303.98 | 304.06 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:50:00 | 308.00 | 304.79 | 0.00 | ORB-long ORB[301.60,305.35] vol=2.8x ATR=1.51 |
| Stop hit — per-position SL triggered | 2023-12-22 10:05:00 | 306.49 | 305.51 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:30:00 | 308.30 | 306.79 | 0.00 | ORB-long ORB[304.60,307.85] vol=1.9x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 09:45:00 | 310.00 | 307.56 | 0.00 | T1 1.5R @ 310.00 |
| Stop hit — per-position SL triggered | 2023-12-26 10:15:00 | 308.30 | 308.08 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2023-12-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:40:00 | 306.75 | 309.17 | 0.00 | ORB-short ORB[308.65,312.25] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-12-27 10:50:00 | 307.55 | 308.91 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-12-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:40:00 | 311.45 | 309.77 | 0.00 | ORB-long ORB[307.75,311.00] vol=3.3x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 10:55:00 | 312.92 | 310.66 | 0.00 | T1 1.5R @ 312.92 |
| Target hit | 2023-12-28 15:20:00 | 313.60 | 312.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — SELL (started 2024-01-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:25:00 | 304.10 | 307.49 | 0.00 | ORB-short ORB[306.80,310.80] vol=3.2x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-01-02 10:30:00 | 305.22 | 307.22 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 11:15:00 | 315.30 | 321.28 | 0.00 | ORB-short ORB[321.00,325.65] vol=1.7x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-01-05 11:55:00 | 316.38 | 320.79 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 09:45:00 | 318.00 | 317.43 | 0.00 | ORB-long ORB[316.05,317.85] vol=4.8x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-01-09 09:50:00 | 317.11 | 317.45 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 09:40:00 | 311.20 | 311.66 | 0.00 | ORB-short ORB[311.35,315.30] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-01-12 09:50:00 | 312.05 | 311.67 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 10:00:00 | 301.95 | 303.68 | 0.00 | ORB-short ORB[307.55,311.30] vol=1.8x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-01-18 10:05:00 | 303.57 | 303.62 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-01-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 10:45:00 | 314.20 | 312.72 | 0.00 | ORB-long ORB[309.00,312.20] vol=1.5x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-01-20 11:30:00 | 313.32 | 313.07 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-31 09:35:00 | 316.35 | 317.77 | 0.00 | ORB-short ORB[316.65,320.00] vol=1.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-01-31 09:40:00 | 317.53 | 317.72 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-02-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 10:25:00 | 325.30 | 321.74 | 0.00 | ORB-long ORB[317.50,321.00] vol=2.3x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-02-01 10:35:00 | 323.89 | 322.02 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-13 09:45:00 | 316.70 | 318.88 | 0.00 | ORB-short ORB[317.65,321.50] vol=2.1x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-02-13 09:50:00 | 318.56 | 318.85 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-02-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 10:00:00 | 343.10 | 340.38 | 0.00 | ORB-long ORB[337.80,340.60] vol=4.4x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-02-20 11:00:00 | 341.91 | 341.95 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 10:35:00 | 335.55 | 336.26 | 0.00 | ORB-short ORB[336.20,338.50] vol=2.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-02-26 11:05:00 | 336.37 | 336.18 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 333.65 | 334.95 | 0.00 | ORB-short ORB[334.45,336.50] vol=2.1x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:55:00 | 332.75 | 334.77 | 0.00 | T1 1.5R @ 332.75 |
| Stop hit — per-position SL triggered | 2024-02-28 11:00:00 | 333.65 | 334.56 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 10:55:00 | 357.75 | 355.86 | 0.00 | ORB-long ORB[353.55,357.00] vol=1.9x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-03-05 11:00:00 | 356.67 | 355.96 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:35:00 | 313.00 | 314.94 | 0.00 | ORB-short ORB[314.00,317.90] vol=1.6x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-03-19 09:40:00 | 314.16 | 314.76 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 09:40:00 | 321.60 | 319.41 | 0.00 | ORB-long ORB[317.00,320.00] vol=2.3x ATR=1.30 |
| Stop hit — per-position SL triggered | 2024-03-21 09:45:00 | 320.30 | 319.69 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 11:15:00 | 340.25 | 338.40 | 0.00 | ORB-long ORB[336.40,339.00] vol=3.1x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 11:50:00 | 341.61 | 338.91 | 0.00 | T1 1.5R @ 341.61 |
| Target hit | 2024-04-01 15:20:00 | 342.30 | 340.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — BUY (started 2024-04-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 09:45:00 | 360.50 | 358.00 | 0.00 | ORB-long ORB[354.80,359.40] vol=1.6x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-04-04 09:50:00 | 358.77 | 358.15 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 09:35:00 | 366.00 | 365.06 | 0.00 | ORB-long ORB[362.95,365.40] vol=3.6x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-04-10 09:40:00 | 364.92 | 365.10 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 09:40:00 | 347.45 | 348.74 | 0.00 | ORB-short ORB[350.50,355.00] vol=12.4x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 09:50:00 | 345.33 | 347.61 | 0.00 | T1 1.5R @ 345.33 |
| Stop hit — per-position SL triggered | 2024-04-22 10:00:00 | 347.45 | 347.57 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-04-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:30:00 | 350.15 | 348.90 | 0.00 | ORB-long ORB[347.30,349.80] vol=1.6x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 12:25:00 | 351.18 | 349.67 | 0.00 | T1 1.5R @ 351.18 |
| Target hit | 2024-04-24 15:20:00 | 352.25 | 350.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 91 — BUY (started 2024-04-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 10:50:00 | 351.55 | 351.36 | 0.00 | ORB-long ORB[349.10,351.35] vol=1.9x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 11:05:00 | 352.72 | 351.42 | 0.00 | T1 1.5R @ 352.72 |
| Stop hit — per-position SL triggered | 2024-04-25 11:50:00 | 351.55 | 351.56 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 10:45:00 | 361.30 | 363.01 | 0.00 | ORB-short ORB[362.65,365.45] vol=1.9x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 10:55:00 | 360.13 | 362.78 | 0.00 | T1 1.5R @ 360.13 |
| Stop hit — per-position SL triggered | 2024-04-30 11:40:00 | 361.30 | 361.79 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-05-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 09:40:00 | 367.85 | 365.67 | 0.00 | ORB-long ORB[363.20,366.35] vol=2.3x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-05-02 09:50:00 | 366.65 | 366.56 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:35:00 | 359.70 | 364.46 | 0.00 | ORB-short ORB[364.10,368.85] vol=1.5x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 10:05:00 | 356.42 | 362.38 | 0.00 | T1 1.5R @ 356.42 |
| Target hit | 2024-05-06 15:20:00 | 356.85 | 358.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 95 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 11:15:00 | 349.40 | 354.08 | 0.00 | ORB-short ORB[354.50,358.45] vol=2.5x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:35:00 | 347.35 | 353.54 | 0.00 | T1 1.5R @ 347.35 |
| Stop hit — per-position SL triggered | 2024-05-07 11:45:00 | 349.40 | 353.13 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 10:00:00 | 354.30 | 350.41 | 0.00 | ORB-long ORB[345.20,350.20] vol=1.6x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-05-08 10:20:00 | 352.92 | 351.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-24 11:05:00 | 174.95 | 2023-05-24 11:45:00 | 174.66 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-05-29 11:15:00 | 176.90 | 2023-05-29 11:35:00 | 177.41 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-05-29 11:15:00 | 176.90 | 2023-05-29 12:05:00 | 176.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-31 11:15:00 | 173.95 | 2023-05-31 11:45:00 | 173.44 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-05-31 11:15:00 | 173.95 | 2023-05-31 13:35:00 | 173.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-01 11:15:00 | 174.00 | 2023-06-01 12:10:00 | 173.68 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-02 11:05:00 | 174.00 | 2023-06-02 11:25:00 | 173.48 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-06-02 11:05:00 | 174.00 | 2023-06-02 11:45:00 | 174.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-05 11:05:00 | 175.00 | 2023-06-05 11:20:00 | 175.30 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-06-12 10:20:00 | 184.60 | 2023-06-12 10:50:00 | 185.27 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-06-12 10:20:00 | 184.60 | 2023-06-12 15:20:00 | 185.55 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2023-06-13 10:25:00 | 185.80 | 2023-06-13 11:25:00 | 185.26 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-06-13 10:25:00 | 185.80 | 2023-06-13 11:45:00 | 185.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-14 10:30:00 | 187.05 | 2023-06-14 11:05:00 | 187.62 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-06-14 10:30:00 | 187.05 | 2023-06-14 14:45:00 | 187.35 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2023-06-22 09:45:00 | 185.95 | 2023-06-22 10:25:00 | 186.39 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-06-23 09:30:00 | 186.00 | 2023-06-23 09:50:00 | 186.82 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-06-23 09:30:00 | 186.00 | 2023-06-23 13:20:00 | 186.50 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2023-06-28 10:15:00 | 186.25 | 2023-06-28 10:20:00 | 185.96 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-07-03 10:00:00 | 188.90 | 2023-07-03 10:20:00 | 189.35 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-06 10:15:00 | 194.60 | 2023-07-06 10:25:00 | 195.16 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-07-06 10:15:00 | 194.60 | 2023-07-06 13:30:00 | 195.80 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2023-07-12 10:55:00 | 194.70 | 2023-07-12 11:10:00 | 194.32 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-07-13 10:50:00 | 190.20 | 2023-07-13 11:20:00 | 190.54 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-17 11:00:00 | 188.60 | 2023-07-17 11:15:00 | 188.17 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-18 09:55:00 | 188.10 | 2023-07-18 10:15:00 | 188.72 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-07-18 09:55:00 | 188.10 | 2023-07-18 10:20:00 | 188.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-20 11:15:00 | 192.15 | 2023-07-20 11:25:00 | 192.46 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-07-24 11:10:00 | 194.35 | 2023-07-24 11:40:00 | 193.82 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-24 11:10:00 | 194.35 | 2023-07-24 12:10:00 | 194.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-25 09:45:00 | 198.50 | 2023-07-25 09:55:00 | 199.25 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-07-25 09:45:00 | 198.50 | 2023-07-25 10:00:00 | 198.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-28 10:15:00 | 205.80 | 2023-07-28 10:20:00 | 206.93 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2023-07-28 10:15:00 | 205.80 | 2023-07-28 15:20:00 | 210.15 | TARGET_HIT | 0.50 | 2.11% |
| BUY | retest1 | 2023-08-07 10:55:00 | 220.65 | 2023-08-07 11:25:00 | 220.16 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-08 09:35:00 | 218.30 | 2023-08-08 09:40:00 | 218.79 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-09 10:55:00 | 217.00 | 2023-08-09 11:20:00 | 217.62 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-08-10 11:00:00 | 218.00 | 2023-08-10 12:40:00 | 218.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-08-16 10:45:00 | 214.30 | 2023-08-16 10:50:00 | 213.88 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-17 11:15:00 | 216.45 | 2023-08-17 11:55:00 | 215.79 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-08-17 11:15:00 | 216.45 | 2023-08-17 12:40:00 | 216.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-18 11:10:00 | 215.65 | 2023-08-18 12:10:00 | 215.14 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-08-18 11:10:00 | 215.65 | 2023-08-18 14:00:00 | 215.00 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2023-08-21 09:40:00 | 218.10 | 2023-08-21 09:50:00 | 217.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-08-22 09:30:00 | 220.80 | 2023-08-22 10:05:00 | 221.46 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-08-22 09:30:00 | 220.80 | 2023-08-22 10:40:00 | 220.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-25 10:30:00 | 219.45 | 2023-08-25 10:40:00 | 219.89 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-29 09:30:00 | 220.40 | 2023-08-29 12:30:00 | 221.03 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-08-29 09:30:00 | 220.40 | 2023-08-29 15:20:00 | 221.25 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2023-09-01 11:15:00 | 225.25 | 2023-09-01 11:35:00 | 226.37 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-09-01 11:15:00 | 225.25 | 2023-09-01 15:20:00 | 231.20 | TARGET_HIT | 0.50 | 2.64% |
| BUY | retest1 | 2023-09-04 10:50:00 | 234.50 | 2023-09-04 10:55:00 | 233.60 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-05 10:40:00 | 233.80 | 2023-09-05 12:20:00 | 232.86 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-09-05 10:40:00 | 233.80 | 2023-09-05 14:35:00 | 233.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-06 09:30:00 | 233.15 | 2023-09-06 09:55:00 | 232.37 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-09-06 09:30:00 | 233.15 | 2023-09-06 12:15:00 | 230.90 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2023-09-07 10:35:00 | 230.00 | 2023-09-07 10:45:00 | 230.52 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-09-12 10:30:00 | 239.35 | 2023-09-12 10:35:00 | 240.57 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2023-09-18 10:55:00 | 240.70 | 2023-09-18 11:05:00 | 239.99 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-26 09:40:00 | 242.15 | 2023-09-26 10:15:00 | 243.15 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-09-26 09:40:00 | 242.15 | 2023-09-26 10:25:00 | 242.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-03 11:05:00 | 242.65 | 2023-10-03 11:25:00 | 243.40 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-10-05 10:45:00 | 232.80 | 2023-10-05 11:25:00 | 233.46 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-10-09 10:00:00 | 234.00 | 2023-10-09 12:15:00 | 233.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-10-10 09:50:00 | 234.70 | 2023-10-10 10:10:00 | 233.76 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-10-10 09:50:00 | 234.70 | 2023-10-10 10:15:00 | 234.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-11 09:35:00 | 238.60 | 2023-10-11 09:55:00 | 237.97 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-10-12 10:40:00 | 242.10 | 2023-10-12 10:55:00 | 241.51 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-13 09:55:00 | 243.15 | 2023-10-13 10:00:00 | 242.54 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-10-17 10:10:00 | 243.85 | 2023-10-17 10:15:00 | 244.34 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-10-18 10:45:00 | 244.40 | 2023-10-18 11:15:00 | 243.82 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-10-18 10:45:00 | 244.40 | 2023-10-18 11:35:00 | 244.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-23 10:55:00 | 239.20 | 2023-10-23 11:25:00 | 238.17 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-10-23 10:55:00 | 239.20 | 2023-10-23 15:20:00 | 235.40 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2023-10-27 09:35:00 | 236.70 | 2023-10-27 10:10:00 | 235.65 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-11-06 10:20:00 | 236.90 | 2023-11-06 10:45:00 | 236.51 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-11-07 11:05:00 | 238.35 | 2023-11-07 11:20:00 | 237.89 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-08 10:55:00 | 237.00 | 2023-11-08 11:50:00 | 236.47 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-11-08 10:55:00 | 237.00 | 2023-11-08 12:20:00 | 237.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-09 11:10:00 | 238.25 | 2023-11-09 11:45:00 | 239.26 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-11-09 11:10:00 | 238.25 | 2023-11-09 13:00:00 | 238.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-10 09:45:00 | 238.85 | 2023-11-10 10:35:00 | 239.70 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-11-10 09:45:00 | 238.85 | 2023-11-10 15:20:00 | 242.75 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2023-11-13 09:40:00 | 247.45 | 2023-11-13 09:45:00 | 248.70 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-11-13 09:40:00 | 247.45 | 2023-11-13 11:30:00 | 248.25 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2023-11-20 09:35:00 | 255.15 | 2023-11-20 09:40:00 | 254.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-21 10:45:00 | 248.20 | 2023-11-21 10:55:00 | 248.78 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-11-22 10:55:00 | 252.45 | 2023-11-22 11:35:00 | 251.97 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-23 09:40:00 | 255.40 | 2023-11-23 09:45:00 | 254.73 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-11-24 09:30:00 | 256.35 | 2023-11-24 09:40:00 | 255.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-12-08 10:20:00 | 288.85 | 2023-12-08 10:35:00 | 290.72 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2023-12-08 10:20:00 | 288.85 | 2023-12-08 11:00:00 | 288.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-15 10:10:00 | 299.05 | 2023-12-15 10:15:00 | 298.24 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-12-19 10:05:00 | 303.05 | 2023-12-19 10:10:00 | 303.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-12-22 09:50:00 | 308.00 | 2023-12-22 10:05:00 | 306.49 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2023-12-26 09:30:00 | 308.30 | 2023-12-26 09:45:00 | 310.00 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2023-12-26 09:30:00 | 308.30 | 2023-12-26 10:15:00 | 308.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-27 10:40:00 | 306.75 | 2023-12-27 10:50:00 | 307.55 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-28 10:40:00 | 311.45 | 2023-12-28 10:55:00 | 312.92 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-12-28 10:40:00 | 311.45 | 2023-12-28 15:20:00 | 313.60 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2024-01-02 10:25:00 | 304.10 | 2024-01-02 10:30:00 | 305.22 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-01-05 11:15:00 | 315.30 | 2024-01-05 11:55:00 | 316.38 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-01-09 09:45:00 | 318.00 | 2024-01-09 09:50:00 | 317.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-12 09:40:00 | 311.20 | 2024-01-12 09:50:00 | 312.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-18 10:00:00 | 301.95 | 2024-01-18 10:05:00 | 303.57 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-01-20 10:45:00 | 314.20 | 2024-01-20 11:30:00 | 313.32 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-31 09:35:00 | 316.35 | 2024-01-31 09:40:00 | 317.53 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-02-01 10:25:00 | 325.30 | 2024-02-01 10:35:00 | 323.89 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-02-13 09:45:00 | 316.70 | 2024-02-13 09:50:00 | 318.56 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-02-20 10:00:00 | 343.10 | 2024-02-20 11:00:00 | 341.91 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-02-26 10:35:00 | 335.55 | 2024-02-26 11:05:00 | 336.37 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-02-28 10:50:00 | 333.65 | 2024-02-28 10:55:00 | 332.75 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-02-28 10:50:00 | 333.65 | 2024-02-28 11:00:00 | 333.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-05 10:55:00 | 357.75 | 2024-03-05 11:00:00 | 356.67 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-19 09:35:00 | 313.00 | 2024-03-19 09:40:00 | 314.16 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-03-21 09:40:00 | 321.60 | 2024-03-21 09:45:00 | 320.30 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-04-01 11:15:00 | 340.25 | 2024-04-01 11:50:00 | 341.61 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-04-01 11:15:00 | 340.25 | 2024-04-01 15:20:00 | 342.30 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2024-04-04 09:45:00 | 360.50 | 2024-04-04 09:50:00 | 358.77 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-04-10 09:35:00 | 366.00 | 2024-04-10 09:40:00 | 364.92 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-04-22 09:40:00 | 347.45 | 2024-04-22 09:50:00 | 345.33 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-04-22 09:40:00 | 347.45 | 2024-04-22 10:00:00 | 347.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-24 10:30:00 | 350.15 | 2024-04-24 12:25:00 | 351.18 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-04-24 10:30:00 | 350.15 | 2024-04-24 15:20:00 | 352.25 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2024-04-25 10:50:00 | 351.55 | 2024-04-25 11:05:00 | 352.72 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-04-25 10:50:00 | 351.55 | 2024-04-25 11:50:00 | 351.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-30 10:45:00 | 361.30 | 2024-04-30 10:55:00 | 360.13 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-04-30 10:45:00 | 361.30 | 2024-04-30 11:40:00 | 361.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-02 09:40:00 | 367.85 | 2024-05-02 09:50:00 | 366.65 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-06 09:35:00 | 359.70 | 2024-05-06 10:05:00 | 356.42 | PARTIAL | 0.50 | 0.91% |
| SELL | retest1 | 2024-05-06 09:35:00 | 359.70 | 2024-05-06 15:20:00 | 356.85 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2024-05-07 11:15:00 | 349.40 | 2024-05-07 11:35:00 | 347.35 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-05-07 11:15:00 | 349.40 | 2024-05-07 11:45:00 | 349.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-08 10:00:00 | 354.30 | 2024-05-08 10:20:00 | 352.92 | STOP_HIT | 1.00 | -0.39% |
