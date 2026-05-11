# IRCON International Ltd. (IRCON)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 158.99
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
| ENTRY1 | 58 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 48
- **Target hits / Stop hits / Partials:** 10 / 48 / 19
- **Avg / median % per leg:** 0.14% / -0.22%
- **Sum % (uncompounded):** 10.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 12 | 30.8% | 5 | 27 | 7 | 0.05% | 2.0% |
| BUY @ 2nd Alert (retest1) | 39 | 12 | 30.8% | 5 | 27 | 7 | 0.05% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 17 | 44.7% | 5 | 21 | 12 | 0.24% | 8.9% |
| SELL @ 2nd Alert (retest1) | 38 | 17 | 44.7% | 5 | 21 | 12 | 0.24% | 8.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 77 | 29 | 37.7% | 10 | 48 | 19 | 0.14% | 11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:15:00 | 189.73 | 187.56 | 0.00 | ORB-long ORB[186.01,188.83] vol=9.5x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-05-27 11:20:00 | 188.94 | 187.74 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 189.15 | 190.31 | 0.00 | ORB-short ORB[190.00,192.29] vol=1.7x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-05-29 10:45:00 | 189.82 | 189.94 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:40:00 | 195.93 | 195.22 | 0.00 | ORB-long ORB[192.92,195.85] vol=4.2x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-06-03 09:50:00 | 194.95 | 195.25 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:20:00 | 196.21 | 193.58 | 0.00 | ORB-long ORB[192.30,194.49] vol=4.3x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-06-04 10:30:00 | 195.16 | 194.04 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:05:00 | 218.02 | 216.62 | 0.00 | ORB-long ORB[214.65,217.74] vol=3.4x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-06-11 10:20:00 | 217.00 | 216.87 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:00:00 | 204.85 | 203.35 | 0.00 | ORB-long ORB[201.61,204.41] vol=2.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-06-25 10:05:00 | 203.93 | 203.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 200.35 | 199.32 | 0.00 | ORB-long ORB[197.70,199.89] vol=2.4x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-07-04 09:40:00 | 199.65 | 199.42 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 190.90 | 191.64 | 0.00 | ORB-short ORB[191.00,192.87] vol=1.6x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 09:45:00 | 190.30 | 191.41 | 0.00 | T1 1.5R @ 190.30 |
| Stop hit — per-position SL triggered | 2025-07-16 10:25:00 | 190.90 | 191.19 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 10:25:00 | 190.30 | 191.02 | 0.00 | ORB-short ORB[190.70,192.08] vol=3.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:30:00 | 189.75 | 190.91 | 0.00 | T1 1.5R @ 189.75 |
| Stop hit — per-position SL triggered | 2025-07-17 11:40:00 | 190.30 | 190.68 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:45:00 | 184.12 | 184.83 | 0.00 | ORB-short ORB[184.31,186.10] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:30:00 | 183.33 | 184.49 | 0.00 | T1 1.5R @ 183.33 |
| Target hit | 2025-07-25 15:20:00 | 180.70 | 183.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 11:15:00 | 176.75 | 176.34 | 0.00 | ORB-long ORB[174.10,176.64] vol=1.7x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 13:05:00 | 177.64 | 176.46 | 0.00 | T1 1.5R @ 177.64 |
| Target hit | 2025-07-29 15:20:00 | 181.76 | 177.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-08-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 09:30:00 | 176.40 | 175.40 | 0.00 | ORB-long ORB[173.74,175.90] vol=2.1x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-08-04 09:35:00 | 175.72 | 175.42 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-08-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:05:00 | 173.46 | 174.83 | 0.00 | ORB-short ORB[175.36,176.83] vol=1.5x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-08-06 10:20:00 | 174.05 | 174.70 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-08-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:00:00 | 169.02 | 168.24 | 0.00 | ORB-long ORB[167.35,168.82] vol=2.2x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-08-13 10:20:00 | 168.31 | 168.28 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:00:00 | 168.48 | 167.45 | 0.00 | ORB-long ORB[166.20,168.26] vol=2.6x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-08-18 10:30:00 | 167.88 | 167.72 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:25:00 | 169.10 | 168.12 | 0.00 | ORB-long ORB[166.90,168.55] vol=2.8x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-08-19 10:35:00 | 168.56 | 168.15 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 11:15:00 | 162.28 | 161.18 | 0.00 | ORB-long ORB[160.36,161.91] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-08-29 11:55:00 | 161.80 | 161.32 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 163.86 | 162.37 | 0.00 | ORB-long ORB[160.53,162.87] vol=3.7x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-09-01 10:05:00 | 163.29 | 162.66 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-09-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:35:00 | 172.47 | 174.18 | 0.00 | ORB-short ORB[173.61,175.28] vol=2.0x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-09-03 12:35:00 | 173.23 | 173.81 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:40:00 | 171.86 | 170.65 | 0.00 | ORB-long ORB[169.35,171.17] vol=1.8x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-09-08 09:55:00 | 171.25 | 170.96 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:30:00 | 173.60 | 172.44 | 0.00 | ORB-long ORB[171.10,173.26] vol=3.9x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 09:35:00 | 174.60 | 174.02 | 0.00 | T1 1.5R @ 174.60 |
| Target hit | 2025-09-09 09:55:00 | 173.98 | 174.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 175.56 | 174.48 | 0.00 | ORB-long ORB[173.40,175.55] vol=2.2x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-09-10 09:50:00 | 174.91 | 174.99 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 09:30:00 | 170.90 | 171.74 | 0.00 | ORB-short ORB[171.00,173.00] vol=2.4x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-09-11 09:35:00 | 171.42 | 171.71 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 180.61 | 181.51 | 0.00 | ORB-short ORB[181.01,182.48] vol=1.5x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-09-23 09:40:00 | 181.27 | 181.40 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 11:00:00 | 173.10 | 171.52 | 0.00 | ORB-long ORB[169.60,171.49] vol=2.2x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-09-29 11:10:00 | 172.48 | 171.74 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 173.72 | 174.30 | 0.00 | ORB-short ORB[173.79,175.80] vol=1.5x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-10-13 09:45:00 | 174.21 | 174.24 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:00:00 | 173.70 | 172.77 | 0.00 | ORB-long ORB[171.72,172.80] vol=3.8x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 12:45:00 | 174.31 | 173.16 | 0.00 | T1 1.5R @ 174.31 |
| Stop hit — per-position SL triggered | 2025-10-15 13:45:00 | 173.70 | 173.48 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 175.13 | 174.46 | 0.00 | ORB-long ORB[173.67,174.82] vol=2.4x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-10-16 09:35:00 | 174.69 | 174.51 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:45:00 | 170.16 | 171.02 | 0.00 | ORB-short ORB[170.64,172.96] vol=1.9x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:25:00 | 169.42 | 170.62 | 0.00 | T1 1.5R @ 169.42 |
| Stop hit — per-position SL triggered | 2025-10-17 10:35:00 | 170.16 | 170.57 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 171.08 | 170.49 | 0.00 | ORB-long ORB[169.51,170.77] vol=1.7x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-10-24 09:45:00 | 170.65 | 170.63 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 10:30:00 | 171.79 | 170.74 | 0.00 | ORB-long ORB[169.81,171.15] vol=4.7x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-10-28 10:35:00 | 171.29 | 170.77 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 170.06 | 169.38 | 0.00 | ORB-long ORB[168.80,169.71] vol=2.0x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:45:00 | 170.80 | 169.65 | 0.00 | T1 1.5R @ 170.80 |
| Stop hit — per-position SL triggered | 2025-10-29 10:50:00 | 170.06 | 169.68 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-11-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:05:00 | 167.87 | 168.66 | 0.00 | ORB-short ORB[168.40,169.89] vol=2.0x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-11-04 10:40:00 | 168.24 | 168.46 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 166.00 | 166.94 | 0.00 | ORB-short ORB[166.10,168.44] vol=1.6x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:00:00 | 165.40 | 166.46 | 0.00 | T1 1.5R @ 165.40 |
| Target hit | 2025-11-06 15:20:00 | 163.28 | 164.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2025-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:35:00 | 164.58 | 164.00 | 0.00 | ORB-long ORB[163.35,164.48] vol=1.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-11-10 09:40:00 | 164.14 | 164.01 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:20:00 | 164.15 | 163.29 | 0.00 | ORB-long ORB[162.60,163.99] vol=2.3x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-11-11 10:30:00 | 163.64 | 163.32 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:30:00 | 166.00 | 166.77 | 0.00 | ORB-short ORB[166.31,168.14] vol=1.8x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 09:40:00 | 165.36 | 166.24 | 0.00 | T1 1.5R @ 165.36 |
| Stop hit — per-position SL triggered | 2025-11-20 10:20:00 | 166.00 | 166.03 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-12-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:20:00 | 155.60 | 156.55 | 0.00 | ORB-short ORB[156.56,158.20] vol=2.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-12-03 10:25:00 | 155.99 | 156.50 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:55:00 | 151.61 | 152.56 | 0.00 | ORB-short ORB[152.91,154.36] vol=3.4x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 152.01 | 152.46 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-12-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:50:00 | 150.10 | 150.75 | 0.00 | ORB-short ORB[150.30,151.70] vol=2.8x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:05:00 | 149.45 | 150.52 | 0.00 | T1 1.5R @ 149.45 |
| Target hit | 2025-12-08 15:20:00 | 146.60 | 148.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 154.37 | 153.27 | 0.00 | ORB-long ORB[152.00,154.20] vol=1.6x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-12-10 10:10:00 | 153.64 | 153.68 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:15:00 | 154.52 | 155.17 | 0.00 | ORB-short ORB[155.10,156.20] vol=4.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 14:15:00 | 153.99 | 154.83 | 0.00 | T1 1.5R @ 153.99 |
| Target hit | 2025-12-16 15:20:00 | 153.54 | 154.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:35:00 | 149.55 | 150.05 | 0.00 | ORB-short ORB[149.60,150.98] vol=2.1x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:45:00 | 148.85 | 149.82 | 0.00 | T1 1.5R @ 148.85 |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 149.55 | 149.44 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 177.09 | 172.76 | 0.00 | ORB-long ORB[170.40,172.77] vol=6.2x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-12-30 10:55:00 | 176.06 | 173.26 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 176.00 | 177.37 | 0.00 | ORB-short ORB[176.45,178.60] vol=1.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-01-08 09:40:00 | 176.55 | 177.23 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-01-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 11:10:00 | 159.70 | 158.25 | 0.00 | ORB-long ORB[156.40,158.79] vol=3.2x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 159.06 | 158.30 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 160.42 | 159.22 | 0.00 | ORB-long ORB[158.30,160.40] vol=1.8x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:35:00 | 161.59 | 159.69 | 0.00 | T1 1.5R @ 161.59 |
| Target hit | 2026-01-30 13:25:00 | 162.82 | 162.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2026-02-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:30:00 | 155.18 | 154.35 | 0.00 | ORB-long ORB[153.36,155.00] vol=1.6x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 10:05:00 | 156.21 | 154.79 | 0.00 | T1 1.5R @ 156.21 |
| Target hit | 2026-02-04 15:20:00 | 157.38 | 156.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 153.24 | 152.23 | 0.00 | ORB-long ORB[151.05,152.89] vol=2.2x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-02-16 09:35:00 | 152.70 | 152.28 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-02-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:20:00 | 151.97 | 152.81 | 0.00 | ORB-short ORB[152.40,153.90] vol=2.5x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-02-18 12:00:00 | 152.33 | 152.57 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-02-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:25:00 | 151.55 | 152.23 | 0.00 | ORB-short ORB[151.95,153.38] vol=1.8x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:35:00 | 150.97 | 151.95 | 0.00 | T1 1.5R @ 150.97 |
| Target hit | 2026-02-19 15:20:00 | 149.00 | 151.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2026-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:05:00 | 149.21 | 150.32 | 0.00 | ORB-short ORB[150.16,151.48] vol=1.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-02-23 11:35:00 | 149.54 | 150.20 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 147.80 | 148.39 | 0.00 | ORB-short ORB[148.05,149.50] vol=1.5x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 148.23 | 148.35 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 148.06 | 148.61 | 0.00 | ORB-short ORB[148.29,149.40] vol=1.7x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-02-25 10:05:00 | 148.43 | 148.29 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 148.19 | 147.58 | 0.00 | ORB-long ORB[146.52,148.18] vol=1.7x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 147.76 | 147.76 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 131.60 | 131.90 | 0.00 | ORB-short ORB[131.65,133.00] vol=1.8x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:40:00 | 130.99 | 131.83 | 0.00 | T1 1.5R @ 130.99 |
| Stop hit — per-position SL triggered | 2026-03-05 12:20:00 | 131.60 | 131.74 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 155.36 | 154.40 | 0.00 | ORB-long ORB[153.13,155.15] vol=2.1x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:35:00 | 156.37 | 156.59 | 0.00 | T1 1.5R @ 156.37 |
| Target hit | 2026-05-05 10:35:00 | 157.80 | 158.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — SELL (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 159.94 | 160.73 | 0.00 | ORB-short ORB[160.30,161.85] vol=1.5x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:40:00 | 158.95 | 160.35 | 0.00 | T1 1.5R @ 158.95 |
| Stop hit — per-position SL triggered | 2026-05-08 11:40:00 | 159.94 | 159.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-27 11:15:00 | 189.73 | 2025-05-27 11:20:00 | 188.94 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-05-29 09:35:00 | 189.15 | 2025-05-29 10:45:00 | 189.82 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-06-03 09:40:00 | 195.93 | 2025-06-03 09:50:00 | 194.95 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-06-04 10:20:00 | 196.21 | 2025-06-04 10:30:00 | 195.16 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-06-11 10:05:00 | 218.02 | 2025-06-11 10:20:00 | 217.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-06-25 10:00:00 | 204.85 | 2025-06-25 10:05:00 | 203.93 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-07-04 09:30:00 | 200.35 | 2025-07-04 09:40:00 | 199.65 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-16 09:30:00 | 190.90 | 2025-07-16 09:45:00 | 190.30 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-16 09:30:00 | 190.90 | 2025-07-16 10:25:00 | 190.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-17 10:25:00 | 190.30 | 2025-07-17 10:30:00 | 189.75 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-07-17 10:25:00 | 190.30 | 2025-07-17 11:40:00 | 190.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 09:45:00 | 184.12 | 2025-07-25 10:30:00 | 183.33 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-07-25 09:45:00 | 184.12 | 2025-07-25 15:20:00 | 180.70 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2025-07-29 11:15:00 | 176.75 | 2025-07-29 13:05:00 | 177.64 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-07-29 11:15:00 | 176.75 | 2025-07-29 15:20:00 | 181.76 | TARGET_HIT | 0.50 | 2.83% |
| BUY | retest1 | 2025-08-04 09:30:00 | 176.40 | 2025-08-04 09:35:00 | 175.72 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-08-06 10:05:00 | 173.46 | 2025-08-06 10:20:00 | 174.05 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-08-13 10:00:00 | 169.02 | 2025-08-13 10:20:00 | 168.31 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-08-18 10:00:00 | 168.48 | 2025-08-18 10:30:00 | 167.88 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-08-19 10:25:00 | 169.10 | 2025-08-19 10:35:00 | 168.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-29 11:15:00 | 162.28 | 2025-08-29 11:55:00 | 161.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-01 09:45:00 | 163.86 | 2025-09-01 10:05:00 | 163.29 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-09-03 10:35:00 | 172.47 | 2025-09-03 12:35:00 | 173.23 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-09-08 09:40:00 | 171.86 | 2025-09-08 09:55:00 | 171.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-09 09:30:00 | 173.60 | 2025-09-09 09:35:00 | 174.60 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-09-09 09:30:00 | 173.60 | 2025-09-09 09:55:00 | 173.98 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-09-10 09:40:00 | 175.56 | 2025-09-10 09:50:00 | 174.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-09-11 09:30:00 | 170.90 | 2025-09-11 09:35:00 | 171.42 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-23 09:30:00 | 180.61 | 2025-09-23 09:40:00 | 181.27 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-29 11:00:00 | 173.10 | 2025-09-29 11:10:00 | 172.48 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-13 09:30:00 | 173.72 | 2025-10-13 09:45:00 | 174.21 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-15 11:00:00 | 173.70 | 2025-10-15 12:45:00 | 174.31 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-15 11:00:00 | 173.70 | 2025-10-15 13:45:00 | 173.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-16 09:30:00 | 175.13 | 2025-10-16 09:35:00 | 174.69 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-17 09:45:00 | 170.16 | 2025-10-17 10:25:00 | 169.42 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-17 09:45:00 | 170.16 | 2025-10-17 10:35:00 | 170.16 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-24 09:30:00 | 171.08 | 2025-10-24 09:45:00 | 170.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-28 10:30:00 | 171.79 | 2025-10-28 10:35:00 | 171.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-29 10:25:00 | 170.06 | 2025-10-29 10:45:00 | 170.80 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-29 10:25:00 | 170.06 | 2025-10-29 10:50:00 | 170.06 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 10:05:00 | 167.87 | 2025-11-04 10:40:00 | 168.24 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-06 09:30:00 | 166.00 | 2025-11-06 10:00:00 | 165.40 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-06 09:30:00 | 166.00 | 2025-11-06 15:20:00 | 163.28 | TARGET_HIT | 0.50 | 1.64% |
| BUY | retest1 | 2025-11-10 09:35:00 | 164.58 | 2025-11-10 09:40:00 | 164.14 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-11 10:20:00 | 164.15 | 2025-11-11 10:30:00 | 163.64 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-20 09:30:00 | 166.00 | 2025-11-20 09:40:00 | 165.36 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-20 09:30:00 | 166.00 | 2025-11-20 10:20:00 | 166.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 10:20:00 | 155.60 | 2025-12-03 10:25:00 | 155.99 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-05 09:55:00 | 151.61 | 2025-12-05 10:00:00 | 152.01 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-08 09:50:00 | 150.10 | 2025-12-08 10:05:00 | 149.45 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-08 09:50:00 | 150.10 | 2025-12-08 15:20:00 | 146.60 | TARGET_HIT | 0.50 | 2.33% |
| BUY | retest1 | 2025-12-10 09:30:00 | 154.37 | 2025-12-10 10:10:00 | 153.64 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-12-16 11:15:00 | 154.52 | 2025-12-16 14:15:00 | 153.99 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-16 11:15:00 | 154.52 | 2025-12-16 15:20:00 | 153.54 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2025-12-18 09:35:00 | 149.55 | 2025-12-18 09:45:00 | 148.85 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-18 09:35:00 | 149.55 | 2025-12-18 10:15:00 | 149.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:50:00 | 177.09 | 2025-12-30 10:55:00 | 176.06 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-01-08 09:30:00 | 176.00 | 2026-01-08 09:40:00 | 176.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-01-28 11:10:00 | 159.70 | 2026-01-28 11:15:00 | 159.06 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-30 09:30:00 | 160.42 | 2026-01-30 09:35:00 | 161.59 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-01-30 09:30:00 | 160.42 | 2026-01-30 13:25:00 | 162.82 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2026-02-04 09:30:00 | 155.18 | 2026-02-04 10:05:00 | 156.21 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-02-04 09:30:00 | 155.18 | 2026-02-04 15:20:00 | 157.38 | TARGET_HIT | 0.50 | 1.42% |
| BUY | retest1 | 2026-02-16 09:30:00 | 153.24 | 2026-02-16 09:35:00 | 152.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-18 10:20:00 | 151.97 | 2026-02-18 12:00:00 | 152.33 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-19 10:25:00 | 151.55 | 2026-02-19 11:35:00 | 150.97 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-19 10:25:00 | 151.55 | 2026-02-19 15:20:00 | 149.00 | TARGET_HIT | 0.50 | 1.68% |
| SELL | retest1 | 2026-02-23 11:05:00 | 149.21 | 2026-02-23 11:35:00 | 149.54 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-24 09:30:00 | 147.80 | 2026-02-24 09:35:00 | 148.23 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-25 09:30:00 | 148.06 | 2026-02-25 10:05:00 | 148.43 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-26 09:35:00 | 148.19 | 2026-02-26 09:55:00 | 147.76 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-05 11:15:00 | 131.60 | 2026-03-05 11:40:00 | 130.99 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-05 11:15:00 | 131.60 | 2026-03-05 12:20:00 | 131.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:30:00 | 155.36 | 2026-05-05 09:35:00 | 156.37 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-05-05 09:30:00 | 155.36 | 2026-05-05 10:35:00 | 157.80 | TARGET_HIT | 0.50 | 1.57% |
| SELL | retest1 | 2026-05-08 09:30:00 | 159.94 | 2026-05-08 09:40:00 | 158.95 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-05-08 09:30:00 | 159.94 | 2026-05-08 11:40:00 | 159.94 | STOP_HIT | 0.50 | 0.00% |
