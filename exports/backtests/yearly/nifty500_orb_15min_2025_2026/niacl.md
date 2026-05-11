# The New India Assurance Company Ltd. (NIACL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 163.20
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 16 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 49
- **Target hits / Stop hits / Partials:** 16 / 49 / 28
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 16.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 15 | 37.5% | 5 | 25 | 10 | 0.12% | 5.0% |
| BUY @ 2nd Alert (retest1) | 40 | 15 | 37.5% | 5 | 25 | 10 | 0.12% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 29 | 54.7% | 11 | 24 | 18 | 0.22% | 11.5% |
| SELL @ 2nd Alert (retest1) | 53 | 29 | 54.7% | 11 | 24 | 18 | 0.22% | 11.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 93 | 44 | 47.3% | 16 | 49 | 28 | 0.18% | 16.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 10:20:00 | 181.00 | 177.85 | 0.00 | ORB-long ORB[176.90,179.55] vol=1.6x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-05-22 10:25:00 | 180.06 | 177.97 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 09:30:00 | 184.77 | 185.95 | 0.00 | ORB-short ORB[185.55,187.33] vol=1.8x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:45:00 | 183.86 | 185.43 | 0.00 | T1 1.5R @ 183.86 |
| Target hit | 2025-05-30 11:45:00 | 184.49 | 184.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:15:00 | 187.51 | 186.42 | 0.00 | ORB-long ORB[184.80,186.50] vol=3.6x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-06-02 11:30:00 | 186.87 | 186.44 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 10:20:00 | 187.19 | 188.17 | 0.00 | ORB-short ORB[187.72,189.49] vol=1.8x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:10:00 | 186.45 | 187.83 | 0.00 | T1 1.5R @ 186.45 |
| Target hit | 2025-06-03 15:20:00 | 184.25 | 186.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2025-06-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:00:00 | 183.78 | 184.24 | 0.00 | ORB-short ORB[184.32,185.97] vol=2.4x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-06-04 11:25:00 | 184.41 | 184.23 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:40:00 | 194.25 | 192.96 | 0.00 | ORB-long ORB[191.78,193.94] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-06-10 09:45:00 | 193.46 | 193.14 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:30:00 | 196.02 | 195.43 | 0.00 | ORB-long ORB[194.03,195.70] vol=3.3x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-06-11 09:40:00 | 195.48 | 195.51 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:45:00 | 197.36 | 197.88 | 0.00 | ORB-short ORB[197.50,199.83] vol=2.8x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 196.14 | 197.79 | 0.00 | T1 1.5R @ 196.14 |
| Target hit | 2025-06-12 15:20:00 | 191.25 | 194.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:35:00 | 183.99 | 182.72 | 0.00 | ORB-long ORB[181.06,183.33] vol=1.7x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:40:00 | 185.21 | 183.19 | 0.00 | T1 1.5R @ 185.21 |
| Stop hit — per-position SL triggered | 2025-06-19 09:50:00 | 183.99 | 183.24 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:45:00 | 180.63 | 178.63 | 0.00 | ORB-long ORB[177.05,179.54] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-06-20 09:50:00 | 179.66 | 178.76 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 10:55:00 | 184.00 | 185.30 | 0.00 | ORB-short ORB[185.43,187.03] vol=2.0x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 14:15:00 | 183.29 | 184.65 | 0.00 | T1 1.5R @ 183.29 |
| Stop hit — per-position SL triggered | 2025-06-27 15:00:00 | 184.00 | 184.42 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:25:00 | 186.98 | 185.47 | 0.00 | ORB-long ORB[184.15,185.52] vol=1.9x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-06-30 10:35:00 | 186.40 | 185.48 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:30:00 | 188.93 | 189.73 | 0.00 | ORB-short ORB[189.25,191.20] vol=2.3x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:40:00 | 187.97 | 189.29 | 0.00 | T1 1.5R @ 187.97 |
| Target hit | 2025-07-02 11:20:00 | 188.85 | 188.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 185.13 | 186.03 | 0.00 | ORB-short ORB[186.02,187.43] vol=4.9x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 09:35:00 | 184.13 | 185.60 | 0.00 | T1 1.5R @ 184.13 |
| Stop hit — per-position SL triggered | 2025-07-04 10:50:00 | 185.13 | 185.35 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:15:00 | 182.76 | 183.59 | 0.00 | ORB-short ORB[183.71,184.90] vol=2.5x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-07-22 10:20:00 | 183.16 | 183.53 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:05:00 | 184.24 | 185.20 | 0.00 | ORB-short ORB[185.00,185.90] vol=1.7x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:30:00 | 183.80 | 185.10 | 0.00 | T1 1.5R @ 183.80 |
| Stop hit — per-position SL triggered | 2025-07-24 12:15:00 | 184.24 | 184.90 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:50:00 | 177.11 | 175.22 | 0.00 | ORB-long ORB[173.04,175.42] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-07-29 09:55:00 | 176.33 | 175.29 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 09:45:00 | 194.77 | 195.39 | 0.00 | ORB-short ORB[194.80,197.65] vol=1.6x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 10:05:00 | 193.89 | 195.24 | 0.00 | T1 1.5R @ 193.89 |
| Target hit | 2025-08-25 15:20:00 | 193.39 | 194.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 189.25 | 191.25 | 0.00 | ORB-short ORB[190.80,193.15] vol=2.3x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-08-26 09:40:00 | 189.87 | 191.12 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:40:00 | 192.29 | 192.89 | 0.00 | ORB-short ORB[192.50,193.83] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-09-18 10:45:00 | 192.81 | 192.72 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:50:00 | 196.83 | 197.77 | 0.00 | ORB-short ORB[197.47,200.00] vol=1.7x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-09-23 10:05:00 | 197.61 | 197.67 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-10-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:05:00 | 189.32 | 189.86 | 0.00 | ORB-short ORB[189.55,191.00] vol=1.7x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 11:25:00 | 188.66 | 189.78 | 0.00 | T1 1.5R @ 188.66 |
| Stop hit — per-position SL triggered | 2025-10-06 12:10:00 | 189.32 | 190.12 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-10-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:45:00 | 190.04 | 190.99 | 0.00 | ORB-short ORB[190.75,193.06] vol=2.5x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-10-08 15:05:00 | 190.54 | 190.40 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 189.00 | 188.68 | 0.00 | ORB-long ORB[188.08,188.79] vol=3.0x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-10-17 09:45:00 | 188.57 | 188.80 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:00:00 | 187.77 | 188.06 | 0.00 | ORB-short ORB[188.00,188.78] vol=2.4x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:25:00 | 187.15 | 187.97 | 0.00 | T1 1.5R @ 187.15 |
| Target hit | 2025-10-24 15:20:00 | 187.17 | 187.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:30:00 | 189.70 | 188.80 | 0.00 | ORB-long ORB[187.16,189.61] vol=2.4x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-10-27 09:35:00 | 189.19 | 188.87 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:30:00 | 188.59 | 188.97 | 0.00 | ORB-short ORB[189.01,190.00] vol=2.6x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-10-31 10:45:00 | 188.91 | 188.91 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:35:00 | 179.64 | 180.30 | 0.00 | ORB-short ORB[180.10,181.49] vol=1.6x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:55:00 | 178.85 | 179.92 | 0.00 | T1 1.5R @ 178.85 |
| Stop hit — per-position SL triggered | 2025-11-07 10:20:00 | 179.64 | 179.73 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-11-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:45:00 | 181.41 | 182.10 | 0.00 | ORB-short ORB[182.70,184.30] vol=2.2x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-11-11 11:55:00 | 181.81 | 181.99 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:45:00 | 184.89 | 184.08 | 0.00 | ORB-long ORB[182.76,184.50] vol=4.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-11-12 09:50:00 | 184.39 | 184.11 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-11-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 10:50:00 | 183.40 | 182.05 | 0.00 | ORB-long ORB[180.31,181.54] vol=8.3x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-11-18 11:00:00 | 182.82 | 182.58 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-11-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:20:00 | 174.90 | 174.40 | 0.00 | ORB-long ORB[172.31,174.00] vol=1.6x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-11-26 10:35:00 | 174.30 | 174.41 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:15:00 | 171.44 | 172.55 | 0.00 | ORB-short ORB[173.15,174.46] vol=3.9x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 13:20:00 | 170.87 | 172.21 | 0.00 | T1 1.5R @ 170.87 |
| Target hit | 2025-11-27 15:20:00 | 170.76 | 171.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-12-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:30:00 | 161.47 | 162.38 | 0.00 | ORB-short ORB[162.38,163.96] vol=2.9x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-12-05 10:45:00 | 161.91 | 162.29 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:15:00 | 158.44 | 156.18 | 0.00 | ORB-long ORB[155.15,157.38] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 12:05:00 | 159.21 | 156.71 | 0.00 | T1 1.5R @ 159.21 |
| Stop hit — per-position SL triggered | 2025-12-09 12:40:00 | 158.44 | 156.94 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 167.80 | 169.01 | 0.00 | ORB-short ORB[168.26,170.27] vol=2.2x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-12-16 12:25:00 | 168.38 | 168.84 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 11:05:00 | 159.85 | 160.51 | 0.00 | ORB-short ORB[160.24,162.21] vol=1.6x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 13:30:00 | 159.21 | 160.30 | 0.00 | T1 1.5R @ 159.21 |
| Target hit | 2025-12-18 15:20:00 | 158.69 | 159.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-12-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:05:00 | 159.09 | 159.93 | 0.00 | ORB-short ORB[159.30,160.48] vol=1.6x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:35:00 | 158.23 | 159.62 | 0.00 | T1 1.5R @ 158.23 |
| Stop hit — per-position SL triggered | 2025-12-19 12:05:00 | 159.09 | 159.38 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:00:00 | 153.56 | 154.57 | 0.00 | ORB-short ORB[154.59,156.58] vol=2.3x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-12-22 11:35:00 | 154.08 | 154.49 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-12-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:05:00 | 157.40 | 155.64 | 0.00 | ORB-long ORB[154.31,156.50] vol=2.1x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-12-31 10:10:00 | 156.65 | 155.74 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-01-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:25:00 | 156.46 | 155.73 | 0.00 | ORB-long ORB[155.11,156.25] vol=2.6x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-01-02 10:35:00 | 156.10 | 155.76 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-01-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:55:00 | 155.00 | 154.12 | 0.00 | ORB-long ORB[152.91,154.67] vol=4.1x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-01-06 10:05:00 | 154.50 | 154.18 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-01-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:55:00 | 152.99 | 153.42 | 0.00 | ORB-short ORB[153.10,154.22] vol=1.8x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 13:25:00 | 152.56 | 153.22 | 0.00 | T1 1.5R @ 152.56 |
| Stop hit — per-position SL triggered | 2026-01-07 14:35:00 | 152.99 | 153.20 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:55:00 | 151.84 | 153.11 | 0.00 | ORB-short ORB[153.25,154.90] vol=2.7x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 151.31 | 152.88 | 0.00 | T1 1.5R @ 151.31 |
| Target hit | 2026-01-08 15:20:00 | 150.82 | 151.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2026-01-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 09:40:00 | 149.17 | 149.72 | 0.00 | ORB-short ORB[149.33,150.68] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-01-16 09:50:00 | 149.60 | 149.70 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-01-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:40:00 | 147.67 | 148.39 | 0.00 | ORB-short ORB[148.50,150.00] vol=2.7x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-01-19 09:50:00 | 148.14 | 148.35 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2026-01-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:00:00 | 146.30 | 147.37 | 0.00 | ORB-short ORB[147.68,149.00] vol=2.3x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 146.75 | 147.33 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-01-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:00:00 | 147.26 | 146.01 | 0.00 | ORB-long ORB[144.00,146.19] vol=2.2x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 10:15:00 | 148.34 | 146.36 | 0.00 | T1 1.5R @ 148.34 |
| Target hit | 2026-01-30 13:10:00 | 148.94 | 148.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2026-02-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 10:10:00 | 148.79 | 147.42 | 0.00 | ORB-long ORB[145.92,147.70] vol=1.9x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-02-04 10:25:00 | 148.29 | 147.63 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:35:00 | 147.86 | 148.31 | 0.00 | ORB-short ORB[148.00,149.34] vol=3.6x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 10:55:00 | 147.20 | 148.05 | 0.00 | T1 1.5R @ 147.20 |
| Target hit | 2026-02-05 14:55:00 | 147.70 | 147.60 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — BUY (started 2026-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:30:00 | 149.14 | 148.39 | 0.00 | ORB-long ORB[147.45,148.77] vol=3.1x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 09:35:00 | 149.96 | 150.48 | 0.00 | T1 1.5R @ 149.96 |
| Target hit | 2026-02-06 09:50:00 | 150.61 | 150.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 151.94 | 150.50 | 0.00 | ORB-long ORB[149.51,151.26] vol=3.5x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 13:15:00 | 152.73 | 151.21 | 0.00 | T1 1.5R @ 152.73 |
| Target hit | 2026-02-09 15:20:00 | 152.82 | 152.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2026-02-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:25:00 | 154.96 | 153.70 | 0.00 | ORB-long ORB[152.47,154.70] vol=2.6x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:30:00 | 155.88 | 154.04 | 0.00 | T1 1.5R @ 155.88 |
| Target hit | 2026-02-10 15:20:00 | 159.16 | 158.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 153.88 | 154.43 | 0.00 | ORB-short ORB[154.01,155.96] vol=1.9x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-02-12 09:50:00 | 154.32 | 154.30 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 152.00 | 151.69 | 0.00 | ORB-long ORB[150.88,151.85] vol=4.9x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:45:00 | 152.70 | 152.18 | 0.00 | T1 1.5R @ 152.70 |
| Stop hit — per-position SL triggered | 2026-02-17 09:55:00 | 152.00 | 152.30 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-02-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:50:00 | 151.12 | 150.49 | 0.00 | ORB-long ORB[149.00,151.04] vol=2.1x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-02-20 10:20:00 | 150.56 | 150.67 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 151.97 | 150.62 | 0.00 | ORB-long ORB[149.38,150.27] vol=3.4x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-02-26 09:45:00 | 151.41 | 150.98 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 147.30 | 147.87 | 0.00 | ORB-short ORB[147.77,149.04] vol=1.7x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-02-27 10:00:00 | 147.77 | 147.85 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-03-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 11:05:00 | 138.89 | 139.40 | 0.00 | ORB-short ORB[139.00,140.50] vol=2.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:30:00 | 138.05 | 139.29 | 0.00 | T1 1.5R @ 138.05 |
| Target hit | 2026-03-04 15:20:00 | 137.45 | 138.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2026-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:40:00 | 139.45 | 138.94 | 0.00 | ORB-long ORB[138.00,139.26] vol=1.5x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 09:50:00 | 140.19 | 139.14 | 0.00 | T1 1.5R @ 140.19 |
| Target hit | 2026-03-06 10:30:00 | 140.44 | 140.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — BUY (started 2026-03-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:55:00 | 137.69 | 136.19 | 0.00 | ORB-long ORB[135.44,136.79] vol=1.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-03-10 11:05:00 | 137.19 | 136.22 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 166.43 | 164.35 | 0.00 | ORB-long ORB[162.42,164.19] vol=7.1x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:05:00 | 167.72 | 165.19 | 0.00 | T1 1.5R @ 167.72 |
| Stop hit — per-position SL triggered | 2026-04-23 10:10:00 | 166.43 | 165.33 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 164.13 | 165.14 | 0.00 | ORB-short ORB[164.51,166.80] vol=1.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-04-28 09:35:00 | 164.75 | 165.07 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 162.79 | 161.77 | 0.00 | ORB-long ORB[160.80,162.40] vol=1.9x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:35:00 | 163.86 | 162.21 | 0.00 | T1 1.5R @ 163.86 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 162.79 | 162.93 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 163.67 | 163.15 | 0.00 | ORB-long ORB[161.96,163.56] vol=3.1x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 163.03 | 163.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-22 10:20:00 | 181.00 | 2025-05-22 10:25:00 | 180.06 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-05-30 09:30:00 | 184.77 | 2025-05-30 09:45:00 | 183.86 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-05-30 09:30:00 | 184.77 | 2025-05-30 11:45:00 | 184.49 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-06-02 11:15:00 | 187.51 | 2025-06-02 11:30:00 | 186.87 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-03 10:20:00 | 187.19 | 2025-06-03 11:10:00 | 186.45 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-03 10:20:00 | 187.19 | 2025-06-03 15:20:00 | 184.25 | TARGET_HIT | 0.50 | 1.57% |
| SELL | retest1 | 2025-06-04 11:00:00 | 183.78 | 2025-06-04 11:25:00 | 184.41 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-10 09:40:00 | 194.25 | 2025-06-10 09:45:00 | 193.46 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-06-11 09:30:00 | 196.02 | 2025-06-11 09:40:00 | 195.48 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-06-12 10:45:00 | 197.36 | 2025-06-12 11:15:00 | 196.14 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-06-12 10:45:00 | 197.36 | 2025-06-12 15:20:00 | 191.25 | TARGET_HIT | 0.50 | 3.10% |
| BUY | retest1 | 2025-06-19 09:35:00 | 183.99 | 2025-06-19 09:40:00 | 185.21 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-06-19 09:35:00 | 183.99 | 2025-06-19 09:50:00 | 183.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-20 09:45:00 | 180.63 | 2025-06-20 09:50:00 | 179.66 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2025-06-27 10:55:00 | 184.00 | 2025-06-27 14:15:00 | 183.29 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-27 10:55:00 | 184.00 | 2025-06-27 15:00:00 | 184.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-30 10:25:00 | 186.98 | 2025-06-30 10:35:00 | 186.40 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-02 09:30:00 | 188.93 | 2025-07-02 09:40:00 | 187.97 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-07-02 09:30:00 | 188.93 | 2025-07-02 11:20:00 | 188.85 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2025-07-04 09:30:00 | 185.13 | 2025-07-04 09:35:00 | 184.13 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-07-04 09:30:00 | 185.13 | 2025-07-04 10:50:00 | 185.13 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 10:15:00 | 182.76 | 2025-07-22 10:20:00 | 183.16 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-24 11:05:00 | 184.24 | 2025-07-24 11:30:00 | 183.80 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-07-24 11:05:00 | 184.24 | 2025-07-24 12:15:00 | 184.24 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-29 09:50:00 | 177.11 | 2025-07-29 09:55:00 | 176.33 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-08-25 09:45:00 | 194.77 | 2025-08-25 10:05:00 | 193.89 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-08-25 09:45:00 | 194.77 | 2025-08-25 15:20:00 | 193.39 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2025-08-26 09:35:00 | 189.25 | 2025-08-26 09:40:00 | 189.87 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-18 09:40:00 | 192.29 | 2025-09-18 10:45:00 | 192.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-23 09:50:00 | 196.83 | 2025-09-23 10:05:00 | 197.61 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-06 11:05:00 | 189.32 | 2025-10-06 11:25:00 | 188.66 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-10-06 11:05:00 | 189.32 | 2025-10-06 12:10:00 | 189.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 10:45:00 | 190.04 | 2025-10-08 15:05:00 | 190.54 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-17 09:30:00 | 189.00 | 2025-10-17 09:45:00 | 188.57 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-24 10:00:00 | 187.77 | 2025-10-24 10:25:00 | 187.15 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-24 10:00:00 | 187.77 | 2025-10-24 15:20:00 | 187.17 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-27 09:30:00 | 189.70 | 2025-10-27 09:35:00 | 189.19 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-31 10:30:00 | 188.59 | 2025-10-31 10:45:00 | 188.91 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-07 09:35:00 | 179.64 | 2025-11-07 09:55:00 | 178.85 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-11-07 09:35:00 | 179.64 | 2025-11-07 10:20:00 | 179.64 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:45:00 | 181.41 | 2025-11-11 11:55:00 | 181.81 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-12 09:45:00 | 184.89 | 2025-11-12 09:50:00 | 184.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-18 10:50:00 | 183.40 | 2025-11-18 11:00:00 | 182.82 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-26 10:20:00 | 174.90 | 2025-11-26 10:35:00 | 174.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-27 11:15:00 | 171.44 | 2025-11-27 13:20:00 | 170.87 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-27 11:15:00 | 171.44 | 2025-11-27 15:20:00 | 170.76 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-05 10:30:00 | 161.47 | 2025-12-05 10:45:00 | 161.91 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-09 11:15:00 | 158.44 | 2025-12-09 12:05:00 | 159.21 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-12-09 11:15:00 | 158.44 | 2025-12-09 12:40:00 | 158.44 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 11:00:00 | 167.80 | 2025-12-16 12:25:00 | 168.38 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-18 11:05:00 | 159.85 | 2025-12-18 13:30:00 | 159.21 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-18 11:05:00 | 159.85 | 2025-12-18 15:20:00 | 158.69 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2025-12-19 10:05:00 | 159.09 | 2025-12-19 10:35:00 | 158.23 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-12-19 10:05:00 | 159.09 | 2025-12-19 12:05:00 | 159.09 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-22 11:00:00 | 153.56 | 2025-12-22 11:35:00 | 154.08 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-31 10:05:00 | 157.40 | 2025-12-31 10:10:00 | 156.65 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-01-02 10:25:00 | 156.46 | 2026-01-02 10:35:00 | 156.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-06 09:55:00 | 155.00 | 2026-01-06 10:05:00 | 154.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-07 10:55:00 | 152.99 | 2026-01-07 13:25:00 | 152.56 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-01-07 10:55:00 | 152.99 | 2026-01-07 14:35:00 | 152.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 10:55:00 | 151.84 | 2026-01-08 11:20:00 | 151.31 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-08 10:55:00 | 151.84 | 2026-01-08 15:20:00 | 150.82 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-01-16 09:40:00 | 149.17 | 2026-01-16 09:50:00 | 149.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-19 09:40:00 | 147.67 | 2026-01-19 09:50:00 | 148.14 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-29 11:00:00 | 146.30 | 2026-01-29 11:15:00 | 146.75 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-30 10:00:00 | 147.26 | 2026-01-30 10:15:00 | 148.34 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-01-30 10:00:00 | 147.26 | 2026-01-30 13:10:00 | 148.94 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2026-02-04 10:10:00 | 148.79 | 2026-02-04 10:25:00 | 148.29 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-05 09:35:00 | 147.86 | 2026-02-05 10:55:00 | 147.20 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-05 09:35:00 | 147.86 | 2026-02-05 14:55:00 | 147.70 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2026-02-06 09:30:00 | 149.14 | 2026-02-06 09:35:00 | 149.96 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-06 09:30:00 | 149.14 | 2026-02-06 09:50:00 | 150.61 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2026-02-09 11:05:00 | 151.94 | 2026-02-09 13:15:00 | 152.73 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-09 11:05:00 | 151.94 | 2026-02-09 15:20:00 | 152.82 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-10 10:25:00 | 154.96 | 2026-02-10 10:30:00 | 155.88 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-10 10:25:00 | 154.96 | 2026-02-10 15:20:00 | 159.16 | TARGET_HIT | 0.50 | 2.71% |
| SELL | retest1 | 2026-02-12 09:35:00 | 153.88 | 2026-02-12 09:50:00 | 154.32 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 09:30:00 | 152.00 | 2026-02-17 09:45:00 | 152.70 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-17 09:30:00 | 152.00 | 2026-02-17 09:55:00 | 152.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:50:00 | 151.12 | 2026-02-20 10:20:00 | 150.56 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-26 09:35:00 | 151.97 | 2026-02-26 09:45:00 | 151.41 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-27 09:55:00 | 147.30 | 2026-02-27 10:00:00 | 147.77 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-04 11:05:00 | 138.89 | 2026-03-04 11:30:00 | 138.05 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-03-04 11:05:00 | 138.89 | 2026-03-04 15:20:00 | 137.45 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2026-03-06 09:40:00 | 139.45 | 2026-03-06 09:50:00 | 140.19 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-06 09:40:00 | 139.45 | 2026-03-06 10:30:00 | 140.44 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-10 10:55:00 | 137.69 | 2026-03-10 11:05:00 | 137.19 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-23 10:00:00 | 166.43 | 2026-04-23 10:05:00 | 167.72 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-04-23 10:00:00 | 166.43 | 2026-04-23 10:10:00 | 166.43 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:30:00 | 164.13 | 2026-04-28 09:35:00 | 164.75 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-05-05 09:30:00 | 162.79 | 2026-05-05 09:35:00 | 163.86 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-05-05 09:30:00 | 162.79 | 2026-05-05 10:10:00 | 162.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:30:00 | 163.67 | 2026-05-06 09:35:00 | 163.03 | STOP_HIT | 1.00 | -0.39% |
