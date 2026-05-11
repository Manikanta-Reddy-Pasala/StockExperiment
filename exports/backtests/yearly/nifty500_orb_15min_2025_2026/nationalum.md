# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 401.75
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
| ENTRY1 | 68 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 9 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 59
- **Target hits / Stop hits / Partials:** 9 / 59 / 21
- **Avg / median % per leg:** 0.08% / -0.20%
- **Sum % (uncompounded):** 7.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 18 | 32.7% | 6 | 37 | 12 | 0.09% | 5.1% |
| BUY @ 2nd Alert (retest1) | 55 | 18 | 32.7% | 6 | 37 | 12 | 0.09% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 34 | 12 | 35.3% | 3 | 22 | 9 | 0.07% | 2.5% |
| SELL @ 2nd Alert (retest1) | 34 | 12 | 35.3% | 3 | 22 | 9 | 0.07% | 2.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 89 | 30 | 33.7% | 9 | 59 | 21 | 0.08% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 174.52 | 173.45 | 0.00 | ORB-long ORB[172.19,174.50] vol=1.6x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 173.83 | 173.46 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:30:00 | 183.02 | 182.28 | 0.00 | ORB-long ORB[181.29,182.71] vol=1.5x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:40:00 | 183.62 | 182.75 | 0.00 | T1 1.5R @ 183.62 |
| Stop hit — per-position SL triggered | 2025-05-28 09:50:00 | 183.02 | 182.82 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:55:00 | 184.20 | 182.89 | 0.00 | ORB-long ORB[180.94,183.50] vol=2.0x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-06-03 11:10:00 | 183.66 | 183.00 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:40:00 | 181.22 | 181.85 | 0.00 | ORB-short ORB[181.24,182.78] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:50:00 | 180.46 | 181.45 | 0.00 | T1 1.5R @ 180.46 |
| Stop hit — per-position SL triggered | 2025-06-04 09:55:00 | 181.22 | 181.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:50:00 | 181.65 | 182.75 | 0.00 | ORB-short ORB[181.96,183.25] vol=1.7x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-06-05 11:00:00 | 182.14 | 182.72 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 11:15:00 | 188.57 | 189.50 | 0.00 | ORB-short ORB[188.62,191.11] vol=1.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-06-09 11:55:00 | 189.05 | 189.43 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:40:00 | 188.64 | 189.90 | 0.00 | ORB-short ORB[189.52,192.00] vol=2.2x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-06-12 09:50:00 | 189.25 | 189.73 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:45:00 | 183.18 | 184.34 | 0.00 | ORB-short ORB[185.15,186.70] vol=1.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-06-19 10:55:00 | 183.75 | 184.29 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:40:00 | 183.68 | 182.31 | 0.00 | ORB-long ORB[180.97,183.00] vol=1.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-06-20 09:50:00 | 183.06 | 182.53 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:25:00 | 191.62 | 190.31 | 0.00 | ORB-long ORB[189.00,190.71] vol=2.1x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-06-24 11:55:00 | 191.06 | 190.88 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 10:45:00 | 189.62 | 188.78 | 0.00 | ORB-long ORB[188.03,189.57] vol=1.9x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-06-26 10:55:00 | 189.05 | 188.83 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:35:00 | 195.64 | 194.35 | 0.00 | ORB-long ORB[193.00,194.70] vol=2.9x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-07-03 09:40:00 | 195.01 | 194.45 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:35:00 | 193.49 | 192.78 | 0.00 | ORB-long ORB[191.80,192.78] vol=2.3x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-07-04 11:05:00 | 193.01 | 192.87 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:10:00 | 188.35 | 189.05 | 0.00 | ORB-short ORB[188.50,190.50] vol=1.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-07-16 11:35:00 | 188.72 | 188.99 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:10:00 | 188.11 | 189.25 | 0.00 | ORB-short ORB[189.12,190.36] vol=1.7x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 187.35 | 188.96 | 0.00 | T1 1.5R @ 187.35 |
| Stop hit — per-position SL triggered | 2025-07-18 15:10:00 | 188.11 | 188.01 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 198.35 | 197.62 | 0.00 | ORB-long ORB[196.53,197.86] vol=2.8x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-07-23 09:35:00 | 197.90 | 197.82 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:35:00 | 187.90 | 188.50 | 0.00 | ORB-short ORB[188.05,189.25] vol=1.9x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 188.48 | 188.30 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:00:00 | 186.83 | 187.98 | 0.00 | ORB-short ORB[187.21,189.21] vol=1.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 12:00:00 | 186.15 | 187.69 | 0.00 | T1 1.5R @ 186.15 |
| Stop hit — per-position SL triggered | 2025-08-07 12:10:00 | 186.83 | 187.68 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:15:00 | 186.83 | 187.92 | 0.00 | ORB-short ORB[187.58,189.65] vol=1.9x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:40:00 | 186.01 | 187.77 | 0.00 | T1 1.5R @ 186.01 |
| Stop hit — per-position SL triggered | 2025-08-11 13:00:00 | 186.83 | 187.61 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:00:00 | 186.81 | 188.06 | 0.00 | ORB-short ORB[187.07,189.69] vol=1.7x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 187.29 | 187.99 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:00:00 | 186.83 | 187.85 | 0.00 | ORB-short ORB[187.26,190.01] vol=1.6x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-08-14 10:05:00 | 187.36 | 187.83 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:15:00 | 189.90 | 189.03 | 0.00 | ORB-long ORB[188.28,189.50] vol=2.4x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 10:25:00 | 190.55 | 189.33 | 0.00 | T1 1.5R @ 190.55 |
| Target hit | 2025-08-19 15:20:00 | 191.46 | 190.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:55:00 | 192.10 | 190.87 | 0.00 | ORB-long ORB[190.28,191.60] vol=3.5x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-08-20 11:35:00 | 191.75 | 191.29 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 11:15:00 | 191.84 | 192.33 | 0.00 | ORB-short ORB[192.06,193.57] vol=1.7x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-08-21 11:20:00 | 192.14 | 192.32 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 11:00:00 | 187.45 | 185.73 | 0.00 | ORB-long ORB[185.00,186.99] vol=1.6x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-08-29 13:55:00 | 186.76 | 186.28 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:45:00 | 193.25 | 192.23 | 0.00 | ORB-long ORB[190.87,193.18] vol=2.2x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:15:00 | 194.09 | 193.10 | 0.00 | T1 1.5R @ 194.09 |
| Target hit | 2025-09-02 15:20:00 | 201.19 | 198.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:35:00 | 204.83 | 203.32 | 0.00 | ORB-long ORB[201.51,203.90] vol=1.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-09-03 09:45:00 | 204.10 | 203.71 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 214.85 | 213.03 | 0.00 | ORB-long ORB[211.79,213.75] vol=1.7x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 214.01 | 213.71 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:05:00 | 210.96 | 208.73 | 0.00 | ORB-long ORB[207.25,209.28] vol=2.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-09-10 10:20:00 | 210.26 | 209.02 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:35:00 | 212.10 | 211.22 | 0.00 | ORB-long ORB[209.80,212.00] vol=1.5x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 09:50:00 | 212.87 | 211.59 | 0.00 | T1 1.5R @ 212.87 |
| Target hit | 2025-09-11 12:20:00 | 214.90 | 215.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2025-09-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:35:00 | 220.25 | 218.92 | 0.00 | ORB-long ORB[218.50,219.90] vol=1.6x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-09-15 11:05:00 | 219.54 | 219.10 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 09:40:00 | 215.64 | 217.00 | 0.00 | ORB-short ORB[216.40,217.90] vol=2.3x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-09-16 09:55:00 | 216.29 | 216.75 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:40:00 | 214.87 | 216.12 | 0.00 | ORB-short ORB[216.42,217.92] vol=2.4x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-09-17 11:00:00 | 215.42 | 215.91 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:45:00 | 208.20 | 210.12 | 0.00 | ORB-short ORB[210.03,211.88] vol=1.9x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-09-23 10:05:00 | 208.91 | 209.55 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:55:00 | 209.93 | 208.23 | 0.00 | ORB-long ORB[206.26,208.77] vol=2.3x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-09-25 10:10:00 | 209.25 | 208.87 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:40:00 | 203.94 | 204.48 | 0.00 | ORB-short ORB[204.01,206.22] vol=3.0x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:05:00 | 202.94 | 204.23 | 0.00 | T1 1.5R @ 202.94 |
| Target hit | 2025-09-26 15:20:00 | 200.43 | 201.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-10-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:45:00 | 217.96 | 216.28 | 0.00 | ORB-long ORB[214.50,217.00] vol=2.1x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 09:50:00 | 219.21 | 217.54 | 0.00 | T1 1.5R @ 219.21 |
| Target hit | 2025-10-03 14:30:00 | 220.90 | 221.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 219.70 | 218.07 | 0.00 | ORB-long ORB[216.55,218.58] vol=2.5x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-10-07 09:35:00 | 218.96 | 218.20 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 225.84 | 227.71 | 0.00 | ORB-short ORB[227.10,229.20] vol=2.4x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-10-14 11:50:00 | 226.44 | 227.52 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:00:00 | 227.16 | 225.67 | 0.00 | ORB-long ORB[223.80,225.79] vol=2.3x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 226.46 | 226.48 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 230.12 | 228.92 | 0.00 | ORB-long ORB[227.06,229.78] vol=2.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-10-17 09:45:00 | 229.41 | 229.20 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 229.78 | 228.71 | 0.00 | ORB-long ORB[227.50,229.63] vol=2.0x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-10-23 09:35:00 | 229.08 | 228.72 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:45:00 | 240.23 | 238.18 | 0.00 | ORB-long ORB[236.29,238.48] vol=3.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-10-28 09:50:00 | 239.42 | 238.51 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:30:00 | 240.50 | 239.67 | 0.00 | ORB-long ORB[238.00,240.32] vol=2.6x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-10-29 09:50:00 | 239.79 | 240.01 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:50:00 | 240.05 | 238.00 | 0.00 | ORB-long ORB[236.28,239.20] vol=2.1x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-11-03 10:25:00 | 238.95 | 238.44 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-11-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:35:00 | 234.80 | 236.61 | 0.00 | ORB-short ORB[235.71,238.14] vol=1.9x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:50:00 | 233.82 | 236.18 | 0.00 | T1 1.5R @ 233.82 |
| Target hit | 2025-11-04 15:20:00 | 233.30 | 233.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2025-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:45:00 | 234.00 | 230.69 | 0.00 | ORB-long ORB[228.00,230.40] vol=1.6x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:10:00 | 235.17 | 231.34 | 0.00 | T1 1.5R @ 235.17 |
| Stop hit — per-position SL triggered | 2025-11-07 12:10:00 | 234.00 | 233.29 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:30:00 | 270.32 | 268.75 | 0.00 | ORB-long ORB[266.11,269.13] vol=2.7x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:45:00 | 271.48 | 269.60 | 0.00 | T1 1.5R @ 271.48 |
| Stop hit — per-position SL triggered | 2025-11-13 10:00:00 | 270.32 | 269.92 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 11:15:00 | 259.54 | 262.23 | 0.00 | ORB-short ORB[260.54,263.28] vol=2.1x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-11-17 11:30:00 | 260.18 | 262.00 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 09:35:00 | 259.63 | 258.28 | 0.00 | ORB-long ORB[255.64,259.40] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-11-19 09:50:00 | 258.81 | 258.50 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 09:30:00 | 260.90 | 259.85 | 0.00 | ORB-long ORB[258.11,260.70] vol=2.1x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 09:35:00 | 261.98 | 260.33 | 0.00 | T1 1.5R @ 261.98 |
| Stop hit — per-position SL triggered | 2025-11-20 09:45:00 | 260.90 | 260.66 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 10:50:00 | 254.40 | 254.36 | 0.00 | ORB-long ORB[251.40,253.98] vol=2.3x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-11-25 10:55:00 | 253.73 | 254.23 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 264.12 | 262.59 | 0.00 | ORB-long ORB[260.06,263.70] vol=1.9x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-11-28 09:35:00 | 263.35 | 262.64 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 266.35 | 265.05 | 0.00 | ORB-long ORB[263.45,266.10] vol=1.9x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-12-10 10:10:00 | 265.35 | 265.88 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 09:45:00 | 268.20 | 266.14 | 0.00 | ORB-long ORB[263.70,267.05] vol=1.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-12-11 09:50:00 | 267.19 | 266.24 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:35:00 | 270.25 | 269.17 | 0.00 | ORB-long ORB[266.35,269.90] vol=1.5x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 09:55:00 | 271.48 | 269.72 | 0.00 | T1 1.5R @ 271.48 |
| Stop hit — per-position SL triggered | 2025-12-12 10:20:00 | 270.25 | 270.13 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:35:00 | 280.50 | 279.41 | 0.00 | ORB-long ORB[277.15,280.40] vol=1.8x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-12-17 10:05:00 | 279.47 | 279.97 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:25:00 | 279.50 | 277.29 | 0.00 | ORB-long ORB[275.60,279.45] vol=1.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 11:50:00 | 280.83 | 278.43 | 0.00 | T1 1.5R @ 280.83 |
| Stop hit — per-position SL triggered | 2025-12-18 14:10:00 | 279.50 | 279.70 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:10:00 | 277.45 | 279.72 | 0.00 | ORB-short ORB[278.50,280.75] vol=1.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-12-19 12:35:00 | 278.22 | 279.30 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:35:00 | 299.90 | 298.66 | 0.00 | ORB-long ORB[296.50,299.75] vol=1.9x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:55:00 | 301.29 | 299.23 | 0.00 | T1 1.5R @ 301.29 |
| Target hit | 2025-12-26 15:20:00 | 307.20 | 303.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2026-01-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 09:45:00 | 358.60 | 355.05 | 0.00 | ORB-long ORB[351.15,355.50] vol=2.3x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-01-13 09:55:00 | 357.02 | 355.67 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 364.30 | 367.10 | 0.00 | ORB-short ORB[365.85,370.80] vol=2.0x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:35:00 | 362.03 | 366.22 | 0.00 | T1 1.5R @ 362.03 |
| Target hit | 2026-01-20 10:40:00 | 363.25 | 363.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — SELL (started 2026-01-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 09:55:00 | 360.15 | 364.93 | 0.00 | ORB-short ORB[361.65,366.60] vol=1.9x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 10:00:00 | 357.55 | 363.65 | 0.00 | T1 1.5R @ 357.55 |
| Stop hit — per-position SL triggered | 2026-01-22 10:05:00 | 360.15 | 363.43 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:15:00 | 351.15 | 353.79 | 0.00 | ORB-short ORB[352.50,357.50] vol=1.9x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-02-06 11:00:00 | 352.75 | 352.80 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 370.35 | 368.77 | 0.00 | ORB-long ORB[366.50,369.65] vol=1.6x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:50:00 | 372.00 | 369.68 | 0.00 | T1 1.5R @ 372.00 |
| Target hit | 2026-02-12 10:15:00 | 370.70 | 371.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 342.45 | 345.96 | 0.00 | ORB-short ORB[346.40,351.45] vol=1.6x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:10:00 | 340.93 | 345.00 | 0.00 | T1 1.5R @ 340.93 |
| Stop hit — per-position SL triggered | 2026-02-19 12:20:00 | 342.45 | 344.85 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 429.70 | 427.23 | 0.00 | ORB-long ORB[424.10,428.75] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-04-21 09:35:00 | 428.52 | 427.35 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-05-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:55:00 | 416.75 | 410.88 | 0.00 | ORB-long ORB[407.65,412.70] vol=2.9x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 414.66 | 411.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:30:00 | 174.52 | 2025-05-15 09:35:00 | 173.83 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-05-28 09:30:00 | 183.02 | 2025-05-28 09:40:00 | 183.62 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-05-28 09:30:00 | 183.02 | 2025-05-28 09:50:00 | 183.02 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-03 10:55:00 | 184.20 | 2025-06-03 11:10:00 | 183.66 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-04 09:40:00 | 181.22 | 2025-06-04 09:50:00 | 180.46 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-06-04 09:40:00 | 181.22 | 2025-06-04 09:55:00 | 181.22 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-05 10:50:00 | 181.65 | 2025-06-05 11:00:00 | 182.14 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-06-09 11:15:00 | 188.57 | 2025-06-09 11:55:00 | 189.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-12 09:40:00 | 188.64 | 2025-06-12 09:50:00 | 189.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-06-19 10:45:00 | 183.18 | 2025-06-19 10:55:00 | 183.75 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-20 09:40:00 | 183.68 | 2025-06-20 09:50:00 | 183.06 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-24 10:25:00 | 191.62 | 2025-06-24 11:55:00 | 191.06 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-26 10:45:00 | 189.62 | 2025-06-26 10:55:00 | 189.05 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-03 09:35:00 | 195.64 | 2025-07-03 09:40:00 | 195.01 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-04 10:35:00 | 193.49 | 2025-07-04 11:05:00 | 193.01 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-16 11:10:00 | 188.35 | 2025-07-16 11:35:00 | 188.72 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-18 10:10:00 | 188.11 | 2025-07-18 10:15:00 | 187.35 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-18 10:10:00 | 188.11 | 2025-07-18 15:10:00 | 188.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-23 09:30:00 | 198.35 | 2025-07-23 09:35:00 | 197.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-06 09:35:00 | 187.90 | 2025-08-06 10:15:00 | 188.48 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-07 11:00:00 | 186.83 | 2025-08-07 12:00:00 | 186.15 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-07 11:00:00 | 186.83 | 2025-08-07 12:10:00 | 186.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-11 11:15:00 | 186.83 | 2025-08-11 11:40:00 | 186.01 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-08-11 11:15:00 | 186.83 | 2025-08-11 13:00:00 | 186.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-12 11:00:00 | 186.81 | 2025-08-12 11:15:00 | 187.29 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-14 10:00:00 | 186.83 | 2025-08-14 10:05:00 | 187.36 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-19 10:15:00 | 189.90 | 2025-08-19 10:25:00 | 190.55 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-08-19 10:15:00 | 189.90 | 2025-08-19 15:20:00 | 191.46 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2025-08-20 10:55:00 | 192.10 | 2025-08-20 11:35:00 | 191.75 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-21 11:15:00 | 191.84 | 2025-08-21 11:20:00 | 192.14 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-08-29 11:00:00 | 187.45 | 2025-08-29 13:55:00 | 186.76 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-09-02 09:45:00 | 193.25 | 2025-09-02 10:15:00 | 194.09 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-02 09:45:00 | 193.25 | 2025-09-02 15:20:00 | 201.19 | TARGET_HIT | 0.50 | 4.11% |
| BUY | retest1 | 2025-09-03 09:35:00 | 204.83 | 2025-09-03 09:45:00 | 204.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-08 09:30:00 | 214.85 | 2025-09-08 10:15:00 | 214.01 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-10 10:05:00 | 210.96 | 2025-09-10 10:20:00 | 210.26 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-11 09:35:00 | 212.10 | 2025-09-11 09:50:00 | 212.87 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-11 09:35:00 | 212.10 | 2025-09-11 12:20:00 | 214.90 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-09-15 10:35:00 | 220.25 | 2025-09-15 11:05:00 | 219.54 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-16 09:40:00 | 215.64 | 2025-09-16 09:55:00 | 216.29 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-17 10:40:00 | 214.87 | 2025-09-17 11:00:00 | 215.42 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-23 09:45:00 | 208.20 | 2025-09-23 10:05:00 | 208.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-25 09:55:00 | 209.93 | 2025-09-25 10:10:00 | 209.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-26 09:40:00 | 203.94 | 2025-09-26 10:05:00 | 202.94 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-09-26 09:40:00 | 203.94 | 2025-09-26 15:20:00 | 200.43 | TARGET_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2025-10-03 09:45:00 | 217.96 | 2025-10-03 09:50:00 | 219.21 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-03 09:45:00 | 217.96 | 2025-10-03 14:30:00 | 220.90 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2025-10-07 09:30:00 | 219.70 | 2025-10-07 09:35:00 | 218.96 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-10-14 11:15:00 | 225.84 | 2025-10-14 11:50:00 | 226.44 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-15 10:00:00 | 227.16 | 2025-10-15 12:15:00 | 226.46 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-17 09:30:00 | 230.12 | 2025-10-17 09:45:00 | 229.41 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-23 09:30:00 | 229.78 | 2025-10-23 09:35:00 | 229.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-28 09:45:00 | 240.23 | 2025-10-28 09:50:00 | 239.42 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-29 09:30:00 | 240.50 | 2025-10-29 09:50:00 | 239.79 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-03 09:50:00 | 240.05 | 2025-11-03 10:25:00 | 238.95 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-11-04 10:35:00 | 234.80 | 2025-11-04 10:50:00 | 233.82 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-11-04 10:35:00 | 234.80 | 2025-11-04 15:20:00 | 233.30 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2025-11-07 10:45:00 | 234.00 | 2025-11-07 11:10:00 | 235.17 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-07 10:45:00 | 234.00 | 2025-11-07 12:10:00 | 234.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 09:30:00 | 270.32 | 2025-11-13 09:45:00 | 271.48 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-11-13 09:30:00 | 270.32 | 2025-11-13 10:00:00 | 270.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-17 11:15:00 | 259.54 | 2025-11-17 11:30:00 | 260.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-19 09:35:00 | 259.63 | 2025-11-19 09:50:00 | 258.81 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-20 09:30:00 | 260.90 | 2025-11-20 09:35:00 | 261.98 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-11-20 09:30:00 | 260.90 | 2025-11-20 09:45:00 | 260.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-25 10:50:00 | 254.40 | 2025-11-25 10:55:00 | 253.73 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-28 09:30:00 | 264.12 | 2025-11-28 09:35:00 | 263.35 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-10 09:30:00 | 266.35 | 2025-12-10 10:10:00 | 265.35 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-11 09:45:00 | 268.20 | 2025-12-11 09:50:00 | 267.19 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-12-12 09:35:00 | 270.25 | 2025-12-12 09:55:00 | 271.48 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-12 09:35:00 | 270.25 | 2025-12-12 10:20:00 | 270.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 09:35:00 | 280.50 | 2025-12-17 10:05:00 | 279.47 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-18 10:25:00 | 279.50 | 2025-12-18 11:50:00 | 280.83 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-12-18 10:25:00 | 279.50 | 2025-12-18 14:10:00 | 279.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-19 11:10:00 | 277.45 | 2025-12-19 12:35:00 | 278.22 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-26 09:35:00 | 299.90 | 2025-12-26 09:55:00 | 301.29 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-12-26 09:35:00 | 299.90 | 2025-12-26 15:20:00 | 307.20 | TARGET_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2026-01-13 09:45:00 | 358.60 | 2026-01-13 09:55:00 | 357.02 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-01-20 09:30:00 | 364.30 | 2026-01-20 09:35:00 | 362.03 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-01-20 09:30:00 | 364.30 | 2026-01-20 10:40:00 | 363.25 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2026-01-22 09:55:00 | 360.15 | 2026-01-22 10:00:00 | 357.55 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-01-22 09:55:00 | 360.15 | 2026-01-22 10:05:00 | 360.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-06 10:15:00 | 351.15 | 2026-02-06 11:00:00 | 352.75 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-02-12 09:35:00 | 370.35 | 2026-02-12 09:50:00 | 372.00 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-12 09:35:00 | 370.35 | 2026-02-12 10:15:00 | 370.70 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2026-02-19 10:55:00 | 342.45 | 2026-02-19 12:10:00 | 340.93 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-19 10:55:00 | 342.45 | 2026-02-19 12:20:00 | 342.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:30:00 | 429.70 | 2026-04-21 09:35:00 | 428.52 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-05 09:55:00 | 416.75 | 2026-05-05 10:00:00 | 414.66 | STOP_HIT | 1.00 | -0.50% |
