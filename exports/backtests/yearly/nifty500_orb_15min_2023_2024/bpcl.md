# Bharat Petroleum Corporation Ltd. (BPCL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-05-30 15:25:00 (38018 bars)
- **Last close:** 317.50
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
| ENTRY1 | 87 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 21 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 66
- **Target hits / Stop hits / Partials:** 21 / 66 / 35
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 23.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 84 | 42 | 50.0% | 17 | 42 | 25 | 0.22% | 18.4% |
| BUY @ 2nd Alert (retest1) | 84 | 42 | 50.0% | 17 | 42 | 25 | 0.22% | 18.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 14 | 36.8% | 4 | 24 | 10 | 0.13% | 4.9% |
| SELL @ 2nd Alert (retest1) | 38 | 14 | 36.8% | 4 | 24 | 10 | 0.13% | 4.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 122 | 56 | 45.9% | 21 | 66 | 35 | 0.19% | 23.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:30:00 | 180.15 | 179.58 | 0.00 | ORB-long ORB[178.50,180.10] vol=1.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2023-05-16 09:35:00 | 179.71 | 179.62 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 10:10:00 | 181.55 | 181.34 | 0.00 | ORB-long ORB[180.50,181.50] vol=1.9x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-05-30 10:15:00 | 181.22 | 181.34 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 09:30:00 | 181.60 | 180.84 | 0.00 | ORB-long ORB[180.10,181.40] vol=1.7x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 10:00:00 | 182.20 | 181.32 | 0.00 | T1 1.5R @ 182.20 |
| Target hit | 2023-05-31 12:30:00 | 181.88 | 181.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2023-06-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:00:00 | 180.00 | 179.30 | 0.00 | ORB-long ORB[178.38,179.53] vol=1.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 10:05:00 | 180.55 | 179.59 | 0.00 | T1 1.5R @ 180.55 |
| Target hit | 2023-06-07 15:20:00 | 183.93 | 182.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2023-06-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:50:00 | 182.75 | 182.07 | 0.00 | ORB-long ORB[180.10,182.73] vol=1.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-06-12 11:35:00 | 182.39 | 182.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:20:00 | 188.83 | 187.21 | 0.00 | ORB-long ORB[185.45,186.80] vol=1.8x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-06-14 10:40:00 | 188.38 | 187.81 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:30:00 | 188.00 | 187.39 | 0.00 | ORB-long ORB[186.50,187.75] vol=3.0x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 10:25:00 | 188.72 | 187.74 | 0.00 | T1 1.5R @ 188.72 |
| Target hit | 2023-06-15 12:35:00 | 189.03 | 189.16 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2023-06-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 10:30:00 | 187.25 | 186.55 | 0.00 | ORB-long ORB[185.68,187.18] vol=1.5x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-06-21 10:40:00 | 186.97 | 186.58 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:45:00 | 187.63 | 186.80 | 0.00 | ORB-long ORB[185.58,187.18] vol=2.9x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 11:05:00 | 188.07 | 187.10 | 0.00 | T1 1.5R @ 188.07 |
| Target hit | 2023-06-22 11:40:00 | 188.00 | 188.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2023-06-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 11:05:00 | 181.28 | 180.38 | 0.00 | ORB-long ORB[179.00,180.50] vol=1.5x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-06-27 11:10:00 | 180.92 | 180.40 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 11:15:00 | 181.53 | 180.60 | 0.00 | ORB-long ORB[180.08,181.20] vol=2.2x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 11:35:00 | 182.08 | 180.83 | 0.00 | T1 1.5R @ 182.08 |
| Target hit | 2023-06-28 15:20:00 | 182.95 | 181.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2023-06-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 10:05:00 | 180.80 | 182.74 | 0.00 | ORB-short ORB[182.55,184.40] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2023-06-30 10:25:00 | 181.36 | 182.43 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 10:20:00 | 185.65 | 184.06 | 0.00 | ORB-long ORB[181.70,183.18] vol=1.8x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 10:25:00 | 186.35 | 184.69 | 0.00 | T1 1.5R @ 186.35 |
| Target hit | 2023-07-03 12:55:00 | 187.35 | 187.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2023-07-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 10:05:00 | 189.05 | 188.01 | 0.00 | ORB-long ORB[186.58,188.60] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-07-04 10:10:00 | 188.44 | 188.03 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:05:00 | 189.40 | 188.48 | 0.00 | ORB-long ORB[187.50,188.93] vol=1.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:15:00 | 190.05 | 189.00 | 0.00 | T1 1.5R @ 190.05 |
| Target hit | 2023-07-05 15:20:00 | 193.43 | 191.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2023-07-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:35:00 | 195.85 | 194.51 | 0.00 | ORB-long ORB[192.50,195.00] vol=2.9x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-07-06 10:00:00 | 195.14 | 194.94 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:30:00 | 198.53 | 197.78 | 0.00 | ORB-long ORB[195.98,198.28] vol=2.3x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-07-07 09:35:00 | 197.96 | 197.81 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 09:30:00 | 193.68 | 194.59 | 0.00 | ORB-short ORB[193.95,196.58] vol=1.7x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 09:35:00 | 192.84 | 194.29 | 0.00 | T1 1.5R @ 192.84 |
| Stop hit — per-position SL triggered | 2023-07-10 09:40:00 | 193.68 | 194.21 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 09:40:00 | 192.43 | 193.07 | 0.00 | ORB-short ORB[193.18,193.83] vol=3.2x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-07-13 09:45:00 | 192.82 | 193.12 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:45:00 | 192.90 | 192.15 | 0.00 | ORB-long ORB[191.03,192.40] vol=2.1x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 09:50:00 | 193.55 | 192.48 | 0.00 | T1 1.5R @ 193.55 |
| Stop hit — per-position SL triggered | 2023-07-18 09:55:00 | 192.90 | 192.52 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:40:00 | 191.38 | 190.45 | 0.00 | ORB-long ORB[189.28,190.90] vol=2.0x ATR=0.41 |
| Stop hit — per-position SL triggered | 2023-07-19 09:45:00 | 190.97 | 190.52 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:40:00 | 191.85 | 192.95 | 0.00 | ORB-short ORB[192.30,193.88] vol=2.1x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-07-20 09:55:00 | 192.37 | 192.76 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 11:15:00 | 195.23 | 194.33 | 0.00 | ORB-long ORB[193.00,195.13] vol=2.6x ATR=0.43 |
| Stop hit — per-position SL triggered | 2023-07-21 11:25:00 | 194.80 | 194.37 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 11:15:00 | 195.58 | 194.75 | 0.00 | ORB-long ORB[193.88,195.30] vol=2.4x ATR=0.41 |
| Stop hit — per-position SL triggered | 2023-07-24 11:35:00 | 195.17 | 194.88 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 11:10:00 | 195.20 | 194.42 | 0.00 | ORB-long ORB[193.00,195.00] vol=3.0x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-07-25 11:20:00 | 194.84 | 194.42 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:15:00 | 195.15 | 194.36 | 0.00 | ORB-long ORB[193.40,194.60] vol=3.4x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 10:30:00 | 195.78 | 194.57 | 0.00 | T1 1.5R @ 195.78 |
| Target hit | 2023-07-26 12:10:00 | 195.63 | 195.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2023-07-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 10:15:00 | 193.63 | 194.53 | 0.00 | ORB-short ORB[194.18,195.83] vol=2.0x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 10:55:00 | 192.60 | 194.25 | 0.00 | T1 1.5R @ 192.60 |
| Target hit | 2023-07-27 15:20:00 | 188.93 | 190.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2023-07-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:45:00 | 187.65 | 187.41 | 0.00 | ORB-long ORB[186.23,187.60] vol=7.4x ATR=0.50 |
| Stop hit — per-position SL triggered | 2023-07-31 09:55:00 | 187.15 | 187.42 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:30:00 | 186.50 | 187.49 | 0.00 | ORB-short ORB[187.40,189.35] vol=2.3x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 10:50:00 | 185.86 | 187.29 | 0.00 | T1 1.5R @ 185.86 |
| Target hit | 2023-08-02 15:20:00 | 184.53 | 185.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 11:15:00 | 185.58 | 184.14 | 0.00 | ORB-long ORB[183.30,185.28] vol=2.0x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-08-03 11:50:00 | 185.09 | 184.31 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 10:10:00 | 182.65 | 183.62 | 0.00 | ORB-short ORB[183.33,185.43] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 11:40:00 | 181.86 | 183.01 | 0.00 | T1 1.5R @ 181.86 |
| Target hit | 2023-08-04 15:20:00 | 180.18 | 181.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2023-08-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 11:00:00 | 179.38 | 180.38 | 0.00 | ORB-short ORB[179.95,181.93] vol=2.2x ATR=0.38 |
| Stop hit — per-position SL triggered | 2023-08-08 12:50:00 | 179.76 | 180.08 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 10:35:00 | 175.85 | 176.62 | 0.00 | ORB-short ORB[176.05,177.10] vol=2.1x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 10:55:00 | 175.31 | 176.35 | 0.00 | T1 1.5R @ 175.31 |
| Target hit | 2023-08-22 15:20:00 | 173.48 | 174.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2023-08-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 10:20:00 | 172.73 | 173.66 | 0.00 | ORB-short ORB[173.30,174.73] vol=2.5x ATR=0.41 |
| Stop hit — per-position SL triggered | 2023-08-23 12:15:00 | 173.14 | 173.33 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 10:20:00 | 176.18 | 174.50 | 0.00 | ORB-long ORB[173.08,173.85] vol=4.1x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-08-24 10:25:00 | 175.76 | 174.65 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:40:00 | 178.15 | 177.15 | 0.00 | ORB-long ORB[176.15,177.83] vol=5.9x ATR=0.56 |
| Stop hit — per-position SL triggered | 2023-08-28 10:00:00 | 177.59 | 177.71 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 09:30:00 | 175.73 | 176.35 | 0.00 | ORB-short ORB[175.80,177.40] vol=1.5x ATR=0.44 |
| Stop hit — per-position SL triggered | 2023-08-30 09:50:00 | 176.17 | 176.04 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:15:00 | 173.00 | 174.26 | 0.00 | ORB-short ORB[174.63,176.00] vol=3.2x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 11:20:00 | 172.57 | 174.11 | 0.00 | T1 1.5R @ 172.57 |
| Stop hit — per-position SL triggered | 2023-08-31 12:35:00 | 173.00 | 173.68 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:50:00 | 174.88 | 173.98 | 0.00 | ORB-long ORB[172.63,173.88] vol=1.6x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 10:05:00 | 175.39 | 174.29 | 0.00 | T1 1.5R @ 175.39 |
| Target hit | 2023-09-05 11:35:00 | 175.03 | 175.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2023-09-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 11:00:00 | 175.53 | 175.11 | 0.00 | ORB-long ORB[174.03,175.10] vol=2.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-09-06 11:35:00 | 175.27 | 175.20 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 11:10:00 | 178.35 | 177.71 | 0.00 | ORB-long ORB[177.30,178.15] vol=2.2x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 11:20:00 | 178.80 | 177.82 | 0.00 | T1 1.5R @ 178.80 |
| Stop hit — per-position SL triggered | 2023-09-08 12:00:00 | 178.35 | 178.10 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:35:00 | 184.15 | 182.15 | 0.00 | ORB-long ORB[180.85,182.80] vol=3.7x ATR=0.55 |
| Stop hit — per-position SL triggered | 2023-09-11 10:45:00 | 183.60 | 182.53 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:35:00 | 178.50 | 180.23 | 0.00 | ORB-short ORB[180.15,182.75] vol=1.7x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:50:00 | 177.63 | 179.64 | 0.00 | T1 1.5R @ 177.63 |
| Stop hit — per-position SL triggered | 2023-09-12 10:00:00 | 178.50 | 179.48 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-09-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:35:00 | 179.25 | 178.70 | 0.00 | ORB-long ORB[178.08,179.03] vol=2.0x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 09:55:00 | 179.96 | 179.28 | 0.00 | T1 1.5R @ 179.96 |
| Stop hit — per-position SL triggered | 2023-09-14 10:00:00 | 179.25 | 179.30 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-09-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 09:45:00 | 176.48 | 177.36 | 0.00 | ORB-short ORB[176.75,178.40] vol=1.5x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 10:00:00 | 175.70 | 176.99 | 0.00 | T1 1.5R @ 175.70 |
| Stop hit — per-position SL triggered | 2023-09-20 10:10:00 | 176.48 | 176.92 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-09-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:00:00 | 174.15 | 175.27 | 0.00 | ORB-short ORB[175.10,176.75] vol=2.0x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-09-25 10:15:00 | 174.61 | 175.10 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 09:30:00 | 174.90 | 175.42 | 0.00 | ORB-short ORB[175.03,176.60] vol=2.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2023-09-27 09:35:00 | 175.34 | 175.38 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:35:00 | 172.30 | 171.68 | 0.00 | ORB-long ORB[170.75,171.78] vol=2.1x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-10-11 11:00:00 | 172.00 | 172.12 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 11:05:00 | 173.88 | 172.39 | 0.00 | ORB-long ORB[171.00,172.75] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-10-16 11:30:00 | 173.55 | 172.52 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:35:00 | 174.83 | 174.05 | 0.00 | ORB-long ORB[173.35,174.45] vol=1.5x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 10:00:00 | 175.31 | 174.61 | 0.00 | T1 1.5R @ 175.31 |
| Target hit | 2023-10-17 15:10:00 | 177.33 | 177.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — SELL (started 2023-10-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 10:55:00 | 166.60 | 167.98 | 0.00 | ORB-short ORB[168.25,169.48] vol=1.9x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-10-26 11:05:00 | 167.06 | 167.92 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 09:35:00 | 176.70 | 175.70 | 0.00 | ORB-long ORB[173.98,176.00] vol=1.9x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-10-31 09:45:00 | 176.13 | 175.88 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:40:00 | 180.63 | 180.00 | 0.00 | ORB-long ORB[178.63,180.48] vol=2.3x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-11-02 11:00:00 | 180.17 | 180.04 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 09:50:00 | 181.53 | 181.01 | 0.00 | ORB-long ORB[180.10,180.98] vol=1.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-11-03 10:05:00 | 181.21 | 181.11 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:40:00 | 182.48 | 181.97 | 0.00 | ORB-long ORB[180.68,182.33] vol=2.2x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 10:15:00 | 183.08 | 182.40 | 0.00 | T1 1.5R @ 183.08 |
| Target hit | 2023-11-06 13:10:00 | 183.00 | 183.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — BUY (started 2023-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 10:45:00 | 185.08 | 184.22 | 0.00 | ORB-long ORB[183.10,184.60] vol=3.4x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 11:00:00 | 185.57 | 184.47 | 0.00 | T1 1.5R @ 185.57 |
| Stop hit — per-position SL triggered | 2023-11-07 11:25:00 | 185.08 | 184.62 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-11-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 11:10:00 | 194.28 | 192.96 | 0.00 | ORB-long ORB[191.75,193.73] vol=4.5x ATR=0.47 |
| Stop hit — per-position SL triggered | 2023-11-13 11:15:00 | 193.81 | 192.99 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 10:45:00 | 195.40 | 194.39 | 0.00 | ORB-long ORB[193.63,194.95] vol=2.5x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 10:50:00 | 196.05 | 194.59 | 0.00 | T1 1.5R @ 196.05 |
| Stop hit — per-position SL triggered | 2023-11-15 10:55:00 | 195.40 | 194.62 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 11:15:00 | 215.95 | 212.61 | 0.00 | ORB-long ORB[210.05,213.25] vol=6.5x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-11-29 11:35:00 | 215.01 | 213.52 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-11-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:50:00 | 215.98 | 217.21 | 0.00 | ORB-short ORB[216.00,218.50] vol=1.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 216.73 | 217.11 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-12-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-01 11:10:00 | 217.10 | 219.36 | 0.00 | ORB-short ORB[218.50,221.18] vol=2.0x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-12-01 11:30:00 | 217.71 | 219.15 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:55:00 | 224.55 | 223.65 | 0.00 | ORB-long ORB[222.03,224.50] vol=1.6x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 10:10:00 | 225.98 | 223.88 | 0.00 | T1 1.5R @ 225.98 |
| Target hit | 2023-12-04 15:20:00 | 231.15 | 228.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2023-12-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:05:00 | 235.30 | 237.21 | 0.00 | ORB-short ORB[236.50,238.73] vol=1.6x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:15:00 | 234.31 | 236.56 | 0.00 | T1 1.5R @ 234.31 |
| Stop hit — per-position SL triggered | 2023-12-08 10:30:00 | 235.30 | 236.36 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:45:00 | 224.25 | 222.99 | 0.00 | ORB-long ORB[221.10,223.78] vol=2.8x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 09:55:00 | 225.23 | 223.52 | 0.00 | T1 1.5R @ 225.23 |
| Stop hit — per-position SL triggered | 2023-12-15 10:45:00 | 224.25 | 224.19 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 11:05:00 | 227.20 | 226.50 | 0.00 | ORB-long ORB[225.63,226.95] vol=2.1x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-12-20 11:25:00 | 226.71 | 226.59 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 10:55:00 | 224.10 | 224.90 | 0.00 | ORB-short ORB[224.83,226.68] vol=2.4x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 11:30:00 | 223.28 | 224.61 | 0.00 | T1 1.5R @ 223.28 |
| Stop hit — per-position SL triggered | 2023-12-22 12:25:00 | 224.10 | 224.30 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:50:00 | 225.25 | 223.81 | 0.00 | ORB-long ORB[222.25,224.98] vol=1.7x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 10:20:00 | 226.32 | 224.41 | 0.00 | T1 1.5R @ 226.32 |
| Target hit | 2023-12-26 11:20:00 | 226.48 | 226.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — BUY (started 2023-12-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:30:00 | 230.50 | 229.11 | 0.00 | ORB-long ORB[227.53,229.85] vol=3.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2023-12-28 10:40:00 | 229.83 | 229.24 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:35:00 | 227.15 | 226.22 | 0.00 | ORB-long ORB[225.05,226.78] vol=1.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 09:45:00 | 227.99 | 226.59 | 0.00 | T1 1.5R @ 227.99 |
| Stop hit — per-position SL triggered | 2024-01-02 09:55:00 | 227.15 | 226.72 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:45:00 | 228.03 | 228.79 | 0.00 | ORB-short ORB[228.55,229.98] vol=1.7x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-01-03 09:50:00 | 228.74 | 228.78 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:55:00 | 228.60 | 229.18 | 0.00 | ORB-short ORB[228.90,230.90] vol=2.3x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-01-09 10:20:00 | 229.21 | 229.03 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:05:00 | 228.38 | 227.70 | 0.00 | ORB-long ORB[226.60,228.13] vol=1.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-01-11 11:55:00 | 227.77 | 228.25 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-01-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:50:00 | 240.25 | 238.55 | 0.00 | ORB-long ORB[234.70,237.70] vol=2.3x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 11:00:00 | 241.66 | 238.85 | 0.00 | T1 1.5R @ 241.66 |
| Target hit | 2024-01-29 15:20:00 | 247.10 | 244.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2024-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 09:35:00 | 251.20 | 249.15 | 0.00 | ORB-long ORB[246.58,249.58] vol=2.5x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-01-30 09:50:00 | 249.71 | 249.96 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 10:15:00 | 331.95 | 328.65 | 0.00 | ORB-long ORB[326.05,330.90] vol=2.7x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-02-19 10:20:00 | 330.27 | 329.25 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 307.55 | 310.30 | 0.00 | ORB-short ORB[310.10,312.77] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-02-28 11:20:00 | 308.67 | 309.67 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-03-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 10:00:00 | 315.38 | 313.05 | 0.00 | ORB-long ORB[310.43,313.73] vol=3.8x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 10:15:00 | 317.61 | 314.38 | 0.00 | T1 1.5R @ 317.61 |
| Target hit | 2024-03-04 15:20:00 | 321.02 | 318.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2024-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:50:00 | 307.33 | 310.04 | 0.00 | ORB-short ORB[310.58,314.00] vol=2.3x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-03-13 10:10:00 | 308.90 | 309.33 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-03-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 09:30:00 | 296.00 | 294.05 | 0.00 | ORB-long ORB[292.27,295.60] vol=2.0x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 09:35:00 | 297.51 | 294.85 | 0.00 | T1 1.5R @ 297.51 |
| Target hit | 2024-03-22 10:30:00 | 298.27 | 298.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — BUY (started 2024-03-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 10:55:00 | 300.70 | 296.58 | 0.00 | ORB-long ORB[292.68,296.50] vol=2.1x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-03-26 15:05:00 | 299.55 | 298.67 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 10:55:00 | 300.83 | 302.88 | 0.00 | ORB-short ORB[301.30,304.40] vol=2.3x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-04-01 11:10:00 | 301.75 | 302.68 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-04-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:30:00 | 307.83 | 306.60 | 0.00 | ORB-long ORB[304.40,307.48] vol=1.7x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 09:35:00 | 309.23 | 306.99 | 0.00 | T1 1.5R @ 309.23 |
| Stop hit — per-position SL triggered | 2024-04-03 09:40:00 | 307.83 | 307.12 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:30:00 | 304.75 | 306.00 | 0.00 | ORB-short ORB[305.05,308.05] vol=1.8x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-04-04 09:35:00 | 305.74 | 305.96 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-04-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 10:30:00 | 305.98 | 302.96 | 0.00 | ORB-long ORB[299.50,303.38] vol=1.9x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-04-12 10:40:00 | 304.77 | 303.18 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-04-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:50:00 | 294.98 | 296.90 | 0.00 | ORB-short ORB[295.70,298.93] vol=2.4x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-04-25 11:05:00 | 295.83 | 296.62 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-04-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 11:00:00 | 305.98 | 303.15 | 0.00 | ORB-long ORB[299.55,301.60] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-04-26 12:15:00 | 304.83 | 304.12 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 10:00:00 | 308.15 | 306.35 | 0.00 | ORB-long ORB[304.05,307.48] vol=2.4x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 10:05:00 | 309.86 | 307.02 | 0.00 | T1 1.5R @ 309.86 |
| Target hit | 2024-04-29 14:40:00 | 308.43 | 309.16 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-16 09:30:00 | 180.15 | 2023-05-16 09:35:00 | 179.71 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-30 10:10:00 | 181.55 | 2023-05-30 10:15:00 | 181.22 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-05-31 09:30:00 | 181.60 | 2023-05-31 10:00:00 | 182.20 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-05-31 09:30:00 | 181.60 | 2023-05-31 12:30:00 | 181.88 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2023-06-07 10:00:00 | 180.00 | 2023-06-07 10:05:00 | 180.55 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-06-07 10:00:00 | 180.00 | 2023-06-07 15:20:00 | 183.93 | TARGET_HIT | 0.50 | 2.18% |
| BUY | retest1 | 2023-06-12 10:50:00 | 182.75 | 2023-06-12 11:35:00 | 182.39 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-14 10:20:00 | 188.83 | 2023-06-14 10:40:00 | 188.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-06-15 09:30:00 | 188.00 | 2023-06-15 10:25:00 | 188.72 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-15 09:30:00 | 188.00 | 2023-06-15 12:35:00 | 189.03 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2023-06-21 10:30:00 | 187.25 | 2023-06-21 10:40:00 | 186.97 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-06-22 10:45:00 | 187.63 | 2023-06-22 11:05:00 | 188.07 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-06-22 10:45:00 | 187.63 | 2023-06-22 11:40:00 | 188.00 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2023-06-27 11:05:00 | 181.28 | 2023-06-27 11:10:00 | 180.92 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-28 11:15:00 | 181.53 | 2023-06-28 11:35:00 | 182.08 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-06-28 11:15:00 | 181.53 | 2023-06-28 15:20:00 | 182.95 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2023-06-30 10:05:00 | 180.80 | 2023-06-30 10:25:00 | 181.36 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-03 10:20:00 | 185.65 | 2023-07-03 10:25:00 | 186.35 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-07-03 10:20:00 | 185.65 | 2023-07-03 12:55:00 | 187.35 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2023-07-04 10:05:00 | 189.05 | 2023-07-04 10:10:00 | 188.44 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-07-05 10:05:00 | 189.40 | 2023-07-05 10:15:00 | 190.05 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-07-05 10:05:00 | 189.40 | 2023-07-05 15:20:00 | 193.43 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2023-07-06 09:35:00 | 195.85 | 2023-07-06 10:00:00 | 195.14 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-07-07 09:30:00 | 198.53 | 2023-07-07 09:35:00 | 197.96 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-07-10 09:30:00 | 193.68 | 2023-07-10 09:35:00 | 192.84 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-07-10 09:30:00 | 193.68 | 2023-07-10 09:40:00 | 193.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-13 09:40:00 | 192.43 | 2023-07-13 09:45:00 | 192.82 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-18 09:45:00 | 192.90 | 2023-07-18 09:50:00 | 193.55 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-07-18 09:45:00 | 192.90 | 2023-07-18 09:55:00 | 192.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-19 09:40:00 | 191.38 | 2023-07-19 09:45:00 | 190.97 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-07-20 09:40:00 | 191.85 | 2023-07-20 09:55:00 | 192.37 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-21 11:15:00 | 195.23 | 2023-07-21 11:25:00 | 194.80 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-24 11:15:00 | 195.58 | 2023-07-24 11:35:00 | 195.17 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-25 11:10:00 | 195.20 | 2023-07-25 11:20:00 | 194.84 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-07-26 10:15:00 | 195.15 | 2023-07-26 10:30:00 | 195.78 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-07-26 10:15:00 | 195.15 | 2023-07-26 12:10:00 | 195.63 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2023-07-27 10:15:00 | 193.63 | 2023-07-27 10:55:00 | 192.60 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-07-27 10:15:00 | 193.63 | 2023-07-27 15:20:00 | 188.93 | TARGET_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2023-07-31 09:45:00 | 187.65 | 2023-07-31 09:55:00 | 187.15 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-02 10:30:00 | 186.50 | 2023-08-02 10:50:00 | 185.86 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-08-02 10:30:00 | 186.50 | 2023-08-02 15:20:00 | 184.53 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2023-08-03 11:15:00 | 185.58 | 2023-08-03 11:50:00 | 185.09 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-04 10:10:00 | 182.65 | 2023-08-04 11:40:00 | 181.86 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-08-04 10:10:00 | 182.65 | 2023-08-04 15:20:00 | 180.18 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2023-08-08 11:00:00 | 179.38 | 2023-08-08 12:50:00 | 179.76 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-08-22 10:35:00 | 175.85 | 2023-08-22 10:55:00 | 175.31 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-08-22 10:35:00 | 175.85 | 2023-08-22 15:20:00 | 173.48 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2023-08-23 10:20:00 | 172.73 | 2023-08-23 12:15:00 | 173.14 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-24 10:20:00 | 176.18 | 2023-08-24 10:25:00 | 175.76 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-28 09:40:00 | 178.15 | 2023-08-28 10:00:00 | 177.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-08-30 09:30:00 | 175.73 | 2023-08-30 09:50:00 | 176.17 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-31 11:15:00 | 173.00 | 2023-08-31 11:20:00 | 172.57 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-08-31 11:15:00 | 173.00 | 2023-08-31 12:35:00 | 173.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-05 09:50:00 | 174.88 | 2023-09-05 10:05:00 | 175.39 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-09-05 09:50:00 | 174.88 | 2023-09-05 11:35:00 | 175.03 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2023-09-06 11:00:00 | 175.53 | 2023-09-06 11:35:00 | 175.27 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-09-08 11:10:00 | 178.35 | 2023-09-08 11:20:00 | 178.80 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-09-08 11:10:00 | 178.35 | 2023-09-08 12:00:00 | 178.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-11 10:35:00 | 184.15 | 2023-09-11 10:45:00 | 183.60 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-12 09:35:00 | 178.50 | 2023-09-12 09:50:00 | 177.63 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-09-12 09:35:00 | 178.50 | 2023-09-12 10:00:00 | 178.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-14 09:35:00 | 179.25 | 2023-09-14 09:55:00 | 179.96 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-09-14 09:35:00 | 179.25 | 2023-09-14 10:00:00 | 179.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-20 09:45:00 | 176.48 | 2023-09-20 10:00:00 | 175.70 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-09-20 09:45:00 | 176.48 | 2023-09-20 10:10:00 | 176.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-25 10:00:00 | 174.15 | 2023-09-25 10:15:00 | 174.61 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-27 09:30:00 | 174.90 | 2023-09-27 09:35:00 | 175.34 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-11 09:35:00 | 172.30 | 2023-10-11 11:00:00 | 172.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-10-16 11:05:00 | 173.88 | 2023-10-16 11:30:00 | 173.55 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-17 09:35:00 | 174.83 | 2023-10-17 10:00:00 | 175.31 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-10-17 09:35:00 | 174.83 | 2023-10-17 15:10:00 | 177.33 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2023-10-26 10:55:00 | 166.60 | 2023-10-26 11:05:00 | 167.06 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-10-31 09:35:00 | 176.70 | 2023-10-31 09:45:00 | 176.13 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-11-02 10:40:00 | 180.63 | 2023-11-02 11:00:00 | 180.17 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-11-03 09:50:00 | 181.53 | 2023-11-03 10:05:00 | 181.21 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-11-06 09:40:00 | 182.48 | 2023-11-06 10:15:00 | 183.08 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-11-06 09:40:00 | 182.48 | 2023-11-06 13:10:00 | 183.00 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2023-11-07 10:45:00 | 185.08 | 2023-11-07 11:00:00 | 185.57 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-11-07 10:45:00 | 185.08 | 2023-11-07 11:25:00 | 185.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-13 11:10:00 | 194.28 | 2023-11-13 11:15:00 | 193.81 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-11-15 10:45:00 | 195.40 | 2023-11-15 10:50:00 | 196.05 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-11-15 10:45:00 | 195.40 | 2023-11-15 10:55:00 | 195.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 11:15:00 | 215.95 | 2023-11-29 11:35:00 | 215.01 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-11-30 09:50:00 | 215.98 | 2023-11-30 10:15:00 | 216.73 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-12-01 11:10:00 | 217.10 | 2023-12-01 11:30:00 | 217.71 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-12-04 09:55:00 | 224.55 | 2023-12-04 10:10:00 | 225.98 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-12-04 09:55:00 | 224.55 | 2023-12-04 15:20:00 | 231.15 | TARGET_HIT | 0.50 | 2.94% |
| SELL | retest1 | 2023-12-08 10:05:00 | 235.30 | 2023-12-08 10:15:00 | 234.31 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-12-08 10:05:00 | 235.30 | 2023-12-08 10:30:00 | 235.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-15 09:45:00 | 224.25 | 2023-12-15 09:55:00 | 225.23 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-12-15 09:45:00 | 224.25 | 2023-12-15 10:45:00 | 224.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-20 11:05:00 | 227.20 | 2023-12-20 11:25:00 | 226.71 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-12-22 10:55:00 | 224.10 | 2023-12-22 11:30:00 | 223.28 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-12-22 10:55:00 | 224.10 | 2023-12-22 12:25:00 | 224.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 09:50:00 | 225.25 | 2023-12-26 10:20:00 | 226.32 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-12-26 09:50:00 | 225.25 | 2023-12-26 11:20:00 | 226.48 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2023-12-28 10:30:00 | 230.50 | 2023-12-28 10:40:00 | 229.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-01-02 09:35:00 | 227.15 | 2024-01-02 09:45:00 | 227.99 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-01-02 09:35:00 | 227.15 | 2024-01-02 09:55:00 | 227.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-03 09:45:00 | 228.03 | 2024-01-03 09:50:00 | 228.74 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-01-09 09:55:00 | 228.60 | 2024-01-09 10:20:00 | 229.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-01-11 10:05:00 | 228.38 | 2024-01-11 11:55:00 | 227.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-01-29 10:50:00 | 240.25 | 2024-01-29 11:00:00 | 241.66 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-01-29 10:50:00 | 240.25 | 2024-01-29 15:20:00 | 247.10 | TARGET_HIT | 0.50 | 2.85% |
| BUY | retest1 | 2024-01-30 09:35:00 | 251.20 | 2024-01-30 09:50:00 | 249.71 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-02-19 10:15:00 | 331.95 | 2024-02-19 10:20:00 | 330.27 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-02-28 10:50:00 | 307.55 | 2024-02-28 11:20:00 | 308.67 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-03-04 10:00:00 | 315.38 | 2024-03-04 10:15:00 | 317.61 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-03-04 10:00:00 | 315.38 | 2024-03-04 15:20:00 | 321.02 | TARGET_HIT | 0.50 | 1.79% |
| SELL | retest1 | 2024-03-13 09:50:00 | 307.33 | 2024-03-13 10:10:00 | 308.90 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-03-22 09:30:00 | 296.00 | 2024-03-22 09:35:00 | 297.51 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-03-22 09:30:00 | 296.00 | 2024-03-22 10:30:00 | 298.27 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2024-03-26 10:55:00 | 300.70 | 2024-03-26 15:05:00 | 299.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-04-01 10:55:00 | 300.83 | 2024-04-01 11:10:00 | 301.75 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-03 09:30:00 | 307.83 | 2024-04-03 09:35:00 | 309.23 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-04-03 09:30:00 | 307.83 | 2024-04-03 09:40:00 | 307.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-04 09:30:00 | 304.75 | 2024-04-04 09:35:00 | 305.74 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-04-12 10:30:00 | 305.98 | 2024-04-12 10:40:00 | 304.77 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-04-25 10:50:00 | 294.98 | 2024-04-25 11:05:00 | 295.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-26 11:00:00 | 305.98 | 2024-04-26 12:15:00 | 304.83 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-04-29 10:00:00 | 308.15 | 2024-04-29 10:05:00 | 309.86 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-04-29 10:00:00 | 308.15 | 2024-04-29 14:40:00 | 308.43 | TARGET_HIT | 0.50 | 0.09% |
