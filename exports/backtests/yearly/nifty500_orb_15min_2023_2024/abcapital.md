# Aditya Birla Capital Ltd. (ABCAPITAL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-02-24 15:25:00 (51831 bars)
- **Last close:** 350.90
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
| ENTRY1 | 106 |
| ENTRY2 | 0 |
| PARTIAL | 48 |
| TARGET_HIT | 22 |
| STOP_HIT | 84 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 154 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 70 / 84
- **Target hits / Stop hits / Partials:** 22 / 84 / 48
- **Avg / median % per leg:** 0.29% / 0.00%
- **Sum % (uncompounded):** 44.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 87 | 34 | 39.1% | 10 | 53 | 24 | 0.12% | 10.7% |
| BUY @ 2nd Alert (retest1) | 87 | 34 | 39.1% | 10 | 53 | 24 | 0.12% | 10.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 67 | 36 | 53.7% | 12 | 31 | 24 | 0.50% | 33.6% |
| SELL @ 2nd Alert (retest1) | 67 | 36 | 53.7% | 12 | 31 | 24 | 0.50% | 33.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 154 | 70 | 45.5% | 22 | 84 | 48 | 0.29% | 44.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 10:35:00 | 168.90 | 168.38 | 0.00 | ORB-long ORB[165.80,167.70] vol=1.8x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-05-16 10:50:00 | 168.33 | 168.42 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 09:30:00 | 167.15 | 166.62 | 0.00 | ORB-long ORB[165.45,167.05] vol=2.2x ATR=0.58 |
| Stop hit — per-position SL triggered | 2023-05-18 09:35:00 | 166.57 | 166.66 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 09:30:00 | 166.60 | 166.15 | 0.00 | ORB-long ORB[164.80,166.50] vol=2.1x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-05-26 09:40:00 | 165.98 | 166.16 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 10:05:00 | 167.45 | 166.95 | 0.00 | ORB-long ORB[166.00,167.40] vol=1.8x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-30 10:45:00 | 168.26 | 167.12 | 0.00 | T1 1.5R @ 168.26 |
| Target hit | 2023-05-30 15:20:00 | 171.60 | 168.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2023-05-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 10:25:00 | 169.95 | 171.02 | 0.00 | ORB-short ORB[170.20,171.60] vol=2.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 10:40:00 | 169.21 | 170.68 | 0.00 | T1 1.5R @ 169.21 |
| Stop hit — per-position SL triggered | 2023-05-31 10:50:00 | 169.95 | 170.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 09:55:00 | 170.35 | 171.76 | 0.00 | ORB-short ORB[171.65,173.95] vol=3.1x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 10:00:00 | 169.41 | 171.32 | 0.00 | T1 1.5R @ 169.41 |
| Target hit | 2023-06-05 13:15:00 | 169.50 | 169.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2023-06-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 09:45:00 | 174.65 | 173.41 | 0.00 | ORB-long ORB[171.50,174.00] vol=3.3x ATR=0.58 |
| Stop hit — per-position SL triggered | 2023-06-09 09:50:00 | 174.07 | 173.49 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 10:25:00 | 177.90 | 176.63 | 0.00 | ORB-long ORB[174.60,177.10] vol=2.3x ATR=0.50 |
| Stop hit — per-position SL triggered | 2023-06-16 10:45:00 | 177.40 | 176.99 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:30:00 | 180.20 | 179.67 | 0.00 | ORB-long ORB[178.10,179.65] vol=9.3x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 09:45:00 | 181.16 | 179.97 | 0.00 | T1 1.5R @ 181.16 |
| Stop hit — per-position SL triggered | 2023-06-19 09:55:00 | 180.20 | 180.02 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 176.85 | 178.21 | 0.00 | ORB-short ORB[177.60,179.90] vol=3.2x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-06-20 09:35:00 | 177.46 | 178.13 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 09:40:00 | 176.85 | 178.43 | 0.00 | ORB-short ORB[179.15,180.50] vol=3.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2023-06-22 09:45:00 | 177.45 | 178.28 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-07-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 09:55:00 | 188.30 | 189.09 | 0.00 | ORB-short ORB[188.55,189.90] vol=2.4x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-07-06 11:20:00 | 188.87 | 188.57 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 11:10:00 | 191.75 | 189.53 | 0.00 | ORB-long ORB[188.50,190.70] vol=2.9x ATR=0.69 |
| Stop hit — per-position SL triggered | 2023-07-07 11:25:00 | 191.06 | 189.76 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-11 10:30:00 | 184.95 | 186.21 | 0.00 | ORB-short ORB[185.15,187.45] vol=2.2x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-07-11 11:20:00 | 185.57 | 185.73 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 10:05:00 | 186.20 | 185.04 | 0.00 | ORB-long ORB[183.80,185.45] vol=1.6x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 10:15:00 | 187.11 | 185.46 | 0.00 | T1 1.5R @ 187.11 |
| Stop hit — per-position SL triggered | 2023-07-12 10:20:00 | 186.20 | 185.55 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 11:00:00 | 188.40 | 189.14 | 0.00 | ORB-short ORB[188.85,190.40] vol=3.1x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 11:10:00 | 187.67 | 188.97 | 0.00 | T1 1.5R @ 187.67 |
| Target hit | 2023-07-17 13:45:00 | 186.80 | 186.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2023-07-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 09:30:00 | 186.60 | 186.35 | 0.00 | ORB-long ORB[184.90,186.55] vol=2.9x ATR=0.58 |
| Stop hit — per-position SL triggered | 2023-07-21 09:45:00 | 186.02 | 186.37 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 09:35:00 | 187.95 | 186.84 | 0.00 | ORB-long ORB[185.25,186.95] vol=5.3x ATR=0.60 |
| Stop hit — per-position SL triggered | 2023-07-24 09:40:00 | 187.35 | 186.93 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:45:00 | 190.45 | 189.55 | 0.00 | ORB-long ORB[188.10,189.85] vol=3.8x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 10:00:00 | 191.24 | 190.01 | 0.00 | T1 1.5R @ 191.24 |
| Stop hit — per-position SL triggered | 2023-07-26 10:10:00 | 190.45 | 190.10 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:50:00 | 194.75 | 193.47 | 0.00 | ORB-long ORB[191.55,194.25] vol=1.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2023-07-28 10:30:00 | 193.97 | 193.92 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-31 09:40:00 | 194.35 | 195.07 | 0.00 | ORB-short ORB[194.40,196.20] vol=1.6x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-07-31 10:00:00 | 194.99 | 194.90 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 09:40:00 | 195.70 | 196.30 | 0.00 | ORB-short ORB[195.90,196.95] vol=1.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 10:10:00 | 195.05 | 196.07 | 0.00 | T1 1.5R @ 195.05 |
| Stop hit — per-position SL triggered | 2023-08-01 10:55:00 | 195.70 | 195.80 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:25:00 | 192.65 | 193.94 | 0.00 | ORB-short ORB[193.20,195.40] vol=1.5x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 11:05:00 | 191.61 | 193.39 | 0.00 | T1 1.5R @ 191.61 |
| Target hit | 2023-08-02 14:35:00 | 191.00 | 188.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2023-08-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 09:45:00 | 189.45 | 188.58 | 0.00 | ORB-long ORB[186.00,188.80] vol=3.8x ATR=0.68 |
| Stop hit — per-position SL triggered | 2023-08-08 10:00:00 | 188.77 | 188.62 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:25:00 | 184.60 | 186.28 | 0.00 | ORB-short ORB[185.95,187.85] vol=1.5x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-08-10 10:40:00 | 185.22 | 186.11 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 09:30:00 | 186.25 | 184.98 | 0.00 | ORB-long ORB[183.10,185.40] vol=2.3x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 09:35:00 | 187.15 | 185.64 | 0.00 | T1 1.5R @ 187.15 |
| Target hit | 2023-08-11 10:10:00 | 186.50 | 186.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2023-08-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 10:05:00 | 179.35 | 180.17 | 0.00 | ORB-short ORB[180.05,181.25] vol=2.3x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-08-18 10:15:00 | 179.84 | 180.10 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:30:00 | 184.90 | 183.94 | 0.00 | ORB-long ORB[182.15,184.25] vol=3.3x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 10:15:00 | 185.80 | 184.47 | 0.00 | T1 1.5R @ 185.80 |
| Stop hit — per-position SL triggered | 2023-08-22 10:30:00 | 184.90 | 184.53 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 11:15:00 | 185.95 | 187.20 | 0.00 | ORB-short ORB[186.80,188.35] vol=2.9x ATR=0.41 |
| Stop hit — per-position SL triggered | 2023-08-24 11:45:00 | 186.36 | 187.10 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:10:00 | 181.90 | 182.78 | 0.00 | ORB-short ORB[182.35,183.75] vol=2.4x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 10:20:00 | 181.16 | 182.60 | 0.00 | T1 1.5R @ 181.16 |
| Target hit | 2023-08-25 15:20:00 | 179.50 | 181.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2023-08-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:45:00 | 180.95 | 180.33 | 0.00 | ORB-long ORB[179.35,180.80] vol=2.0x ATR=0.51 |
| Stop hit — per-position SL triggered | 2023-08-28 09:50:00 | 180.44 | 180.37 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 10:50:00 | 182.50 | 181.82 | 0.00 | ORB-long ORB[180.60,181.95] vol=2.2x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 10:55:00 | 183.15 | 181.93 | 0.00 | T1 1.5R @ 183.15 |
| Stop hit — per-position SL triggered | 2023-08-30 11:45:00 | 182.50 | 182.07 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 09:30:00 | 180.20 | 181.05 | 0.00 | ORB-short ORB[180.55,182.25] vol=2.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2023-08-31 09:35:00 | 180.70 | 180.97 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 11:00:00 | 187.85 | 187.19 | 0.00 | ORB-long ORB[186.10,187.50] vol=1.9x ATR=0.41 |
| Stop hit — per-position SL triggered | 2023-09-07 11:10:00 | 187.44 | 187.20 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:05:00 | 187.25 | 187.92 | 0.00 | ORB-short ORB[187.50,188.75] vol=3.5x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-09-08 11:35:00 | 187.79 | 187.59 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 09:35:00 | 188.85 | 187.78 | 0.00 | ORB-long ORB[187.00,188.20] vol=2.1x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 09:55:00 | 189.75 | 188.47 | 0.00 | T1 1.5R @ 189.75 |
| Target hit | 2023-09-11 12:20:00 | 189.70 | 189.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2023-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:35:00 | 189.25 | 190.20 | 0.00 | ORB-short ORB[189.85,192.00] vol=2.3x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:40:00 | 188.36 | 189.79 | 0.00 | T1 1.5R @ 188.36 |
| Target hit | 2023-09-12 15:20:00 | 180.85 | 184.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2023-09-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 09:35:00 | 179.75 | 181.47 | 0.00 | ORB-short ORB[180.75,182.80] vol=1.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 10:15:00 | 178.52 | 180.35 | 0.00 | T1 1.5R @ 178.52 |
| Stop hit — per-position SL triggered | 2023-09-13 11:00:00 | 179.75 | 179.82 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 10:10:00 | 184.70 | 183.29 | 0.00 | ORB-long ORB[182.05,183.60] vol=1.8x ATR=0.60 |
| Stop hit — per-position SL triggered | 2023-09-15 10:20:00 | 184.10 | 183.57 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 09:35:00 | 185.50 | 184.70 | 0.00 | ORB-long ORB[183.55,185.40] vol=1.9x ATR=0.75 |
| Stop hit — per-position SL triggered | 2023-09-18 09:55:00 | 184.75 | 185.05 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 09:30:00 | 183.20 | 182.29 | 0.00 | ORB-long ORB[181.00,182.50] vol=4.3x ATR=0.59 |
| Stop hit — per-position SL triggered | 2023-09-20 09:55:00 | 182.61 | 182.71 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 10:10:00 | 177.25 | 176.84 | 0.00 | ORB-long ORB[175.75,177.20] vol=2.0x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 10:35:00 | 177.90 | 177.10 | 0.00 | T1 1.5R @ 177.90 |
| Target hit | 2023-10-06 11:10:00 | 177.45 | 177.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2023-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:30:00 | 179.90 | 179.43 | 0.00 | ORB-long ORB[178.50,179.80] vol=1.6x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-10-11 09:40:00 | 179.45 | 179.45 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 10:50:00 | 180.70 | 179.74 | 0.00 | ORB-long ORB[178.85,180.15] vol=3.8x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 11:05:00 | 181.46 | 180.79 | 0.00 | T1 1.5R @ 181.46 |
| Target hit | 2023-10-12 11:25:00 | 181.00 | 181.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — BUY (started 2023-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 11:10:00 | 183.95 | 182.76 | 0.00 | ORB-long ORB[181.00,183.40] vol=3.5x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-10-17 11:50:00 | 183.41 | 183.14 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:45:00 | 181.65 | 182.81 | 0.00 | ORB-short ORB[182.55,183.60] vol=2.6x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 10:50:00 | 180.96 | 182.62 | 0.00 | T1 1.5R @ 180.96 |
| Stop hit — per-position SL triggered | 2023-10-18 10:55:00 | 181.65 | 182.57 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 11:15:00 | 182.80 | 180.41 | 0.00 | ORB-long ORB[177.65,180.30] vol=1.5x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 12:50:00 | 183.67 | 181.47 | 0.00 | T1 1.5R @ 183.67 |
| Target hit | 2023-10-19 15:20:00 | 183.95 | 182.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2023-10-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 09:50:00 | 180.75 | 181.80 | 0.00 | ORB-short ORB[181.00,183.00] vol=1.6x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:05:00 | 179.79 | 181.62 | 0.00 | T1 1.5R @ 179.79 |
| Target hit | 2023-10-23 15:20:00 | 172.10 | 176.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2023-10-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 10:05:00 | 173.25 | 174.19 | 0.00 | ORB-short ORB[173.35,175.30] vol=1.7x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 11:20:00 | 172.26 | 173.60 | 0.00 | T1 1.5R @ 172.26 |
| Stop hit — per-position SL triggered | 2023-10-31 13:05:00 | 173.25 | 173.03 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:00:00 | 175.80 | 174.45 | 0.00 | ORB-long ORB[173.15,174.50] vol=3.5x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-11-03 10:10:00 | 175.26 | 174.67 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 11:00:00 | 172.35 | 173.49 | 0.00 | ORB-short ORB[173.00,175.30] vol=1.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-11-06 11:25:00 | 173.00 | 173.44 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:35:00 | 176.15 | 175.35 | 0.00 | ORB-long ORB[174.65,175.90] vol=2.2x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 10:00:00 | 177.04 | 176.06 | 0.00 | T1 1.5R @ 177.04 |
| Stop hit — per-position SL triggered | 2023-11-08 10:10:00 | 176.15 | 176.08 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 11:05:00 | 175.90 | 174.86 | 0.00 | ORB-long ORB[174.45,175.50] vol=3.1x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-11-09 11:15:00 | 175.48 | 174.90 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-11-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 09:55:00 | 172.40 | 173.28 | 0.00 | ORB-short ORB[172.55,174.35] vol=1.9x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-11-10 10:05:00 | 172.92 | 173.15 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 09:55:00 | 175.80 | 175.38 | 0.00 | ORB-long ORB[174.65,175.55] vol=1.7x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 10:05:00 | 176.57 | 175.79 | 0.00 | T1 1.5R @ 176.57 |
| Stop hit — per-position SL triggered | 2023-11-13 10:25:00 | 175.80 | 175.93 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 10:35:00 | 180.85 | 179.45 | 0.00 | ORB-long ORB[178.75,180.30] vol=3.7x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 11:10:00 | 181.70 | 180.10 | 0.00 | T1 1.5R @ 181.70 |
| Target hit | 2023-11-15 15:20:00 | 181.65 | 181.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2023-11-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 10:50:00 | 172.30 | 171.24 | 0.00 | ORB-long ORB[170.25,171.80] vol=2.0x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-11-20 11:00:00 | 171.82 | 171.29 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:45:00 | 171.75 | 171.24 | 0.00 | ORB-long ORB[170.75,171.60] vol=1.7x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-11-29 09:55:00 | 171.29 | 171.26 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-11-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:50:00 | 168.85 | 169.20 | 0.00 | ORB-short ORB[169.10,170.20] vol=3.0x ATR=0.44 |
| Stop hit — per-position SL triggered | 2023-11-30 10:00:00 | 169.29 | 169.19 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 10:15:00 | 171.50 | 170.38 | 0.00 | ORB-long ORB[169.00,170.70] vol=3.4x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-12-01 11:00:00 | 170.87 | 170.56 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 10:05:00 | 172.75 | 171.60 | 0.00 | ORB-long ORB[171.00,172.00] vol=3.0x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-12-04 10:10:00 | 172.18 | 171.65 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 09:40:00 | 174.40 | 173.32 | 0.00 | ORB-long ORB[172.15,173.10] vol=2.9x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 09:50:00 | 175.15 | 174.31 | 0.00 | T1 1.5R @ 175.15 |
| Stop hit — per-position SL triggered | 2023-12-05 10:00:00 | 174.40 | 174.44 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:35:00 | 174.05 | 173.57 | 0.00 | ORB-long ORB[173.00,173.95] vol=2.8x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-12-06 09:50:00 | 173.48 | 173.56 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 164.85 | 166.37 | 0.00 | ORB-short ORB[166.30,168.40] vol=2.1x ATR=0.51 |
| Stop hit — per-position SL triggered | 2023-12-08 11:10:00 | 165.36 | 166.24 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 09:45:00 | 165.35 | 165.85 | 0.00 | ORB-short ORB[165.60,166.90] vol=2.1x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 10:10:00 | 164.74 | 165.58 | 0.00 | T1 1.5R @ 164.74 |
| Target hit | 2023-12-12 15:20:00 | 163.60 | 164.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2023-12-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 09:45:00 | 166.70 | 165.75 | 0.00 | ORB-long ORB[164.95,166.00] vol=5.4x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-12-14 09:50:00 | 166.21 | 165.81 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:35:00 | 164.75 | 165.15 | 0.00 | ORB-short ORB[164.80,166.20] vol=2.9x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 10:05:00 | 164.11 | 164.69 | 0.00 | T1 1.5R @ 164.11 |
| Stop hit — per-position SL triggered | 2023-12-19 11:40:00 | 164.75 | 164.58 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:45:00 | 161.05 | 160.47 | 0.00 | ORB-long ORB[159.80,160.95] vol=1.9x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-12-22 10:20:00 | 160.51 | 160.62 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2023-12-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:50:00 | 161.25 | 160.58 | 0.00 | ORB-long ORB[159.75,161.05] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-12-26 10:00:00 | 160.77 | 160.61 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:40:00 | 163.50 | 162.79 | 0.00 | ORB-long ORB[161.40,163.05] vol=4.5x ATR=0.51 |
| Stop hit — per-position SL triggered | 2023-12-27 10:00:00 | 162.99 | 162.90 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2023-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 09:30:00 | 164.40 | 163.52 | 0.00 | ORB-long ORB[162.60,163.90] vol=1.6x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 09:40:00 | 165.32 | 164.20 | 0.00 | T1 1.5R @ 165.32 |
| Target hit | 2023-12-29 13:35:00 | 165.15 | 165.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2024-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:45:00 | 167.50 | 166.52 | 0.00 | ORB-long ORB[165.40,167.00] vol=2.2x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-01-02 09:55:00 | 166.98 | 166.76 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 09:30:00 | 172.20 | 171.39 | 0.00 | ORB-long ORB[170.30,171.90] vol=1.8x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-01-03 09:35:00 | 171.44 | 171.38 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:35:00 | 176.90 | 176.09 | 0.00 | ORB-long ORB[174.45,176.65] vol=1.9x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 09:45:00 | 177.95 | 176.81 | 0.00 | T1 1.5R @ 177.95 |
| Stop hit — per-position SL triggered | 2024-01-05 09:55:00 | 176.90 | 176.97 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:30:00 | 176.05 | 176.48 | 0.00 | ORB-short ORB[176.25,177.45] vol=2.9x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 09:50:00 | 175.15 | 176.15 | 0.00 | T1 1.5R @ 175.15 |
| Stop hit — per-position SL triggered | 2024-01-09 10:00:00 | 176.05 | 176.00 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-01-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:30:00 | 176.45 | 175.68 | 0.00 | ORB-long ORB[174.05,176.10] vol=3.3x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 09:40:00 | 177.30 | 176.42 | 0.00 | T1 1.5R @ 177.30 |
| Stop hit — per-position SL triggered | 2024-01-11 09:55:00 | 176.45 | 176.54 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-01-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 09:35:00 | 181.50 | 180.86 | 0.00 | ORB-long ORB[180.00,180.90] vol=2.9x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-01-15 09:45:00 | 180.91 | 180.77 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 09:30:00 | 179.75 | 179.16 | 0.00 | ORB-long ORB[178.10,179.70] vol=2.0x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-01-16 09:35:00 | 179.24 | 179.20 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 173.30 | 174.52 | 0.00 | ORB-short ORB[174.55,175.90] vol=1.7x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:45:00 | 172.20 | 173.71 | 0.00 | T1 1.5R @ 172.20 |
| Stop hit — per-position SL triggered | 2024-01-18 10:05:00 | 173.30 | 173.13 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:35:00 | 173.45 | 174.12 | 0.00 | ORB-short ORB[173.50,175.30] vol=3.4x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:50:00 | 172.54 | 173.46 | 0.00 | T1 1.5R @ 172.54 |
| Target hit | 2024-01-23 15:20:00 | 163.75 | 167.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2024-01-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:20:00 | 168.60 | 167.69 | 0.00 | ORB-long ORB[165.80,167.75] vol=1.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-01-29 10:35:00 | 168.02 | 167.76 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-01-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:00:00 | 169.70 | 168.88 | 0.00 | ORB-long ORB[168.30,169.55] vol=1.7x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 12:05:00 | 170.47 | 169.57 | 0.00 | T1 1.5R @ 170.47 |
| Stop hit — per-position SL triggered | 2024-01-30 14:25:00 | 169.70 | 169.97 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 11:15:00 | 170.15 | 169.36 | 0.00 | ORB-long ORB[168.35,169.60] vol=1.7x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 11:45:00 | 170.76 | 169.54 | 0.00 | T1 1.5R @ 170.76 |
| Stop hit — per-position SL triggered | 2024-01-31 12:25:00 | 170.15 | 169.78 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-02-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 09:35:00 | 191.35 | 190.07 | 0.00 | ORB-long ORB[188.35,190.95] vol=2.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-02-08 09:45:00 | 190.46 | 190.14 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:35:00 | 186.25 | 188.08 | 0.00 | ORB-short ORB[186.55,189.10] vol=1.5x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:45:00 | 184.89 | 187.35 | 0.00 | T1 1.5R @ 184.89 |
| Target hit | 2024-02-09 12:05:00 | 185.20 | 185.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 86 — SELL (started 2024-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 09:40:00 | 186.15 | 187.47 | 0.00 | ORB-short ORB[186.95,189.35] vol=1.7x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 10:00:00 | 185.00 | 186.76 | 0.00 | T1 1.5R @ 185.00 |
| Target hit | 2024-02-12 15:20:00 | 180.15 | 183.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — SELL (started 2024-02-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 09:55:00 | 181.65 | 182.73 | 0.00 | ORB-short ORB[182.15,184.00] vol=1.5x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-02-15 10:10:00 | 182.27 | 182.65 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 09:30:00 | 183.30 | 182.83 | 0.00 | ORB-long ORB[181.75,183.15] vol=2.1x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-02-16 09:40:00 | 182.81 | 183.01 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-02-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 09:30:00 | 180.75 | 182.10 | 0.00 | ORB-short ORB[181.85,183.80] vol=2.4x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-02-22 09:35:00 | 181.56 | 182.00 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 09:40:00 | 184.15 | 184.87 | 0.00 | ORB-short ORB[184.85,185.95] vol=2.6x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 10:05:00 | 183.27 | 184.17 | 0.00 | T1 1.5R @ 183.27 |
| Stop hit — per-position SL triggered | 2024-02-23 10:25:00 | 184.15 | 184.08 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 09:30:00 | 190.30 | 189.31 | 0.00 | ORB-long ORB[187.40,190.00] vol=1.9x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 09:35:00 | 191.49 | 189.70 | 0.00 | T1 1.5R @ 191.49 |
| Stop hit — per-position SL triggered | 2024-02-26 09:40:00 | 190.30 | 189.86 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-02-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 10:35:00 | 191.80 | 190.49 | 0.00 | ORB-long ORB[189.15,191.00] vol=2.1x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 10:55:00 | 192.68 | 191.10 | 0.00 | T1 1.5R @ 192.68 |
| Stop hit — per-position SL triggered | 2024-02-27 11:15:00 | 191.80 | 191.36 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-02-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:45:00 | 186.80 | 187.95 | 0.00 | ORB-short ORB[187.45,188.50] vol=1.8x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:55:00 | 185.92 | 187.61 | 0.00 | T1 1.5R @ 185.92 |
| Target hit | 2024-02-28 15:20:00 | 181.30 | 185.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 94 — SELL (started 2024-02-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:50:00 | 180.00 | 181.20 | 0.00 | ORB-short ORB[180.05,182.60] vol=2.3x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-02-29 11:00:00 | 180.69 | 181.14 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-03-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 09:40:00 | 185.85 | 185.09 | 0.00 | ORB-long ORB[183.00,185.60] vol=1.9x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-03-01 10:00:00 | 185.14 | 185.16 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 09:30:00 | 188.50 | 187.84 | 0.00 | ORB-long ORB[186.60,188.30] vol=5.6x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 09:35:00 | 189.56 | 188.19 | 0.00 | T1 1.5R @ 189.56 |
| Target hit | 2024-03-04 10:50:00 | 191.70 | 191.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 97 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 11:15:00 | 180.35 | 182.25 | 0.00 | ORB-short ORB[182.25,184.85] vol=2.3x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 11:55:00 | 179.50 | 181.88 | 0.00 | T1 1.5R @ 179.50 |
| Stop hit — per-position SL triggered | 2024-03-11 12:30:00 | 180.35 | 181.75 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 09:30:00 | 171.45 | 173.05 | 0.00 | ORB-short ORB[171.85,174.40] vol=2.4x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-03-18 09:35:00 | 172.37 | 172.84 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2024-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:50:00 | 172.50 | 173.46 | 0.00 | ORB-short ORB[172.90,174.35] vol=1.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 10:20:00 | 171.56 | 173.02 | 0.00 | T1 1.5R @ 171.56 |
| Stop hit — per-position SL triggered | 2024-03-19 12:35:00 | 172.50 | 172.28 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 09:45:00 | 167.95 | 169.53 | 0.00 | ORB-short ORB[169.20,171.55] vol=2.1x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 09:55:00 | 166.82 | 169.12 | 0.00 | T1 1.5R @ 166.82 |
| Stop hit — per-position SL triggered | 2024-03-20 10:55:00 | 167.95 | 168.23 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-03-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:20:00 | 173.30 | 172.66 | 0.00 | ORB-long ORB[170.80,172.65] vol=2.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-03-21 10:30:00 | 172.65 | 172.68 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-03-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 11:05:00 | 177.05 | 175.89 | 0.00 | ORB-long ORB[174.55,175.90] vol=2.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-03-26 12:00:00 | 176.55 | 176.12 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2024-04-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:00:00 | 190.95 | 187.80 | 0.00 | ORB-long ORB[186.00,188.35] vol=3.8x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 10:10:00 | 192.75 | 189.38 | 0.00 | T1 1.5R @ 192.75 |
| Target hit | 2024-04-02 15:20:00 | 200.55 | 196.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 104 — BUY (started 2024-04-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 09:55:00 | 206.55 | 204.95 | 0.00 | ORB-long ORB[203.15,205.95] vol=1.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-04-18 10:05:00 | 205.57 | 205.08 | 0.00 | SL hit |

### Cycle 105 — SELL (started 2024-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 09:40:00 | 201.45 | 202.94 | 0.00 | ORB-short ORB[201.65,203.85] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-04-22 09:55:00 | 202.42 | 202.56 | 0.00 | SL hit |

### Cycle 106 — SELL (started 2024-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 11:00:00 | 222.95 | 224.18 | 0.00 | ORB-short ORB[224.50,226.70] vol=2.1x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 221.44 | 223.74 | 0.00 | T1 1.5R @ 221.44 |
| Target hit | 2024-05-07 13:30:00 | 219.45 | 219.45 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-16 10:35:00 | 168.90 | 2023-05-16 10:50:00 | 168.33 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-05-18 09:30:00 | 167.15 | 2023-05-18 09:35:00 | 166.57 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-05-26 09:30:00 | 166.60 | 2023-05-26 09:40:00 | 165.98 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-05-30 10:05:00 | 167.45 | 2023-05-30 10:45:00 | 168.26 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-05-30 10:05:00 | 167.45 | 2023-05-30 15:20:00 | 171.60 | TARGET_HIT | 0.50 | 2.48% |
| SELL | retest1 | 2023-05-31 10:25:00 | 169.95 | 2023-05-31 10:40:00 | 169.21 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-05-31 10:25:00 | 169.95 | 2023-05-31 10:50:00 | 169.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-05 09:55:00 | 170.35 | 2023-06-05 10:00:00 | 169.41 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-06-05 09:55:00 | 170.35 | 2023-06-05 13:15:00 | 169.50 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2023-06-09 09:45:00 | 174.65 | 2023-06-09 09:50:00 | 174.07 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-06-16 10:25:00 | 177.90 | 2023-06-16 10:45:00 | 177.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-19 09:30:00 | 180.20 | 2023-06-19 09:45:00 | 181.16 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-06-19 09:30:00 | 180.20 | 2023-06-19 09:55:00 | 180.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-20 09:30:00 | 176.85 | 2023-06-20 09:35:00 | 177.46 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-06-22 09:40:00 | 176.85 | 2023-06-22 09:45:00 | 177.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-06 09:55:00 | 188.30 | 2023-07-06 11:20:00 | 188.87 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-07-07 11:10:00 | 191.75 | 2023-07-07 11:25:00 | 191.06 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-07-11 10:30:00 | 184.95 | 2023-07-11 11:20:00 | 185.57 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-07-12 10:05:00 | 186.20 | 2023-07-12 10:15:00 | 187.11 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-07-12 10:05:00 | 186.20 | 2023-07-12 10:20:00 | 186.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-17 11:00:00 | 188.40 | 2023-07-17 11:10:00 | 187.67 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-07-17 11:00:00 | 188.40 | 2023-07-17 13:45:00 | 186.80 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2023-07-21 09:30:00 | 186.60 | 2023-07-21 09:45:00 | 186.02 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-24 09:35:00 | 187.95 | 2023-07-24 09:40:00 | 187.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-07-26 09:45:00 | 190.45 | 2023-07-26 10:00:00 | 191.24 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-07-26 09:45:00 | 190.45 | 2023-07-26 10:10:00 | 190.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-28 09:50:00 | 194.75 | 2023-07-28 10:30:00 | 193.97 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-07-31 09:40:00 | 194.35 | 2023-07-31 10:00:00 | 194.99 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-08-01 09:40:00 | 195.70 | 2023-08-01 10:10:00 | 195.05 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-08-01 09:40:00 | 195.70 | 2023-08-01 10:55:00 | 195.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-02 10:25:00 | 192.65 | 2023-08-02 11:05:00 | 191.61 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2023-08-02 10:25:00 | 192.65 | 2023-08-02 14:35:00 | 191.00 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2023-08-08 09:45:00 | 189.45 | 2023-08-08 10:00:00 | 188.77 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-08-10 10:25:00 | 184.60 | 2023-08-10 10:40:00 | 185.22 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-08-11 09:30:00 | 186.25 | 2023-08-11 09:35:00 | 187.15 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-08-11 09:30:00 | 186.25 | 2023-08-11 10:10:00 | 186.50 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2023-08-18 10:05:00 | 179.35 | 2023-08-18 10:15:00 | 179.84 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-08-22 09:30:00 | 184.90 | 2023-08-22 10:15:00 | 185.80 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-08-22 09:30:00 | 184.90 | 2023-08-22 10:30:00 | 184.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-24 11:15:00 | 185.95 | 2023-08-24 11:45:00 | 186.36 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-25 10:10:00 | 181.90 | 2023-08-25 10:20:00 | 181.16 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-08-25 10:10:00 | 181.90 | 2023-08-25 15:20:00 | 179.50 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2023-08-28 09:45:00 | 180.95 | 2023-08-28 09:50:00 | 180.44 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-30 10:50:00 | 182.50 | 2023-08-30 10:55:00 | 183.15 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-08-30 10:50:00 | 182.50 | 2023-08-30 11:45:00 | 182.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-31 09:30:00 | 180.20 | 2023-08-31 09:35:00 | 180.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-07 11:00:00 | 187.85 | 2023-09-07 11:10:00 | 187.44 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-08 10:05:00 | 187.25 | 2023-09-08 11:35:00 | 187.79 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-11 09:35:00 | 188.85 | 2023-09-11 09:55:00 | 189.75 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-09-11 09:35:00 | 188.85 | 2023-09-11 12:20:00 | 189.70 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2023-09-12 09:35:00 | 189.25 | 2023-09-12 09:40:00 | 188.36 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-09-12 09:35:00 | 189.25 | 2023-09-12 15:20:00 | 180.85 | TARGET_HIT | 0.50 | 4.44% |
| SELL | retest1 | 2023-09-13 09:35:00 | 179.75 | 2023-09-13 10:15:00 | 178.52 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2023-09-13 09:35:00 | 179.75 | 2023-09-13 11:00:00 | 179.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-15 10:10:00 | 184.70 | 2023-09-15 10:20:00 | 184.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-09-18 09:35:00 | 185.50 | 2023-09-18 09:55:00 | 184.75 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-09-20 09:30:00 | 183.20 | 2023-09-20 09:55:00 | 182.61 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-10-06 10:10:00 | 177.25 | 2023-10-06 10:35:00 | 177.90 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-10-06 10:10:00 | 177.25 | 2023-10-06 11:10:00 | 177.45 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2023-10-11 09:30:00 | 179.90 | 2023-10-11 09:40:00 | 179.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-12 10:50:00 | 180.70 | 2023-10-12 11:05:00 | 181.46 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-10-12 10:50:00 | 180.70 | 2023-10-12 11:25:00 | 181.00 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2023-10-17 11:10:00 | 183.95 | 2023-10-17 11:50:00 | 183.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-18 10:45:00 | 181.65 | 2023-10-18 10:50:00 | 180.96 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-10-18 10:45:00 | 181.65 | 2023-10-18 10:55:00 | 181.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-19 11:15:00 | 182.80 | 2023-10-19 12:50:00 | 183.67 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-10-19 11:15:00 | 182.80 | 2023-10-19 15:20:00 | 183.95 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2023-10-23 09:50:00 | 180.75 | 2023-10-23 10:05:00 | 179.79 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-10-23 09:50:00 | 180.75 | 2023-10-23 15:20:00 | 172.10 | TARGET_HIT | 0.50 | 4.79% |
| SELL | retest1 | 2023-10-31 10:05:00 | 173.25 | 2023-10-31 11:20:00 | 172.26 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2023-10-31 10:05:00 | 173.25 | 2023-10-31 13:05:00 | 173.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-03 10:00:00 | 175.80 | 2023-11-03 10:10:00 | 175.26 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-11-06 11:00:00 | 172.35 | 2023-11-06 11:25:00 | 173.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-11-08 09:35:00 | 176.15 | 2023-11-08 10:00:00 | 177.04 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-11-08 09:35:00 | 176.15 | 2023-11-08 10:10:00 | 176.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-09 11:05:00 | 175.90 | 2023-11-09 11:15:00 | 175.48 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-11-10 09:55:00 | 172.40 | 2023-11-10 10:05:00 | 172.92 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-11-13 09:55:00 | 175.80 | 2023-11-13 10:05:00 | 176.57 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-11-13 09:55:00 | 175.80 | 2023-11-13 10:25:00 | 175.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-15 10:35:00 | 180.85 | 2023-11-15 11:10:00 | 181.70 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-11-15 10:35:00 | 180.85 | 2023-11-15 15:20:00 | 181.65 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2023-11-20 10:50:00 | 172.30 | 2023-11-20 11:00:00 | 171.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-29 09:45:00 | 171.75 | 2023-11-29 09:55:00 | 171.29 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-11-30 09:50:00 | 168.85 | 2023-11-30 10:00:00 | 169.29 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-01 10:15:00 | 171.50 | 2023-12-01 11:00:00 | 170.87 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-04 10:05:00 | 172.75 | 2023-12-04 10:10:00 | 172.18 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-05 09:40:00 | 174.40 | 2023-12-05 09:50:00 | 175.15 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-12-05 09:40:00 | 174.40 | 2023-12-05 10:00:00 | 174.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-06 09:35:00 | 174.05 | 2023-12-06 09:50:00 | 173.48 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-08 11:00:00 | 164.85 | 2023-12-08 11:10:00 | 165.36 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-12-12 09:45:00 | 165.35 | 2023-12-12 10:10:00 | 164.74 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-12-12 09:45:00 | 165.35 | 2023-12-12 15:20:00 | 163.60 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2023-12-14 09:45:00 | 166.70 | 2023-12-14 09:50:00 | 166.21 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-12-19 09:35:00 | 164.75 | 2023-12-19 10:05:00 | 164.11 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-12-19 09:35:00 | 164.75 | 2023-12-19 11:40:00 | 164.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-22 09:45:00 | 161.05 | 2023-12-22 10:20:00 | 160.51 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-26 09:50:00 | 161.25 | 2023-12-26 10:00:00 | 160.77 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-27 09:40:00 | 163.50 | 2023-12-27 10:00:00 | 162.99 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-12-29 09:30:00 | 164.40 | 2023-12-29 09:40:00 | 165.32 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-12-29 09:30:00 | 164.40 | 2023-12-29 13:35:00 | 165.15 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2024-01-02 09:45:00 | 167.50 | 2024-01-02 09:55:00 | 166.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-01-03 09:30:00 | 172.20 | 2024-01-03 09:35:00 | 171.44 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-01-05 09:35:00 | 176.90 | 2024-01-05 09:45:00 | 177.95 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-01-05 09:35:00 | 176.90 | 2024-01-05 09:55:00 | 176.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-09 09:30:00 | 176.05 | 2024-01-09 09:50:00 | 175.15 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-01-09 09:30:00 | 176.05 | 2024-01-09 10:00:00 | 176.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-11 09:30:00 | 176.45 | 2024-01-11 09:40:00 | 177.30 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-01-11 09:30:00 | 176.45 | 2024-01-11 09:55:00 | 176.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-15 09:35:00 | 181.50 | 2024-01-15 09:45:00 | 180.91 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-01-16 09:30:00 | 179.75 | 2024-01-16 09:35:00 | 179.24 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-18 09:35:00 | 173.30 | 2024-01-18 09:45:00 | 172.20 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-01-18 09:35:00 | 173.30 | 2024-01-18 10:05:00 | 173.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-23 09:35:00 | 173.45 | 2024-01-23 09:50:00 | 172.54 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-01-23 09:35:00 | 173.45 | 2024-01-23 15:20:00 | 163.75 | TARGET_HIT | 0.50 | 5.59% |
| BUY | retest1 | 2024-01-29 10:20:00 | 168.60 | 2024-01-29 10:35:00 | 168.02 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-01-30 10:00:00 | 169.70 | 2024-01-30 12:05:00 | 170.47 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-01-30 10:00:00 | 169.70 | 2024-01-30 14:25:00 | 169.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-31 11:15:00 | 170.15 | 2024-01-31 11:45:00 | 170.76 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-01-31 11:15:00 | 170.15 | 2024-01-31 12:25:00 | 170.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-08 09:35:00 | 191.35 | 2024-02-08 09:45:00 | 190.46 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-02-09 09:35:00 | 186.25 | 2024-02-09 09:45:00 | 184.89 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-02-09 09:35:00 | 186.25 | 2024-02-09 12:05:00 | 185.20 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2024-02-12 09:40:00 | 186.15 | 2024-02-12 10:00:00 | 185.00 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-02-12 09:40:00 | 186.15 | 2024-02-12 15:20:00 | 180.15 | TARGET_HIT | 0.50 | 3.22% |
| SELL | retest1 | 2024-02-15 09:55:00 | 181.65 | 2024-02-15 10:10:00 | 182.27 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-02-16 09:30:00 | 183.30 | 2024-02-16 09:40:00 | 182.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-22 09:30:00 | 180.75 | 2024-02-22 09:35:00 | 181.56 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-02-23 09:40:00 | 184.15 | 2024-02-23 10:05:00 | 183.27 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-02-23 09:40:00 | 184.15 | 2024-02-23 10:25:00 | 184.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-26 09:30:00 | 190.30 | 2024-02-26 09:35:00 | 191.49 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-02-26 09:30:00 | 190.30 | 2024-02-26 09:40:00 | 190.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-27 10:35:00 | 191.80 | 2024-02-27 10:55:00 | 192.68 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-02-27 10:35:00 | 191.80 | 2024-02-27 11:15:00 | 191.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-28 10:45:00 | 186.80 | 2024-02-28 10:55:00 | 185.92 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-02-28 10:45:00 | 186.80 | 2024-02-28 15:20:00 | 181.30 | TARGET_HIT | 0.50 | 2.94% |
| SELL | retest1 | 2024-02-29 10:50:00 | 180.00 | 2024-02-29 11:00:00 | 180.69 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-03-01 09:40:00 | 185.85 | 2024-03-01 10:00:00 | 185.14 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-03-04 09:30:00 | 188.50 | 2024-03-04 09:35:00 | 189.56 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-03-04 09:30:00 | 188.50 | 2024-03-04 10:50:00 | 191.70 | TARGET_HIT | 0.50 | 1.70% |
| SELL | retest1 | 2024-03-11 11:15:00 | 180.35 | 2024-03-11 11:55:00 | 179.50 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-03-11 11:15:00 | 180.35 | 2024-03-11 12:30:00 | 180.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-18 09:30:00 | 171.45 | 2024-03-18 09:35:00 | 172.37 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-03-19 09:50:00 | 172.50 | 2024-03-19 10:20:00 | 171.56 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-03-19 09:50:00 | 172.50 | 2024-03-19 12:35:00 | 172.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-20 09:45:00 | 167.95 | 2024-03-20 09:55:00 | 166.82 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-03-20 09:45:00 | 167.95 | 2024-03-20 10:55:00 | 167.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 10:20:00 | 173.30 | 2024-03-21 10:30:00 | 172.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-03-26 11:05:00 | 177.05 | 2024-03-26 12:00:00 | 176.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-02 10:00:00 | 190.95 | 2024-04-02 10:10:00 | 192.75 | PARTIAL | 0.50 | 0.94% |
| BUY | retest1 | 2024-04-02 10:00:00 | 190.95 | 2024-04-02 15:20:00 | 200.55 | TARGET_HIT | 0.50 | 5.03% |
| BUY | retest1 | 2024-04-18 09:55:00 | 206.55 | 2024-04-18 10:05:00 | 205.57 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-04-22 09:40:00 | 201.45 | 2024-04-22 09:55:00 | 202.42 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-05-07 11:00:00 | 222.95 | 2024-05-07 11:15:00 | 221.44 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-05-07 11:00:00 | 222.95 | 2024-05-07 13:30:00 | 219.45 | TARGET_HIT | 0.50 | 1.57% |
