# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (51156 bars)
- **Last close:** 290.00
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 15 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 48
- **Target hits / Stop hits / Partials:** 15 / 48 / 26
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 24.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 22 | 48.9% | 7 | 23 | 15 | 0.19% | 8.6% |
| BUY @ 2nd Alert (retest1) | 45 | 22 | 48.9% | 7 | 23 | 15 | 0.19% | 8.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 19 | 43.2% | 8 | 25 | 11 | 0.35% | 15.4% |
| SELL @ 2nd Alert (retest1) | 44 | 19 | 43.2% | 8 | 25 | 11 | 0.35% | 15.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 89 | 41 | 46.1% | 15 | 48 | 26 | 0.27% | 24.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:15:00 | 190.50 | 191.44 | 0.00 | ORB-short ORB[191.70,193.40] vol=2.5x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 10:20:00 | 189.26 | 191.37 | 0.00 | T1 1.5R @ 189.26 |
| Target hit | 2023-05-17 15:20:00 | 187.30 | 188.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2023-05-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 11:10:00 | 167.90 | 168.98 | 0.00 | ORB-short ORB[168.10,170.45] vol=1.5x ATR=0.86 |
| Stop hit — per-position SL triggered | 2023-05-25 11:30:00 | 168.76 | 168.88 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:55:00 | 175.15 | 173.46 | 0.00 | ORB-long ORB[171.25,173.45] vol=5.9x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 11:00:00 | 176.06 | 173.89 | 0.00 | T1 1.5R @ 176.06 |
| Target hit | 2023-05-26 15:20:00 | 178.80 | 177.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2023-06-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:35:00 | 186.60 | 185.44 | 0.00 | ORB-long ORB[183.70,185.90] vol=1.5x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 09:45:00 | 187.41 | 185.96 | 0.00 | T1 1.5R @ 187.41 |
| Stop hit — per-position SL triggered | 2023-06-02 09:50:00 | 186.60 | 186.09 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:50:00 | 181.35 | 182.40 | 0.00 | ORB-short ORB[181.50,183.80] vol=2.0x ATR=0.72 |
| Stop hit — per-position SL triggered | 2023-06-09 10:00:00 | 182.07 | 182.20 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 10:05:00 | 181.00 | 181.88 | 0.00 | ORB-short ORB[181.30,182.65] vol=1.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-06-13 10:25:00 | 181.62 | 181.74 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 09:45:00 | 193.40 | 194.61 | 0.00 | ORB-short ORB[193.50,196.40] vol=2.3x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-06-16 09:50:00 | 194.38 | 194.60 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 10:55:00 | 192.95 | 194.70 | 0.00 | ORB-short ORB[193.55,195.95] vol=1.7x ATR=0.75 |
| Stop hit — per-position SL triggered | 2023-06-20 13:10:00 | 193.70 | 194.28 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 10:15:00 | 197.90 | 195.63 | 0.00 | ORB-long ORB[194.00,195.10] vol=4.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-06-21 10:20:00 | 197.03 | 196.21 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:30:00 | 201.80 | 200.05 | 0.00 | ORB-long ORB[198.10,200.85] vol=2.7x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 09:40:00 | 203.29 | 201.51 | 0.00 | T1 1.5R @ 203.29 |
| Target hit | 2023-06-22 10:00:00 | 202.00 | 203.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2023-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 09:40:00 | 211.15 | 208.76 | 0.00 | ORB-long ORB[206.20,208.55] vol=4.9x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 09:55:00 | 212.35 | 210.36 | 0.00 | T1 1.5R @ 212.35 |
| Target hit | 2023-07-10 10:55:00 | 212.15 | 212.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2023-07-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 10:40:00 | 217.95 | 219.53 | 0.00 | ORB-short ORB[219.65,222.20] vol=2.0x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-07-19 10:55:00 | 218.75 | 219.29 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 11:15:00 | 217.25 | 218.15 | 0.00 | ORB-short ORB[217.65,220.30] vol=2.3x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-07-20 11:25:00 | 217.70 | 218.14 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 09:50:00 | 216.85 | 215.80 | 0.00 | ORB-long ORB[214.15,216.40] vol=2.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2023-07-21 09:55:00 | 216.12 | 215.87 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 10:00:00 | 218.45 | 216.95 | 0.00 | ORB-long ORB[215.60,217.90] vol=2.1x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 10:10:00 | 219.58 | 217.53 | 0.00 | T1 1.5R @ 219.58 |
| Stop hit — per-position SL triggered | 2023-07-25 10:25:00 | 218.45 | 218.62 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:50:00 | 216.95 | 215.43 | 0.00 | ORB-long ORB[214.30,216.40] vol=1.9x ATR=0.55 |
| Stop hit — per-position SL triggered | 2023-07-26 10:55:00 | 216.40 | 215.50 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 10:40:00 | 216.65 | 214.18 | 0.00 | ORB-long ORB[213.15,214.90] vol=2.3x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-07-27 10:45:00 | 215.84 | 214.46 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 11:00:00 | 213.00 | 213.96 | 0.00 | ORB-short ORB[213.50,215.10] vol=2.1x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-07-28 11:45:00 | 213.52 | 213.88 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:45:00 | 249.90 | 250.37 | 0.00 | ORB-short ORB[250.00,253.70] vol=3.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 12:30:00 | 248.57 | 250.25 | 0.00 | T1 1.5R @ 248.57 |
| Stop hit — per-position SL triggered | 2023-08-11 15:05:00 | 249.90 | 250.12 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:35:00 | 248.70 | 246.85 | 0.00 | ORB-long ORB[244.90,247.55] vol=2.8x ATR=0.89 |
| Stop hit — per-position SL triggered | 2023-08-17 09:40:00 | 247.81 | 247.17 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 10:50:00 | 258.60 | 256.37 | 0.00 | ORB-long ORB[254.80,258.50] vol=3.5x ATR=1.05 |
| Stop hit — per-position SL triggered | 2023-08-18 10:55:00 | 257.55 | 256.54 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 10:20:00 | 275.05 | 277.15 | 0.00 | ORB-short ORB[279.15,282.80] vol=2.5x ATR=1.24 |
| Stop hit — per-position SL triggered | 2023-08-23 10:35:00 | 276.29 | 276.96 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:40:00 | 292.60 | 289.74 | 0.00 | ORB-long ORB[284.60,287.90] vol=3.4x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 09:45:00 | 294.14 | 291.53 | 0.00 | T1 1.5R @ 294.14 |
| Stop hit — per-position SL triggered | 2023-09-05 09:55:00 | 292.60 | 292.43 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-09-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 10:40:00 | 296.45 | 293.72 | 0.00 | ORB-long ORB[291.00,294.50] vol=5.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 10:45:00 | 298.17 | 294.81 | 0.00 | T1 1.5R @ 298.17 |
| Stop hit — per-position SL triggered | 2023-09-07 12:55:00 | 296.45 | 296.49 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 290.25 | 291.48 | 0.00 | ORB-short ORB[291.10,295.40] vol=4.6x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:40:00 | 288.82 | 288.57 | 0.00 | T1 1.5R @ 288.82 |
| Target hit | 2023-09-12 10:05:00 | 286.05 | 285.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — SELL (started 2023-09-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-18 10:55:00 | 272.65 | 274.72 | 0.00 | ORB-short ORB[273.65,277.00] vol=2.0x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-09-18 12:15:00 | 273.74 | 273.66 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-09-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 09:30:00 | 271.95 | 269.94 | 0.00 | ORB-long ORB[268.00,270.90] vol=2.2x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-09-21 09:55:00 | 270.73 | 270.23 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-10-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 11:00:00 | 246.10 | 244.36 | 0.00 | ORB-long ORB[242.10,245.55] vol=2.1x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 11:05:00 | 247.47 | 245.26 | 0.00 | T1 1.5R @ 247.47 |
| Stop hit — per-position SL triggered | 2023-10-05 11:50:00 | 246.10 | 245.53 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-10-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 09:35:00 | 247.55 | 245.67 | 0.00 | ORB-long ORB[243.35,246.45] vol=2.5x ATR=1.05 |
| Stop hit — per-position SL triggered | 2023-10-06 09:55:00 | 246.50 | 245.95 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-10-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:50:00 | 252.85 | 250.99 | 0.00 | ORB-long ORB[247.45,250.60] vol=2.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2023-10-11 10:00:00 | 251.78 | 251.18 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-10-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:00:00 | 246.70 | 248.21 | 0.00 | ORB-short ORB[247.00,250.60] vol=1.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 10:45:00 | 245.16 | 247.63 | 0.00 | T1 1.5R @ 245.16 |
| Stop hit — per-position SL triggered | 2023-10-13 11:15:00 | 246.70 | 247.42 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-10-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 09:40:00 | 260.50 | 258.63 | 0.00 | ORB-long ORB[256.05,259.30] vol=1.5x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-10-20 10:05:00 | 259.38 | 259.01 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-10-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 10:00:00 | 245.35 | 247.43 | 0.00 | ORB-short ORB[246.10,249.70] vol=1.5x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:00:00 | 243.30 | 245.40 | 0.00 | T1 1.5R @ 243.30 |
| Target hit | 2023-10-25 15:20:00 | 234.60 | 240.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2023-11-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 11:05:00 | 275.10 | 275.87 | 0.00 | ORB-short ORB[275.30,278.60] vol=3.4x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-11-08 11:25:00 | 275.97 | 275.84 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-11-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 09:50:00 | 269.40 | 270.75 | 0.00 | ORB-short ORB[271.00,274.65] vol=3.6x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-11-09 10:00:00 | 270.48 | 270.38 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-17 10:15:00 | 267.55 | 269.53 | 0.00 | ORB-short ORB[269.80,271.70] vol=1.6x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 12:10:00 | 265.90 | 268.43 | 0.00 | T1 1.5R @ 265.90 |
| Stop hit — per-position SL triggered | 2023-11-17 14:55:00 | 267.55 | 266.89 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 11:15:00 | 262.00 | 262.90 | 0.00 | ORB-short ORB[263.75,267.20] vol=2.0x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-11-22 11:35:00 | 262.71 | 262.87 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-11-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 11:10:00 | 264.90 | 267.56 | 0.00 | ORB-short ORB[266.50,269.70] vol=2.1x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 12:15:00 | 263.62 | 266.82 | 0.00 | T1 1.5R @ 263.62 |
| Target hit | 2023-11-23 15:20:00 | 262.85 | 265.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2023-11-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:25:00 | 262.80 | 263.82 | 0.00 | ORB-short ORB[263.00,265.85] vol=2.0x ATR=0.76 |
| Stop hit — per-position SL triggered | 2023-11-24 11:30:00 | 263.56 | 263.70 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 10:55:00 | 259.15 | 261.44 | 0.00 | ORB-short ORB[261.80,263.60] vol=1.5x ATR=0.77 |
| Stop hit — per-position SL triggered | 2023-11-29 11:15:00 | 259.92 | 261.29 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:35:00 | 277.40 | 273.00 | 0.00 | ORB-long ORB[269.00,272.25] vol=10.5x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 09:45:00 | 280.12 | 275.44 | 0.00 | T1 1.5R @ 280.12 |
| Target hit | 2023-12-04 15:20:00 | 281.90 | 280.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2023-12-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 10:25:00 | 286.80 | 284.55 | 0.00 | ORB-long ORB[281.15,284.50] vol=4.1x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-12-08 11:00:00 | 285.95 | 285.38 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-12-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:45:00 | 310.40 | 308.41 | 0.00 | ORB-long ORB[304.90,308.80] vol=2.2x ATR=1.61 |
| Stop hit — per-position SL triggered | 2023-12-22 10:40:00 | 308.79 | 309.43 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-12-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:40:00 | 306.95 | 308.92 | 0.00 | ORB-short ORB[309.50,312.70] vol=2.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-12-27 10:50:00 | 308.04 | 308.87 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-12-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:55:00 | 311.85 | 309.50 | 0.00 | ORB-long ORB[307.95,310.70] vol=1.9x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 10:00:00 | 313.66 | 311.57 | 0.00 | T1 1.5R @ 313.66 |
| Target hit | 2023-12-28 10:20:00 | 312.50 | 313.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2024-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:35:00 | 322.95 | 321.82 | 0.00 | ORB-long ORB[318.60,321.75] vol=3.0x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 09:40:00 | 325.06 | 322.50 | 0.00 | T1 1.5R @ 325.06 |
| Target hit | 2024-01-01 11:00:00 | 324.50 | 324.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2024-01-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 10:10:00 | 326.30 | 323.76 | 0.00 | ORB-long ORB[320.75,324.70] vol=1.7x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-01-03 10:20:00 | 324.58 | 323.99 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-01-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:55:00 | 333.20 | 335.96 | 0.00 | ORB-short ORB[337.50,340.60] vol=1.8x ATR=1.30 |
| Stop hit — per-position SL triggered | 2024-01-05 11:20:00 | 334.50 | 335.78 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 10:55:00 | 344.25 | 338.83 | 0.00 | ORB-long ORB[337.05,341.20] vol=2.8x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-01-08 11:00:00 | 342.56 | 340.62 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-01-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 10:00:00 | 395.30 | 392.23 | 0.00 | ORB-long ORB[387.30,393.20] vol=1.5x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 10:25:00 | 397.79 | 394.08 | 0.00 | T1 1.5R @ 397.79 |
| Stop hit — per-position SL triggered | 2024-01-19 10:45:00 | 395.30 | 394.66 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-01-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:45:00 | 399.00 | 395.96 | 0.00 | ORB-long ORB[392.65,397.25] vol=10.5x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 11:30:00 | 401.61 | 397.74 | 0.00 | T1 1.5R @ 401.61 |
| Target hit | 2024-01-29 15:20:00 | 401.95 | 401.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2024-01-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 09:50:00 | 409.95 | 407.56 | 0.00 | ORB-long ORB[403.25,409.00] vol=2.3x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 10:00:00 | 412.71 | 408.82 | 0.00 | T1 1.5R @ 412.71 |
| Stop hit — per-position SL triggered | 2024-01-30 10:05:00 | 409.95 | 409.01 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-02 11:15:00 | 414.95 | 418.20 | 0.00 | ORB-short ORB[415.65,420.95] vol=2.0x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-02-02 11:20:00 | 416.52 | 417.97 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-02-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:40:00 | 390.80 | 394.30 | 0.00 | ORB-short ORB[392.30,397.90] vol=1.6x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:50:00 | 387.22 | 392.18 | 0.00 | T1 1.5R @ 387.22 |
| Target hit | 2024-02-09 15:20:00 | 385.50 | 388.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2024-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:10:00 | 382.15 | 384.20 | 0.00 | ORB-short ORB[383.25,388.65] vol=4.4x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 10:20:00 | 378.79 | 383.22 | 0.00 | T1 1.5R @ 378.79 |
| Target hit | 2024-02-12 15:20:00 | 360.45 | 376.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2024-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 11:05:00 | 362.90 | 363.82 | 0.00 | ORB-short ORB[363.55,367.00] vol=3.0x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-02-16 11:20:00 | 363.82 | 363.73 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 10:05:00 | 380.05 | 376.95 | 0.00 | ORB-long ORB[367.75,372.65] vol=16.4x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-02-27 10:10:00 | 378.14 | 377.09 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-02-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 11:00:00 | 371.70 | 377.10 | 0.00 | ORB-short ORB[376.75,380.85] vol=4.6x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-02-28 11:05:00 | 373.05 | 377.06 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-03-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 10:05:00 | 366.45 | 368.75 | 0.00 | ORB-short ORB[368.00,373.00] vol=4.6x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 10:15:00 | 364.32 | 368.17 | 0.00 | T1 1.5R @ 364.32 |
| Target hit | 2024-03-04 14:35:00 | 364.80 | 362.82 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — SELL (started 2024-03-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:20:00 | 349.15 | 349.36 | 0.00 | ORB-short ORB[351.95,355.40] vol=1.8x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 10:30:00 | 346.25 | 349.28 | 0.00 | T1 1.5R @ 346.25 |
| Target hit | 2024-03-06 13:50:00 | 348.60 | 347.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — SELL (started 2024-03-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 11:10:00 | 319.70 | 321.03 | 0.00 | ORB-short ORB[321.60,325.50] vol=1.9x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-03-26 12:05:00 | 321.12 | 320.86 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-04-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 11:05:00 | 350.85 | 347.06 | 0.00 | ORB-long ORB[345.00,348.95] vol=8.3x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-05 11:35:00 | 352.64 | 349.36 | 0.00 | T1 1.5R @ 352.64 |
| Stop hit — per-position SL triggered | 2024-04-05 12:10:00 | 350.85 | 350.07 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-05-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:40:00 | 319.40 | 320.92 | 0.00 | ORB-short ORB[320.85,324.15] vol=2.1x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-05-09 10:55:00 | 320.66 | 320.90 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-17 10:15:00 | 190.50 | 2023-05-17 10:20:00 | 189.26 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2023-05-17 10:15:00 | 190.50 | 2023-05-17 15:20:00 | 187.30 | TARGET_HIT | 0.50 | 1.68% |
| SELL | retest1 | 2023-05-25 11:10:00 | 167.90 | 2023-05-25 11:30:00 | 168.76 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2023-05-26 10:55:00 | 175.15 | 2023-05-26 11:00:00 | 176.06 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-05-26 10:55:00 | 175.15 | 2023-05-26 15:20:00 | 178.80 | TARGET_HIT | 0.50 | 2.08% |
| BUY | retest1 | 2023-06-02 09:35:00 | 186.60 | 2023-06-02 09:45:00 | 187.41 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-06-02 09:35:00 | 186.60 | 2023-06-02 09:50:00 | 186.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-09 09:50:00 | 181.35 | 2023-06-09 10:00:00 | 182.07 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-06-13 10:05:00 | 181.00 | 2023-06-13 10:25:00 | 181.62 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-06-16 09:45:00 | 193.40 | 2023-06-16 09:50:00 | 194.38 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2023-06-20 10:55:00 | 192.95 | 2023-06-20 13:10:00 | 193.70 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-06-21 10:15:00 | 197.90 | 2023-06-21 10:20:00 | 197.03 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-06-22 09:30:00 | 201.80 | 2023-06-22 09:40:00 | 203.29 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2023-06-22 09:30:00 | 201.80 | 2023-06-22 10:00:00 | 202.00 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2023-07-10 09:40:00 | 211.15 | 2023-07-10 09:55:00 | 212.35 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2023-07-10 09:40:00 | 211.15 | 2023-07-10 10:55:00 | 212.15 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2023-07-19 10:40:00 | 217.95 | 2023-07-19 10:55:00 | 218.75 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-07-20 11:15:00 | 217.25 | 2023-07-20 11:25:00 | 217.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-21 09:50:00 | 216.85 | 2023-07-21 09:55:00 | 216.12 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-07-25 10:00:00 | 218.45 | 2023-07-25 10:10:00 | 219.58 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-25 10:00:00 | 218.45 | 2023-07-25 10:25:00 | 218.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-26 10:50:00 | 216.95 | 2023-07-26 10:55:00 | 216.40 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-27 10:40:00 | 216.65 | 2023-07-27 10:45:00 | 215.84 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-07-28 11:00:00 | 213.00 | 2023-07-28 11:45:00 | 213.52 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-11 09:45:00 | 249.90 | 2023-08-11 12:30:00 | 248.57 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-08-11 09:45:00 | 249.90 | 2023-08-11 15:05:00 | 249.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-17 09:35:00 | 248.70 | 2023-08-17 09:40:00 | 247.81 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-18 10:50:00 | 258.60 | 2023-08-18 10:55:00 | 257.55 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-08-23 10:20:00 | 275.05 | 2023-08-23 10:35:00 | 276.29 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-09-05 09:40:00 | 292.60 | 2023-09-05 09:45:00 | 294.14 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-09-05 09:40:00 | 292.60 | 2023-09-05 09:55:00 | 292.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-07 10:40:00 | 296.45 | 2023-09-07 10:45:00 | 298.17 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-09-07 10:40:00 | 296.45 | 2023-09-07 12:55:00 | 296.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-12 09:30:00 | 290.25 | 2023-09-12 09:40:00 | 288.82 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-09-12 09:30:00 | 290.25 | 2023-09-12 10:05:00 | 286.05 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2023-09-18 10:55:00 | 272.65 | 2023-09-18 12:15:00 | 273.74 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-09-21 09:30:00 | 271.95 | 2023-09-21 09:55:00 | 270.73 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-10-05 11:00:00 | 246.10 | 2023-10-05 11:05:00 | 247.47 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-10-05 11:00:00 | 246.10 | 2023-10-05 11:50:00 | 246.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-06 09:35:00 | 247.55 | 2023-10-06 09:55:00 | 246.50 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-10-11 09:50:00 | 252.85 | 2023-10-11 10:00:00 | 251.78 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-10-13 10:00:00 | 246.70 | 2023-10-13 10:45:00 | 245.16 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2023-10-13 10:00:00 | 246.70 | 2023-10-13 11:15:00 | 246.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-20 09:40:00 | 260.50 | 2023-10-20 10:05:00 | 259.38 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-10-25 10:00:00 | 245.35 | 2023-10-25 12:00:00 | 243.30 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2023-10-25 10:00:00 | 245.35 | 2023-10-25 15:20:00 | 234.60 | TARGET_HIT | 0.50 | 4.38% |
| SELL | retest1 | 2023-11-08 11:05:00 | 275.10 | 2023-11-08 11:25:00 | 275.97 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-11-09 09:50:00 | 269.40 | 2023-11-09 10:00:00 | 270.48 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-11-17 10:15:00 | 267.55 | 2023-11-17 12:10:00 | 265.90 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2023-11-17 10:15:00 | 267.55 | 2023-11-17 14:55:00 | 267.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-22 11:15:00 | 262.00 | 2023-11-22 11:35:00 | 262.71 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-11-23 11:10:00 | 264.90 | 2023-11-23 12:15:00 | 263.62 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-11-23 11:10:00 | 264.90 | 2023-11-23 15:20:00 | 262.85 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2023-11-24 10:25:00 | 262.80 | 2023-11-24 11:30:00 | 263.56 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-29 10:55:00 | 259.15 | 2023-11-29 11:15:00 | 259.92 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-04 09:35:00 | 277.40 | 2023-12-04 09:45:00 | 280.12 | PARTIAL | 0.50 | 0.98% |
| BUY | retest1 | 2023-12-04 09:35:00 | 277.40 | 2023-12-04 15:20:00 | 281.90 | TARGET_HIT | 0.50 | 1.62% |
| BUY | retest1 | 2023-12-08 10:25:00 | 286.80 | 2023-12-08 11:00:00 | 285.95 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-22 09:45:00 | 310.40 | 2023-12-22 10:40:00 | 308.79 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2023-12-27 10:40:00 | 306.95 | 2023-12-27 10:50:00 | 308.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-28 09:55:00 | 311.85 | 2023-12-28 10:00:00 | 313.66 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-12-28 09:55:00 | 311.85 | 2023-12-28 10:20:00 | 312.50 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2024-01-01 09:35:00 | 322.95 | 2024-01-01 09:40:00 | 325.06 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-01-01 09:35:00 | 322.95 | 2024-01-01 11:00:00 | 324.50 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2024-01-03 10:10:00 | 326.30 | 2024-01-03 10:20:00 | 324.58 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-01-05 10:55:00 | 333.20 | 2024-01-05 11:20:00 | 334.50 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-01-08 10:55:00 | 344.25 | 2024-01-08 11:00:00 | 342.56 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-01-19 10:00:00 | 395.30 | 2024-01-19 10:25:00 | 397.79 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-01-19 10:00:00 | 395.30 | 2024-01-19 10:45:00 | 395.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-29 10:45:00 | 399.00 | 2024-01-29 11:30:00 | 401.61 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-01-29 10:45:00 | 399.00 | 2024-01-29 15:20:00 | 401.95 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2024-01-30 09:50:00 | 409.95 | 2024-01-30 10:00:00 | 412.71 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-01-30 09:50:00 | 409.95 | 2024-01-30 10:05:00 | 409.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-02 11:15:00 | 414.95 | 2024-02-02 11:20:00 | 416.52 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-09 09:40:00 | 390.80 | 2024-02-09 09:50:00 | 387.22 | PARTIAL | 0.50 | 0.92% |
| SELL | retest1 | 2024-02-09 09:40:00 | 390.80 | 2024-02-09 15:20:00 | 385.50 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2024-02-12 10:10:00 | 382.15 | 2024-02-12 10:20:00 | 378.79 | PARTIAL | 0.50 | 0.88% |
| SELL | retest1 | 2024-02-12 10:10:00 | 382.15 | 2024-02-12 15:20:00 | 360.45 | TARGET_HIT | 0.50 | 5.68% |
| SELL | retest1 | 2024-02-16 11:05:00 | 362.90 | 2024-02-16 11:20:00 | 363.82 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-27 10:05:00 | 380.05 | 2024-02-27 10:10:00 | 378.14 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-02-28 11:00:00 | 371.70 | 2024-02-28 11:05:00 | 373.05 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-03-04 10:05:00 | 366.45 | 2024-03-04 10:15:00 | 364.32 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-03-04 10:05:00 | 366.45 | 2024-03-04 14:35:00 | 364.80 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2024-03-06 10:20:00 | 349.15 | 2024-03-06 10:30:00 | 346.25 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2024-03-06 10:20:00 | 349.15 | 2024-03-06 13:50:00 | 348.60 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2024-03-26 11:10:00 | 319.70 | 2024-03-26 12:05:00 | 321.12 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-04-05 11:05:00 | 350.85 | 2024-04-05 11:35:00 | 352.64 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-04-05 11:05:00 | 350.85 | 2024-04-05 12:10:00 | 350.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-09 10:40:00 | 319.40 | 2024-05-09 10:55:00 | 320.66 | STOP_HIT | 1.00 | -0.40% |
