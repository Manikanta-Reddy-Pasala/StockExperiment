# Steel Authority of India Ltd. (SAIL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 184.80
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
| ENTRY1 | 71 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 14 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 57
- **Target hits / Stop hits / Partials:** 14 / 57 / 29
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 15.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 25 | 43.1% | 9 | 33 | 16 | 0.13% | 7.5% |
| BUY @ 2nd Alert (retest1) | 58 | 25 | 43.1% | 9 | 33 | 16 | 0.13% | 7.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 18 | 42.9% | 5 | 24 | 13 | 0.18% | 7.6% |
| SELL @ 2nd Alert (retest1) | 42 | 18 | 42.9% | 5 | 24 | 13 | 0.18% | 7.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 100 | 43 | 43.0% | 14 | 57 | 29 | 0.15% | 15.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 164.85 | 166.92 | 0.00 | ORB-short ORB[167.35,168.75] vol=2.0x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 11:20:00 | 164.01 | 166.64 | 0.00 | T1 1.5R @ 164.01 |
| Stop hit — per-position SL triggered | 2024-05-16 11:55:00 | 164.85 | 166.22 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:00:00 | 166.55 | 165.65 | 0.00 | ORB-long ORB[164.00,166.35] vol=3.4x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 11:40:00 | 167.32 | 165.97 | 0.00 | T1 1.5R @ 167.32 |
| Target hit | 2024-05-17 14:55:00 | 167.00 | 167.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2024-05-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:30:00 | 170.20 | 169.37 | 0.00 | ORB-long ORB[167.60,169.50] vol=2.9x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-05-18 09:50:00 | 169.52 | 169.81 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 167.30 | 169.37 | 0.00 | ORB-short ORB[169.25,171.30] vol=1.6x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 168.11 | 168.75 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:10:00 | 165.20 | 167.17 | 0.00 | ORB-short ORB[167.20,169.20] vol=1.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:35:00 | 163.99 | 166.48 | 0.00 | T1 1.5R @ 163.99 |
| Target hit | 2024-05-28 15:20:00 | 162.05 | 164.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2024-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:35:00 | 149.63 | 150.81 | 0.00 | ORB-short ORB[150.70,151.83] vol=2.4x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-06-11 09:40:00 | 150.23 | 150.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:40:00 | 150.81 | 151.78 | 0.00 | ORB-short ORB[151.46,152.52] vol=2.5x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-06-13 10:05:00 | 151.25 | 151.53 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 154.95 | 154.28 | 0.00 | ORB-long ORB[153.51,154.50] vol=2.3x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-06-18 09:40:00 | 154.46 | 154.36 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:55:00 | 153.72 | 152.30 | 0.00 | ORB-long ORB[150.50,152.40] vol=1.9x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-06-20 13:45:00 | 153.21 | 152.90 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 156.36 | 155.36 | 0.00 | ORB-long ORB[153.91,155.95] vol=2.8x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-06-21 09:35:00 | 155.70 | 155.44 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:55:00 | 150.20 | 150.73 | 0.00 | ORB-short ORB[150.31,151.25] vol=1.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:15:00 | 149.45 | 150.45 | 0.00 | T1 1.5R @ 149.45 |
| Target hit | 2024-06-25 15:20:00 | 147.13 | 148.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-06-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:00:00 | 145.25 | 146.45 | 0.00 | ORB-short ORB[146.00,147.59] vol=2.4x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-06-26 12:05:00 | 145.79 | 145.75 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 146.40 | 145.54 | 0.00 | ORB-long ORB[144.30,146.25] vol=1.5x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-06-27 10:05:00 | 145.82 | 145.92 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:20:00 | 148.04 | 148.72 | 0.00 | ORB-short ORB[148.40,149.88] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-07-02 10:50:00 | 148.54 | 148.66 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 11:05:00 | 148.40 | 147.39 | 0.00 | ORB-long ORB[147.12,148.20] vol=2.3x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-07-03 11:10:00 | 147.90 | 147.42 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:15:00 | 151.93 | 150.94 | 0.00 | ORB-long ORB[150.00,151.40] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:35:00 | 152.72 | 151.18 | 0.00 | T1 1.5R @ 152.72 |
| Target hit | 2024-07-05 13:50:00 | 154.86 | 154.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2024-07-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:20:00 | 157.57 | 156.10 | 0.00 | ORB-long ORB[155.40,156.50] vol=2.2x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-07-08 10:30:00 | 156.88 | 156.24 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:45:00 | 152.69 | 151.92 | 0.00 | ORB-long ORB[151.10,152.35] vol=1.8x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-07-11 10:10:00 | 152.04 | 151.98 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:30:00 | 151.80 | 152.27 | 0.00 | ORB-short ORB[151.90,153.33] vol=3.1x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-07-12 10:50:00 | 152.23 | 152.25 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:55:00 | 151.30 | 150.10 | 0.00 | ORB-long ORB[149.01,151.20] vol=1.8x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:05:00 | 152.35 | 150.46 | 0.00 | T1 1.5R @ 152.35 |
| Target hit | 2024-07-15 13:10:00 | 151.78 | 151.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 153.48 | 153.04 | 0.00 | ORB-long ORB[152.31,153.19] vol=3.2x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-07-16 09:35:00 | 153.06 | 153.08 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:05:00 | 143.62 | 145.29 | 0.00 | ORB-short ORB[145.50,147.45] vol=2.7x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:10:00 | 142.61 | 144.99 | 0.00 | T1 1.5R @ 142.61 |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 143.62 | 144.92 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:05:00 | 149.73 | 148.39 | 0.00 | ORB-long ORB[147.25,149.14] vol=2.2x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-07-29 10:10:00 | 149.15 | 148.52 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:40:00 | 148.19 | 147.41 | 0.00 | ORB-long ORB[146.30,147.74] vol=1.6x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-07-30 09:50:00 | 147.73 | 147.50 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:35:00 | 150.30 | 149.29 | 0.00 | ORB-long ORB[147.66,149.58] vol=5.1x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 149.73 | 149.39 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:55:00 | 126.22 | 128.22 | 0.00 | ORB-short ORB[128.35,129.50] vol=3.0x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 11:05:00 | 125.31 | 127.70 | 0.00 | T1 1.5R @ 125.31 |
| Stop hit — per-position SL triggered | 2024-08-14 11:10:00 | 126.22 | 127.61 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 134.60 | 133.58 | 0.00 | ORB-long ORB[132.45,133.70] vol=3.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-08-21 10:00:00 | 134.16 | 133.79 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:45:00 | 133.74 | 134.53 | 0.00 | ORB-short ORB[134.30,135.39] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2024-08-22 09:50:00 | 134.07 | 134.48 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 135.16 | 135.72 | 0.00 | ORB-short ORB[135.40,136.20] vol=2.0x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 135.50 | 135.68 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 132.84 | 133.81 | 0.00 | ORB-short ORB[133.40,134.25] vol=1.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 12:00:00 | 132.25 | 133.56 | 0.00 | T1 1.5R @ 132.25 |
| Stop hit — per-position SL triggered | 2024-08-29 12:20:00 | 132.84 | 133.45 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:45:00 | 135.66 | 134.95 | 0.00 | ORB-long ORB[134.41,135.33] vol=1.8x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-08-30 09:50:00 | 135.13 | 134.99 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:20:00 | 132.16 | 132.88 | 0.00 | ORB-short ORB[132.25,133.50] vol=1.8x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 14:15:00 | 131.65 | 132.46 | 0.00 | T1 1.5R @ 131.65 |
| Target hit | 2024-09-03 15:20:00 | 131.72 | 132.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 130.21 | 129.78 | 0.00 | ORB-long ORB[128.70,130.14] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-09-04 10:25:00 | 129.78 | 129.96 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:50:00 | 129.66 | 130.49 | 0.00 | ORB-short ORB[130.26,131.77] vol=1.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:00:00 | 129.07 | 130.24 | 0.00 | T1 1.5R @ 129.07 |
| Stop hit — per-position SL triggered | 2024-09-06 11:50:00 | 129.66 | 129.36 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:50:00 | 131.26 | 132.15 | 0.00 | ORB-short ORB[131.96,133.00] vol=2.3x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:15:00 | 130.59 | 131.83 | 0.00 | T1 1.5R @ 130.59 |
| Stop hit — per-position SL triggered | 2024-09-17 10:35:00 | 131.26 | 131.71 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:05:00 | 127.80 | 129.38 | 0.00 | ORB-short ORB[129.80,130.99] vol=1.7x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:20:00 | 127.05 | 128.83 | 0.00 | T1 1.5R @ 127.05 |
| Target hit | 2024-09-19 12:45:00 | 126.58 | 126.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — SELL (started 2024-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 09:30:00 | 127.00 | 127.62 | 0.00 | ORB-short ORB[127.10,128.29] vol=1.8x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-09-20 09:55:00 | 127.56 | 127.45 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 128.80 | 127.66 | 0.00 | ORB-long ORB[126.50,127.94] vol=2.0x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:50:00 | 129.39 | 128.03 | 0.00 | T1 1.5R @ 129.39 |
| Target hit | 2024-09-23 15:20:00 | 129.89 | 128.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 134.05 | 132.95 | 0.00 | ORB-long ORB[131.29,133.29] vol=2.2x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:15:00 | 134.84 | 133.35 | 0.00 | T1 1.5R @ 134.84 |
| Stop hit — per-position SL triggered | 2024-09-24 10:45:00 | 134.05 | 133.59 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 132.10 | 131.07 | 0.00 | ORB-long ORB[129.97,131.40] vol=2.2x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 09:40:00 | 132.76 | 131.63 | 0.00 | T1 1.5R @ 132.76 |
| Target hit | 2024-10-11 11:15:00 | 132.53 | 132.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2024-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:35:00 | 128.98 | 129.83 | 0.00 | ORB-short ORB[129.78,130.99] vol=2.0x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:00:00 | 128.20 | 129.35 | 0.00 | T1 1.5R @ 128.20 |
| Stop hit — per-position SL triggered | 2024-10-17 10:50:00 | 128.98 | 129.02 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:55:00 | 117.14 | 116.62 | 0.00 | ORB-long ORB[115.15,116.59] vol=3.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-10-31 10:20:00 | 116.70 | 116.67 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 112.73 | 113.59 | 0.00 | ORB-short ORB[113.07,114.49] vol=1.8x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 111.87 | 112.87 | 0.00 | T1 1.5R @ 111.87 |
| Target hit | 2024-11-13 10:50:00 | 112.34 | 112.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — BUY (started 2024-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 09:35:00 | 115.79 | 114.97 | 0.00 | ORB-long ORB[113.88,115.08] vol=1.6x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-11-26 09:40:00 | 115.45 | 115.04 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:30:00 | 114.78 | 115.28 | 0.00 | ORB-short ORB[115.09,115.85] vol=1.6x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-11-27 09:35:00 | 115.10 | 115.25 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:45:00 | 117.39 | 116.66 | 0.00 | ORB-long ORB[115.86,117.26] vol=1.8x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 10:00:00 | 117.98 | 116.95 | 0.00 | T1 1.5R @ 117.98 |
| Stop hit — per-position SL triggered | 2024-12-02 10:35:00 | 117.39 | 117.16 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:05:00 | 121.89 | 122.79 | 0.00 | ORB-short ORB[122.79,123.54] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-12-04 11:50:00 | 122.18 | 122.58 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:50:00 | 127.90 | 127.18 | 0.00 | ORB-long ORB[126.31,127.58] vol=2.0x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-12-10 09:55:00 | 127.31 | 127.25 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:40:00 | 128.28 | 127.20 | 0.00 | ORB-long ORB[126.04,127.40] vol=2.0x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-12-11 09:45:00 | 127.81 | 127.32 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:40:00 | 119.00 | 117.83 | 0.00 | ORB-long ORB[116.50,117.72] vol=4.5x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-12-19 09:45:00 | 118.37 | 117.90 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:50:00 | 119.64 | 118.63 | 0.00 | ORB-long ORB[117.90,118.97] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-12-20 10:55:00 | 119.19 | 118.67 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:50:00 | 115.40 | 114.90 | 0.00 | ORB-long ORB[113.50,115.15] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-01-03 09:55:00 | 115.07 | 114.91 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:35:00 | 111.00 | 110.58 | 0.00 | ORB-long ORB[109.60,110.90] vol=1.8x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-01-21 09:55:00 | 110.61 | 110.70 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:50:00 | 108.78 | 107.99 | 0.00 | ORB-long ORB[106.50,107.21] vol=1.8x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-01-23 11:40:00 | 108.44 | 108.15 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 105.64 | 105.03 | 0.00 | ORB-long ORB[104.10,105.40] vol=1.9x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:05:00 | 106.16 | 105.24 | 0.00 | T1 1.5R @ 106.16 |
| Stop hit — per-position SL triggered | 2025-01-30 10:55:00 | 105.64 | 105.54 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:35:00 | 108.17 | 107.80 | 0.00 | ORB-long ORB[106.68,108.00] vol=1.7x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 11:35:00 | 108.80 | 108.00 | 0.00 | T1 1.5R @ 108.80 |
| Target hit | 2025-02-05 15:20:00 | 108.83 | 108.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 107.68 | 108.41 | 0.00 | ORB-short ORB[108.40,109.66] vol=4.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-02-06 09:40:00 | 108.07 | 108.34 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 09:30:00 | 109.15 | 108.57 | 0.00 | ORB-long ORB[107.84,108.80] vol=4.3x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-02-07 09:45:00 | 108.68 | 108.80 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:30:00 | 108.73 | 108.25 | 0.00 | ORB-long ORB[107.19,108.65] vol=3.7x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 09:40:00 | 109.57 | 108.57 | 0.00 | T1 1.5R @ 109.57 |
| Stop hit — per-position SL triggered | 2025-03-13 10:10:00 | 108.73 | 108.75 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-03-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 10:45:00 | 105.59 | 106.21 | 0.00 | ORB-short ORB[106.65,107.33] vol=4.1x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 12:25:00 | 105.06 | 106.02 | 0.00 | T1 1.5R @ 105.06 |
| Stop hit — per-position SL triggered | 2025-03-17 12:40:00 | 105.59 | 105.99 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:20:00 | 107.90 | 107.45 | 0.00 | ORB-long ORB[106.76,107.75] vol=2.1x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:50:00 | 108.44 | 107.57 | 0.00 | T1 1.5R @ 108.44 |
| Target hit | 2025-03-18 15:20:00 | 108.98 | 108.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2025-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 09:45:00 | 111.83 | 113.30 | 0.00 | ORB-short ORB[113.50,114.36] vol=1.9x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-03-20 09:55:00 | 112.23 | 113.03 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 114.80 | 114.30 | 0.00 | ORB-long ORB[113.45,114.48] vol=4.2x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:45:00 | 115.35 | 114.53 | 0.00 | T1 1.5R @ 115.35 |
| Stop hit — per-position SL triggered | 2025-03-21 10:05:00 | 114.80 | 114.61 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:30:00 | 110.97 | 110.42 | 0.00 | ORB-long ORB[109.66,110.88] vol=2.6x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 10:00:00 | 111.69 | 110.97 | 0.00 | T1 1.5R @ 111.69 |
| Target hit | 2025-04-15 15:20:00 | 113.42 | 112.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:15:00 | 115.01 | 114.08 | 0.00 | ORB-long ORB[112.80,114.25] vol=2.1x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:25:00 | 115.59 | 114.30 | 0.00 | T1 1.5R @ 115.59 |
| Target hit | 2025-04-21 15:20:00 | 115.99 | 115.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 115.81 | 116.69 | 0.00 | ORB-short ORB[116.51,117.35] vol=1.6x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 116.23 | 116.53 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:35:00 | 117.37 | 116.98 | 0.00 | ORB-long ORB[116.20,117.18] vol=1.7x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:35:00 | 117.91 | 117.28 | 0.00 | T1 1.5R @ 117.91 |
| Stop hit — per-position SL triggered | 2025-04-24 10:40:00 | 117.37 | 117.28 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 10:55:00 | 116.77 | 115.72 | 0.00 | ORB-long ORB[114.43,115.75] vol=2.1x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 11:25:00 | 117.36 | 116.10 | 0.00 | T1 1.5R @ 117.36 |
| Stop hit — per-position SL triggered | 2025-04-28 13:30:00 | 116.77 | 116.55 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:30:00 | 118.12 | 117.70 | 0.00 | ORB-long ORB[116.75,117.99] vol=1.9x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-04-29 09:35:00 | 117.77 | 117.76 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 09:50:00 | 112.88 | 113.80 | 0.00 | ORB-short ORB[113.45,114.84] vol=1.5x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-05-06 10:05:00 | 113.31 | 113.66 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:45:00 | 113.10 | 113.48 | 0.00 | ORB-short ORB[113.18,114.80] vol=2.0x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-05-08 10:05:00 | 113.51 | 113.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 11:15:00 | 164.85 | 2024-05-16 11:20:00 | 164.01 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-16 11:15:00 | 164.85 | 2024-05-16 11:55:00 | 164.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 11:00:00 | 166.55 | 2024-05-17 11:40:00 | 167.32 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-05-17 11:00:00 | 166.55 | 2024-05-17 14:55:00 | 167.00 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2024-05-18 09:30:00 | 170.20 | 2024-05-18 09:50:00 | 169.52 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-27 09:45:00 | 167.30 | 2024-05-27 10:05:00 | 168.11 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-05-28 10:10:00 | 165.20 | 2024-05-28 11:35:00 | 163.99 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-05-28 10:10:00 | 165.20 | 2024-05-28 15:20:00 | 162.05 | TARGET_HIT | 0.50 | 1.91% |
| SELL | retest1 | 2024-06-11 09:35:00 | 149.63 | 2024-06-11 09:40:00 | 150.23 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-06-13 09:40:00 | 150.81 | 2024-06-13 10:05:00 | 151.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-18 09:30:00 | 154.95 | 2024-06-18 09:40:00 | 154.46 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-20 10:55:00 | 153.72 | 2024-06-20 13:45:00 | 153.21 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-21 09:30:00 | 156.36 | 2024-06-21 09:35:00 | 155.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-06-25 09:55:00 | 150.20 | 2024-06-25 11:15:00 | 149.45 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-06-25 09:55:00 | 150.20 | 2024-06-25 15:20:00 | 147.13 | TARGET_HIT | 0.50 | 2.04% |
| SELL | retest1 | 2024-06-26 10:00:00 | 145.25 | 2024-06-26 12:05:00 | 145.79 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-27 09:40:00 | 146.40 | 2024-06-27 10:05:00 | 145.82 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-02 10:20:00 | 148.04 | 2024-07-02 10:50:00 | 148.54 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-03 11:05:00 | 148.40 | 2024-07-03 11:10:00 | 147.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-05 10:15:00 | 151.93 | 2024-07-05 10:35:00 | 152.72 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-07-05 10:15:00 | 151.93 | 2024-07-05 13:50:00 | 154.86 | TARGET_HIT | 0.50 | 1.93% |
| BUY | retest1 | 2024-07-08 10:20:00 | 157.57 | 2024-07-08 10:30:00 | 156.88 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-11 09:45:00 | 152.69 | 2024-07-11 10:10:00 | 152.04 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-12 10:30:00 | 151.80 | 2024-07-12 10:50:00 | 152.23 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-15 09:55:00 | 151.30 | 2024-07-15 10:05:00 | 152.35 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-07-15 09:55:00 | 151.30 | 2024-07-15 13:10:00 | 151.78 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-07-16 09:30:00 | 153.48 | 2024-07-16 09:35:00 | 153.06 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-19 10:05:00 | 143.62 | 2024-07-19 10:10:00 | 142.61 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-07-19 10:05:00 | 143.62 | 2024-07-19 10:15:00 | 143.62 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-29 10:05:00 | 149.73 | 2024-07-29 10:10:00 | 149.15 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-30 09:40:00 | 148.19 | 2024-07-30 09:50:00 | 147.73 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-31 09:35:00 | 150.30 | 2024-07-31 09:45:00 | 149.73 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-14 10:55:00 | 126.22 | 2024-08-14 11:05:00 | 125.31 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-08-14 10:55:00 | 126.22 | 2024-08-14 11:10:00 | 126.22 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 09:45:00 | 134.60 | 2024-08-21 10:00:00 | 134.16 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-22 09:45:00 | 133.74 | 2024-08-22 09:50:00 | 134.07 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-28 09:30:00 | 135.16 | 2024-08-28 09:35:00 | 135.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-29 10:55:00 | 132.84 | 2024-08-29 12:00:00 | 132.25 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-29 10:55:00 | 132.84 | 2024-08-29 12:20:00 | 132.84 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 09:45:00 | 135.66 | 2024-08-30 09:50:00 | 135.13 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-03 10:20:00 | 132.16 | 2024-09-03 14:15:00 | 131.65 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-09-03 10:20:00 | 132.16 | 2024-09-03 15:20:00 | 131.72 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2024-09-04 09:30:00 | 130.21 | 2024-09-04 10:25:00 | 129.78 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-06 09:50:00 | 129.66 | 2024-09-06 10:00:00 | 129.07 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-06 09:50:00 | 129.66 | 2024-09-06 11:50:00 | 129.66 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 09:50:00 | 131.26 | 2024-09-17 10:15:00 | 130.59 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-17 09:50:00 | 131.26 | 2024-09-17 10:35:00 | 131.26 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:05:00 | 127.80 | 2024-09-19 10:20:00 | 127.05 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-19 10:05:00 | 127.80 | 2024-09-19 12:45:00 | 126.58 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2024-09-20 09:30:00 | 127.00 | 2024-09-20 09:55:00 | 127.56 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-09-23 11:00:00 | 128.80 | 2024-09-23 11:50:00 | 129.39 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-09-23 11:00:00 | 128.80 | 2024-09-23 15:20:00 | 129.89 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2024-09-24 09:55:00 | 134.05 | 2024-09-24 10:15:00 | 134.84 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-09-24 09:55:00 | 134.05 | 2024-09-24 10:45:00 | 134.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:35:00 | 132.10 | 2024-10-11 09:40:00 | 132.76 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-10-11 09:35:00 | 132.10 | 2024-10-11 11:15:00 | 132.53 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2024-10-17 09:35:00 | 128.98 | 2024-10-17 10:00:00 | 128.20 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-10-17 09:35:00 | 128.98 | 2024-10-17 10:50:00 | 128.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 09:55:00 | 117.14 | 2024-10-31 10:20:00 | 116.70 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-13 09:30:00 | 112.73 | 2024-11-13 09:40:00 | 111.87 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-11-13 09:30:00 | 112.73 | 2024-11-13 10:50:00 | 112.34 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-11-26 09:35:00 | 115.79 | 2024-11-26 09:40:00 | 115.45 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-27 09:30:00 | 114.78 | 2024-11-27 09:35:00 | 115.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-02 09:45:00 | 117.39 | 2024-12-02 10:00:00 | 117.98 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-12-02 09:45:00 | 117.39 | 2024-12-02 10:35:00 | 117.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-04 11:05:00 | 121.89 | 2024-12-04 11:50:00 | 122.18 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-10 09:50:00 | 127.90 | 2024-12-10 09:55:00 | 127.31 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-12-11 09:40:00 | 128.28 | 2024-12-11 09:45:00 | 127.81 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-19 09:40:00 | 119.00 | 2024-12-19 09:45:00 | 118.37 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-12-20 10:50:00 | 119.64 | 2024-12-20 10:55:00 | 119.19 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-03 09:50:00 | 115.40 | 2025-01-03 09:55:00 | 115.07 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-21 09:35:00 | 111.00 | 2025-01-21 09:55:00 | 110.61 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-23 10:50:00 | 108.78 | 2025-01-23 11:40:00 | 108.44 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-30 09:45:00 | 105.64 | 2025-01-30 10:05:00 | 106.16 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-01-30 09:45:00 | 105.64 | 2025-01-30 10:55:00 | 105.64 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 10:35:00 | 108.17 | 2025-02-05 11:35:00 | 108.80 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-02-05 10:35:00 | 108.17 | 2025-02-05 15:20:00 | 108.83 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2025-02-06 09:30:00 | 107.68 | 2025-02-06 09:40:00 | 108.07 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-02-07 09:30:00 | 109.15 | 2025-02-07 09:45:00 | 108.68 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-03-13 09:30:00 | 108.73 | 2025-03-13 09:40:00 | 109.57 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2025-03-13 09:30:00 | 108.73 | 2025-03-13 10:10:00 | 108.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-17 10:45:00 | 105.59 | 2025-03-17 12:25:00 | 105.06 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-03-17 10:45:00 | 105.59 | 2025-03-17 12:40:00 | 105.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:20:00 | 107.90 | 2025-03-18 10:50:00 | 108.44 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-03-18 10:20:00 | 107.90 | 2025-03-18 15:20:00 | 108.98 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2025-03-20 09:45:00 | 111.83 | 2025-03-20 09:55:00 | 112.23 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-21 09:30:00 | 114.80 | 2025-03-21 09:45:00 | 115.35 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-03-21 09:30:00 | 114.80 | 2025-03-21 10:05:00 | 114.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-15 09:30:00 | 110.97 | 2025-04-15 10:00:00 | 111.69 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-04-15 09:30:00 | 110.97 | 2025-04-15 15:20:00 | 113.42 | TARGET_HIT | 0.50 | 2.21% |
| BUY | retest1 | 2025-04-21 10:15:00 | 115.01 | 2025-04-21 10:25:00 | 115.59 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-21 10:15:00 | 115.01 | 2025-04-21 15:20:00 | 115.99 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2025-04-23 09:30:00 | 115.81 | 2025-04-23 09:45:00 | 116.23 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-24 09:35:00 | 117.37 | 2025-04-24 10:35:00 | 117.91 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-24 09:35:00 | 117.37 | 2025-04-24 10:40:00 | 117.37 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-28 10:55:00 | 116.77 | 2025-04-28 11:25:00 | 117.36 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-04-28 10:55:00 | 116.77 | 2025-04-28 13:30:00 | 116.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-29 09:30:00 | 118.12 | 2025-04-29 09:35:00 | 117.77 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-06 09:50:00 | 112.88 | 2025-05-06 10:05:00 | 113.31 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-05-08 09:45:00 | 113.10 | 2025-05-08 10:05:00 | 113.51 | STOP_HIT | 1.00 | -0.36% |
