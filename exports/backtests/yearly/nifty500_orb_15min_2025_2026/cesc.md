# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 185.00
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
| ENTRY1 | 88 |
| ENTRY2 | 0 |
| PARTIAL | 42 |
| TARGET_HIT | 21 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 130 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 67
- **Target hits / Stop hits / Partials:** 21 / 67 / 42
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 34.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 22 | 36.1% | 8 | 39 | 14 | 0.22% | 13.1% |
| BUY @ 2nd Alert (retest1) | 61 | 22 | 36.1% | 8 | 39 | 14 | 0.22% | 13.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 69 | 41 | 59.4% | 13 | 28 | 28 | 0.32% | 21.7% |
| SELL @ 2nd Alert (retest1) | 69 | 41 | 59.4% | 13 | 28 | 28 | 0.32% | 21.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 130 | 63 | 48.5% | 21 | 67 | 42 | 0.27% | 34.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:35:00 | 165.70 | 164.75 | 0.00 | ORB-long ORB[163.50,165.50] vol=2.3x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 09:40:00 | 166.60 | 165.53 | 0.00 | T1 1.5R @ 166.60 |
| Stop hit — per-position SL triggered | 2025-05-14 10:05:00 | 165.70 | 166.77 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:10:00 | 166.70 | 164.07 | 0.00 | ORB-long ORB[161.78,163.87] vol=3.0x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-05-16 10:15:00 | 165.84 | 164.65 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 169.50 | 168.84 | 0.00 | ORB-long ORB[167.81,169.46] vol=2.2x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-05-23 10:10:00 | 168.90 | 168.96 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:05:00 | 169.03 | 170.27 | 0.00 | ORB-short ORB[170.30,171.50] vol=1.6x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 10:15:00 | 168.32 | 169.82 | 0.00 | T1 1.5R @ 168.32 |
| Stop hit — per-position SL triggered | 2025-05-26 11:45:00 | 169.03 | 169.06 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:40:00 | 169.12 | 168.59 | 0.00 | ORB-long ORB[167.65,168.88] vol=2.0x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-05-28 09:45:00 | 168.71 | 168.61 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:30:00 | 163.14 | 165.80 | 0.00 | ORB-short ORB[167.62,168.65] vol=1.6x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 12:35:00 | 162.23 | 164.25 | 0.00 | T1 1.5R @ 162.23 |
| Stop hit — per-position SL triggered | 2025-05-30 13:35:00 | 163.14 | 163.93 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:40:00 | 163.55 | 164.23 | 0.00 | ORB-short ORB[164.01,165.00] vol=1.9x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 10:00:00 | 162.84 | 163.98 | 0.00 | T1 1.5R @ 162.84 |
| Target hit | 2025-06-03 15:20:00 | 161.35 | 162.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 163.65 | 163.01 | 0.00 | ORB-long ORB[161.80,163.30] vol=2.3x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-06-04 09:35:00 | 163.19 | 163.05 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 11:15:00 | 167.69 | 168.93 | 0.00 | ORB-short ORB[168.51,171.00] vol=1.8x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 13:00:00 | 166.95 | 168.66 | 0.00 | T1 1.5R @ 166.95 |
| Target hit | 2025-06-05 15:20:00 | 166.88 | 167.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:10:00 | 169.50 | 170.86 | 0.00 | ORB-short ORB[170.61,171.87] vol=2.3x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:55:00 | 168.79 | 170.51 | 0.00 | T1 1.5R @ 168.79 |
| Stop hit — per-position SL triggered | 2025-06-12 12:45:00 | 169.50 | 170.36 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 163.53 | 164.57 | 0.00 | ORB-short ORB[164.02,165.90] vol=1.8x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:45:00 | 162.87 | 164.14 | 0.00 | T1 1.5R @ 162.87 |
| Stop hit — per-position SL triggered | 2025-06-16 09:50:00 | 163.53 | 163.94 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 164.45 | 163.74 | 0.00 | ORB-long ORB[162.15,164.35] vol=1.9x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:45:00 | 165.33 | 164.04 | 0.00 | T1 1.5R @ 165.33 |
| Stop hit — per-position SL triggered | 2025-06-18 10:30:00 | 164.45 | 164.25 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:35:00 | 163.00 | 163.78 | 0.00 | ORB-short ORB[163.26,164.78] vol=3.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-06-19 10:40:00 | 163.60 | 163.75 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:10:00 | 163.22 | 162.43 | 0.00 | ORB-long ORB[161.10,163.20] vol=2.0x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-06-23 11:20:00 | 162.61 | 162.46 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:45:00 | 169.29 | 168.01 | 0.00 | ORB-long ORB[166.74,168.10] vol=2.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-06-25 09:50:00 | 168.75 | 168.14 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:05:00 | 175.88 | 176.20 | 0.00 | ORB-short ORB[176.60,178.78] vol=2.0x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-07-02 11:45:00 | 176.48 | 176.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:30:00 | 179.28 | 178.59 | 0.00 | ORB-long ORB[177.26,178.98] vol=1.7x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-07-07 09:40:00 | 178.77 | 178.74 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 09:45:00 | 182.24 | 181.40 | 0.00 | ORB-long ORB[180.53,181.75] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-07-08 10:30:00 | 181.73 | 181.90 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:50:00 | 179.33 | 180.60 | 0.00 | ORB-short ORB[179.74,182.00] vol=2.7x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:00:00 | 178.80 | 180.53 | 0.00 | T1 1.5R @ 178.80 |
| Stop hit — per-position SL triggered | 2025-07-10 11:05:00 | 179.33 | 180.40 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:10:00 | 176.40 | 178.04 | 0.00 | ORB-short ORB[177.54,179.56] vol=2.3x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:20:00 | 175.37 | 177.67 | 0.00 | T1 1.5R @ 175.37 |
| Target hit | 2025-07-11 15:20:00 | 174.42 | 176.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-07-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:30:00 | 179.35 | 178.95 | 0.00 | ORB-long ORB[177.00,178.75] vol=8.9x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-07-21 10:50:00 | 178.96 | 178.92 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:30:00 | 179.33 | 178.94 | 0.00 | ORB-long ORB[177.57,179.20] vol=4.3x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-07-24 09:40:00 | 179.02 | 179.01 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 09:30:00 | 177.40 | 176.36 | 0.00 | ORB-long ORB[174.56,177.20] vol=2.2x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-07-25 09:40:00 | 176.96 | 176.60 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:50:00 | 175.40 | 174.25 | 0.00 | ORB-long ORB[172.16,174.15] vol=2.3x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-07-28 11:05:00 | 174.89 | 174.32 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:40:00 | 163.83 | 164.41 | 0.00 | ORB-short ORB[163.98,165.44] vol=2.3x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:05:00 | 163.10 | 164.08 | 0.00 | T1 1.5R @ 163.10 |
| Stop hit — per-position SL triggered | 2025-08-06 10:20:00 | 163.83 | 164.06 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:10:00 | 161.40 | 162.36 | 0.00 | ORB-short ORB[161.61,163.69] vol=2.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:40:00 | 160.76 | 162.12 | 0.00 | T1 1.5R @ 160.76 |
| Stop hit — per-position SL triggered | 2025-08-11 12:35:00 | 161.40 | 161.87 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:00:00 | 165.00 | 163.92 | 0.00 | ORB-long ORB[162.00,163.99] vol=1.9x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-08-12 10:15:00 | 164.54 | 164.10 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:10:00 | 165.72 | 164.90 | 0.00 | ORB-long ORB[164.04,165.35] vol=2.8x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-08-13 10:20:00 | 165.26 | 164.98 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 11:05:00 | 164.66 | 164.38 | 0.00 | ORB-long ORB[163.60,164.60] vol=1.9x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 164.47 | 164.39 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:35:00 | 165.20 | 164.59 | 0.00 | ORB-long ORB[163.63,164.90] vol=1.7x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:40:00 | 165.84 | 164.70 | 0.00 | T1 1.5R @ 165.84 |
| Target hit | 2025-08-22 14:25:00 | 166.14 | 166.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2025-08-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:35:00 | 167.90 | 166.62 | 0.00 | ORB-long ORB[165.60,167.64] vol=1.9x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-08-25 10:40:00 | 167.42 | 166.80 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 165.72 | 164.89 | 0.00 | ORB-long ORB[163.00,164.81] vol=6.2x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:40:00 | 166.50 | 166.27 | 0.00 | T1 1.5R @ 166.50 |
| Target hit | 2025-08-26 10:10:00 | 168.70 | 169.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-28 11:15:00 | 160.00 | 160.28 | 0.00 | ORB-short ORB[160.06,162.10] vol=7.8x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:20:00 | 159.41 | 160.22 | 0.00 | T1 1.5R @ 159.41 |
| Target hit | 2025-08-28 15:20:00 | 153.85 | 156.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 09:35:00 | 155.91 | 156.41 | 0.00 | ORB-short ORB[156.00,157.61] vol=2.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:45:00 | 155.23 | 156.00 | 0.00 | T1 1.5R @ 155.23 |
| Target hit | 2025-09-03 15:20:00 | 155.15 | 155.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 156.19 | 155.43 | 0.00 | ORB-long ORB[154.06,156.06] vol=1.7x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-09-08 09:35:00 | 155.67 | 155.49 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:50:00 | 162.05 | 160.11 | 0.00 | ORB-long ORB[158.11,160.43] vol=4.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 161.40 | 160.31 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 11:00:00 | 160.89 | 161.30 | 0.00 | ORB-short ORB[160.93,162.54] vol=3.1x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:35:00 | 160.26 | 161.16 | 0.00 | T1 1.5R @ 160.26 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 160.89 | 161.05 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:55:00 | 160.01 | 161.15 | 0.00 | ORB-short ORB[161.00,162.32] vol=1.7x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:30:00 | 159.43 | 161.01 | 0.00 | T1 1.5R @ 159.43 |
| Stop hit — per-position SL triggered | 2025-09-12 12:40:00 | 160.01 | 160.79 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:20:00 | 168.48 | 167.44 | 0.00 | ORB-long ORB[165.40,167.48] vol=3.2x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-09-16 11:50:00 | 167.83 | 167.66 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:10:00 | 165.41 | 166.30 | 0.00 | ORB-short ORB[165.70,167.67] vol=2.5x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-09-18 11:25:00 | 165.79 | 166.27 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 11:05:00 | 167.09 | 165.85 | 0.00 | ORB-long ORB[165.10,166.49] vol=10.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-09-19 11:10:00 | 166.61 | 165.98 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:45:00 | 164.13 | 163.30 | 0.00 | ORB-long ORB[162.36,164.03] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-09-29 10:05:00 | 163.59 | 163.46 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:55:00 | 162.00 | 162.92 | 0.00 | ORB-short ORB[162.40,163.87] vol=3.6x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 12:15:00 | 161.28 | 162.30 | 0.00 | T1 1.5R @ 161.28 |
| Target hit | 2025-10-01 13:55:00 | 161.75 | 161.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — BUY (started 2025-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 09:30:00 | 165.00 | 164.17 | 0.00 | ORB-long ORB[163.05,164.50] vol=1.5x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 09:40:00 | 165.77 | 164.72 | 0.00 | T1 1.5R @ 165.77 |
| Target hit | 2025-10-08 13:10:00 | 166.31 | 166.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — BUY (started 2025-10-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:25:00 | 168.00 | 166.34 | 0.00 | ORB-long ORB[165.25,167.00] vol=3.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:40:00 | 168.95 | 167.29 | 0.00 | T1 1.5R @ 168.95 |
| Target hit | 2025-10-09 15:20:00 | 170.00 | 169.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 171.25 | 170.39 | 0.00 | ORB-long ORB[169.21,170.88] vol=4.8x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 09:40:00 | 172.12 | 170.95 | 0.00 | T1 1.5R @ 172.12 |
| Stop hit — per-position SL triggered | 2025-10-10 09:55:00 | 171.25 | 171.29 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:35:00 | 169.64 | 170.72 | 0.00 | ORB-short ORB[169.76,171.68] vol=2.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 168.96 | 170.46 | 0.00 | T1 1.5R @ 168.96 |
| Target hit | 2025-10-14 15:20:00 | 167.46 | 168.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:35:00 | 171.00 | 170.33 | 0.00 | ORB-long ORB[168.71,170.62] vol=2.0x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:05:00 | 172.03 | 170.84 | 0.00 | T1 1.5R @ 172.03 |
| Target hit | 2025-10-15 15:20:00 | 176.35 | 174.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:15:00 | 179.11 | 180.13 | 0.00 | ORB-short ORB[179.50,181.39] vol=1.6x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 179.58 | 179.92 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:45:00 | 181.58 | 181.33 | 0.00 | ORB-long ORB[179.51,181.47] vol=1.8x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-11-04 11:25:00 | 181.04 | 181.33 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:15:00 | 179.34 | 180.28 | 0.00 | ORB-short ORB[180.50,181.80] vol=2.3x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:35:00 | 178.49 | 179.99 | 0.00 | T1 1.5R @ 178.49 |
| Stop hit — per-position SL triggered | 2025-11-06 11:00:00 | 179.34 | 179.89 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:15:00 | 173.64 | 174.15 | 0.00 | ORB-short ORB[174.21,175.95] vol=1.8x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:35:00 | 173.05 | 173.96 | 0.00 | T1 1.5R @ 173.05 |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 173.64 | 173.62 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:30:00 | 170.98 | 172.40 | 0.00 | ORB-short ORB[172.13,173.88] vol=3.3x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-11-11 09:45:00 | 171.58 | 171.65 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:45:00 | 172.63 | 171.77 | 0.00 | ORB-long ORB[170.15,171.69] vol=2.2x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-11-12 11:05:00 | 172.12 | 171.85 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 11:15:00 | 173.93 | 174.22 | 0.00 | ORB-short ORB[174.50,175.93] vol=1.6x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-11-18 11:45:00 | 174.22 | 174.19 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:45:00 | 172.30 | 173.00 | 0.00 | ORB-short ORB[172.80,174.51] vol=2.6x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 14:05:00 | 171.70 | 172.25 | 0.00 | T1 1.5R @ 171.70 |
| Target hit | 2025-11-19 15:20:00 | 171.34 | 171.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-11-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:35:00 | 169.85 | 169.97 | 0.00 | ORB-short ORB[170.10,171.69] vol=1.9x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:20:00 | 169.21 | 169.08 | 0.00 | T1 1.5R @ 169.21 |
| Stop hit — per-position SL triggered | 2025-11-24 15:00:00 | 169.85 | 168.67 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:05:00 | 170.27 | 171.91 | 0.00 | ORB-short ORB[172.25,174.40] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-11-27 10:30:00 | 170.77 | 171.42 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:15:00 | 176.60 | 175.65 | 0.00 | ORB-long ORB[175.03,176.00] vol=1.7x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-12-05 10:20:00 | 176.17 | 175.67 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:30:00 | 166.61 | 167.34 | 0.00 | ORB-short ORB[167.00,168.48] vol=1.5x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:50:00 | 165.58 | 166.79 | 0.00 | T1 1.5R @ 165.58 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 166.61 | 166.30 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:10:00 | 169.25 | 168.49 | 0.00 | ORB-long ORB[167.07,168.39] vol=1.5x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:20:00 | 169.89 | 168.77 | 0.00 | T1 1.5R @ 169.89 |
| Stop hit — per-position SL triggered | 2025-12-11 11:50:00 | 169.25 | 168.96 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:25:00 | 171.63 | 171.04 | 0.00 | ORB-long ORB[170.00,170.95] vol=3.4x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-12-16 11:05:00 | 171.29 | 171.15 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:05:00 | 165.11 | 165.88 | 0.00 | ORB-short ORB[165.80,166.88] vol=2.3x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:20:00 | 164.75 | 165.78 | 0.00 | T1 1.5R @ 164.75 |
| Stop hit — per-position SL triggered | 2025-12-29 11:50:00 | 165.11 | 165.73 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:30:00 | 165.40 | 164.61 | 0.00 | ORB-long ORB[163.99,165.00] vol=1.8x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-12-30 10:35:00 | 165.03 | 164.63 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 166.90 | 166.25 | 0.00 | ORB-long ORB[165.64,166.48] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-12-31 11:20:00 | 166.61 | 166.27 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:25:00 | 167.97 | 167.64 | 0.00 | ORB-long ORB[167.01,167.95] vol=2.2x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:45:00 | 168.49 | 167.94 | 0.00 | T1 1.5R @ 168.49 |
| Target hit | 2026-01-02 15:20:00 | 175.66 | 173.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2026-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:40:00 | 171.81 | 172.78 | 0.00 | ORB-short ORB[172.51,174.50] vol=2.0x ATR=0.53 |
| Stop hit — per-position SL triggered | 2026-01-06 09:45:00 | 172.34 | 172.77 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 170.15 | 170.89 | 0.00 | ORB-short ORB[170.45,171.80] vol=2.9x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:25:00 | 169.64 | 170.74 | 0.00 | T1 1.5R @ 169.64 |
| Target hit | 2026-01-08 15:20:00 | 166.92 | 169.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2026-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:35:00 | 165.14 | 165.75 | 0.00 | ORB-short ORB[165.21,167.19] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-01-09 09:40:00 | 165.68 | 165.74 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:00:00 | 161.92 | 162.99 | 0.00 | ORB-short ORB[162.45,164.42] vol=2.6x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 12:25:00 | 160.91 | 162.23 | 0.00 | T1 1.5R @ 160.91 |
| Target hit | 2026-01-13 15:20:00 | 160.00 | 161.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — SELL (started 2026-01-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 157.80 | 158.55 | 0.00 | ORB-short ORB[158.01,160.00] vol=2.0x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:05:00 | 156.98 | 158.06 | 0.00 | T1 1.5R @ 156.98 |
| Target hit | 2026-01-14 10:20:00 | 157.53 | 157.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — SELL (started 2026-01-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 11:10:00 | 153.80 | 154.68 | 0.00 | ORB-short ORB[153.88,155.71] vol=2.8x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:50:00 | 153.22 | 154.47 | 0.00 | T1 1.5R @ 153.22 |
| Target hit | 2026-01-19 15:20:00 | 152.14 | 152.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2026-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:45:00 | 149.83 | 150.65 | 0.00 | ORB-short ORB[150.21,152.21] vol=1.7x ATR=0.60 |
| Stop hit — per-position SL triggered | 2026-01-20 09:50:00 | 150.43 | 150.62 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:25:00 | 146.87 | 145.60 | 0.00 | ORB-long ORB[143.60,145.60] vol=2.5x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 11:25:00 | 147.71 | 146.18 | 0.00 | T1 1.5R @ 147.71 |
| Stop hit — per-position SL triggered | 2026-01-30 12:00:00 | 146.87 | 146.44 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:10:00 | 151.80 | 152.86 | 0.00 | ORB-short ORB[152.65,154.43] vol=2.2x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-02-06 11:30:00 | 152.29 | 152.80 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:25:00 | 155.35 | 154.89 | 0.00 | ORB-long ORB[154.41,155.29] vol=1.6x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 155.00 | 154.93 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 153.87 | 154.23 | 0.00 | ORB-short ORB[154.00,155.23] vol=2.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 153.35 | 153.99 | 0.00 | T1 1.5R @ 153.35 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 153.87 | 153.81 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 152.30 | 151.53 | 0.00 | ORB-long ORB[150.33,151.95] vol=1.8x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-02-16 09:45:00 | 151.79 | 151.67 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 153.50 | 152.17 | 0.00 | ORB-long ORB[150.82,152.61] vol=1.6x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:55:00 | 154.51 | 153.26 | 0.00 | T1 1.5R @ 154.51 |
| Target hit | 2026-02-20 12:10:00 | 154.56 | 154.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 161.20 | 160.98 | 0.00 | ORB-long ORB[159.79,161.18] vol=1.7x ATR=0.42 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 160.78 | 160.94 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-04-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:10:00 | 152.00 | 153.12 | 0.00 | ORB-short ORB[152.98,154.90] vol=1.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-04-01 11:25:00 | 152.49 | 153.05 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-04-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:50:00 | 155.99 | 157.28 | 0.00 | ORB-short ORB[156.18,158.50] vol=2.0x ATR=0.59 |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 156.58 | 156.94 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 158.40 | 157.93 | 0.00 | ORB-long ORB[156.75,158.38] vol=2.0x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 157.91 | 158.06 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:40:00 | 157.14 | 156.04 | 0.00 | ORB-long ORB[155.00,156.74] vol=2.2x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 09:50:00 | 158.02 | 156.65 | 0.00 | T1 1.5R @ 158.02 |
| Target hit | 2026-04-13 14:05:00 | 161.42 | 161.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 85 — BUY (started 2026-04-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:30:00 | 164.40 | 163.28 | 0.00 | ORB-long ORB[162.30,163.80] vol=2.3x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 163.79 | 163.46 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:25:00 | 165.40 | 164.32 | 0.00 | ORB-long ORB[163.82,165.00] vol=1.9x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-16 10:40:00 | 164.85 | 164.46 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 190.50 | 189.45 | 0.00 | ORB-long ORB[187.78,190.20] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:45:00 | 191.71 | 189.78 | 0.00 | T1 1.5R @ 191.71 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 190.50 | 189.80 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 185.50 | 185.67 | 0.00 | ORB-short ORB[185.61,187.45] vol=1.6x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:45:00 | 184.69 | 185.62 | 0.00 | T1 1.5R @ 184.69 |
| Target hit | 2026-05-08 15:20:00 | 184.65 | 185.24 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:35:00 | 165.70 | 2025-05-14 09:40:00 | 166.60 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-05-14 09:35:00 | 165.70 | 2025-05-14 10:05:00 | 165.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-16 10:10:00 | 166.70 | 2025-05-16 10:15:00 | 165.84 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-05-23 09:30:00 | 169.50 | 2025-05-23 10:10:00 | 168.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-05-26 10:05:00 | 169.03 | 2025-05-26 10:15:00 | 168.32 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-05-26 10:05:00 | 169.03 | 2025-05-26 11:45:00 | 169.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 09:40:00 | 169.12 | 2025-05-28 09:45:00 | 168.71 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-30 10:30:00 | 163.14 | 2025-05-30 12:35:00 | 162.23 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-05-30 10:30:00 | 163.14 | 2025-05-30 13:35:00 | 163.14 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-03 09:40:00 | 163.55 | 2025-06-03 10:00:00 | 162.84 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-03 09:40:00 | 163.55 | 2025-06-03 15:20:00 | 161.35 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2025-06-04 09:30:00 | 163.65 | 2025-06-04 09:35:00 | 163.19 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-05 11:15:00 | 167.69 | 2025-06-05 13:00:00 | 166.95 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-06-05 11:15:00 | 167.69 | 2025-06-05 15:20:00 | 166.88 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2025-06-12 11:10:00 | 169.50 | 2025-06-12 11:55:00 | 168.79 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-06-12 11:10:00 | 169.50 | 2025-06-12 12:45:00 | 169.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-16 09:30:00 | 163.53 | 2025-06-16 09:45:00 | 162.87 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-06-16 09:30:00 | 163.53 | 2025-06-16 09:50:00 | 163.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-18 09:30:00 | 164.45 | 2025-06-18 09:45:00 | 165.33 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-18 09:30:00 | 164.45 | 2025-06-18 10:30:00 | 164.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-19 10:35:00 | 163.00 | 2025-06-19 10:40:00 | 163.60 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-23 11:10:00 | 163.22 | 2025-06-23 11:20:00 | 162.61 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-25 09:45:00 | 169.29 | 2025-06-25 09:50:00 | 168.75 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-02 11:05:00 | 175.88 | 2025-07-02 11:45:00 | 176.48 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-07 09:30:00 | 179.28 | 2025-07-07 09:40:00 | 178.77 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-08 09:45:00 | 182.24 | 2025-07-08 10:30:00 | 181.73 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-10 10:50:00 | 179.33 | 2025-07-10 11:00:00 | 178.80 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-10 10:50:00 | 179.33 | 2025-07-10 11:05:00 | 179.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:10:00 | 176.40 | 2025-07-11 10:20:00 | 175.37 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-07-11 10:10:00 | 176.40 | 2025-07-11 15:20:00 | 174.42 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2025-07-21 10:30:00 | 179.35 | 2025-07-21 10:50:00 | 178.96 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-24 09:30:00 | 179.33 | 2025-07-24 09:40:00 | 179.02 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-25 09:30:00 | 177.40 | 2025-07-25 09:40:00 | 176.96 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-28 10:50:00 | 175.40 | 2025-07-28 11:05:00 | 174.89 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-06 09:40:00 | 163.83 | 2025-08-06 10:05:00 | 163.10 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-08-06 09:40:00 | 163.83 | 2025-08-06 10:20:00 | 163.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-11 11:10:00 | 161.40 | 2025-08-11 11:40:00 | 160.76 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-11 11:10:00 | 161.40 | 2025-08-11 12:35:00 | 161.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-12 10:00:00 | 165.00 | 2025-08-12 10:15:00 | 164.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-13 10:10:00 | 165.72 | 2025-08-13 10:20:00 | 165.26 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-21 11:05:00 | 164.66 | 2025-08-21 11:15:00 | 164.47 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-08-22 10:35:00 | 165.20 | 2025-08-22 10:40:00 | 165.84 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-08-22 10:35:00 | 165.20 | 2025-08-22 14:25:00 | 166.14 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2025-08-25 10:35:00 | 167.90 | 2025-08-25 10:40:00 | 167.42 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-26 09:30:00 | 165.72 | 2025-08-26 09:40:00 | 166.50 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-26 09:30:00 | 165.72 | 2025-08-26 10:10:00 | 168.70 | TARGET_HIT | 0.50 | 1.80% |
| SELL | retest1 | 2025-08-28 11:15:00 | 160.00 | 2025-08-28 11:20:00 | 159.41 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-08-28 11:15:00 | 160.00 | 2025-08-28 15:20:00 | 153.85 | TARGET_HIT | 0.50 | 3.84% |
| SELL | retest1 | 2025-09-03 09:35:00 | 155.91 | 2025-09-03 10:45:00 | 155.23 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-09-03 09:35:00 | 155.91 | 2025-09-03 15:20:00 | 155.15 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2025-09-08 09:30:00 | 156.19 | 2025-09-08 09:35:00 | 155.67 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-10 10:50:00 | 162.05 | 2025-09-10 11:15:00 | 161.40 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-09-11 11:00:00 | 160.89 | 2025-09-11 11:35:00 | 160.26 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-09-11 11:00:00 | 160.89 | 2025-09-11 12:15:00 | 160.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 10:55:00 | 160.01 | 2025-09-12 11:30:00 | 159.43 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-09-12 10:55:00 | 160.01 | 2025-09-12 12:40:00 | 160.01 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-16 10:20:00 | 168.48 | 2025-09-16 11:50:00 | 167.83 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-09-18 11:10:00 | 165.41 | 2025-09-18 11:25:00 | 165.79 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-19 11:05:00 | 167.09 | 2025-09-19 11:10:00 | 166.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-29 09:45:00 | 164.13 | 2025-09-29 10:05:00 | 163.59 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-01 10:55:00 | 162.00 | 2025-10-01 12:15:00 | 161.28 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-10-01 10:55:00 | 162.00 | 2025-10-01 13:55:00 | 161.75 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-10-08 09:30:00 | 165.00 | 2025-10-08 09:40:00 | 165.77 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-10-08 09:30:00 | 165.00 | 2025-10-08 13:10:00 | 166.31 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2025-10-09 10:25:00 | 168.00 | 2025-10-09 10:40:00 | 168.95 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-09 10:25:00 | 168.00 | 2025-10-09 15:20:00 | 170.00 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2025-10-10 09:30:00 | 171.25 | 2025-10-10 09:40:00 | 172.12 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-10 09:30:00 | 171.25 | 2025-10-10 09:55:00 | 171.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 10:35:00 | 169.64 | 2025-10-14 11:15:00 | 168.96 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-14 10:35:00 | 169.64 | 2025-10-14 15:20:00 | 167.46 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-10-15 09:35:00 | 171.00 | 2025-10-15 10:05:00 | 172.03 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-10-15 09:35:00 | 171.00 | 2025-10-15 15:20:00 | 176.35 | TARGET_HIT | 0.50 | 3.13% |
| SELL | retest1 | 2025-10-31 10:15:00 | 179.11 | 2025-10-31 11:15:00 | 179.58 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-04 09:45:00 | 181.58 | 2025-11-04 11:25:00 | 181.04 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-06 10:15:00 | 179.34 | 2025-11-06 10:35:00 | 178.49 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-11-06 10:15:00 | 179.34 | 2025-11-06 11:00:00 | 179.34 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-10 10:15:00 | 173.64 | 2025-11-10 11:35:00 | 173.05 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-10 10:15:00 | 173.64 | 2025-11-10 14:15:00 | 173.64 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 09:30:00 | 170.98 | 2025-11-11 09:45:00 | 171.58 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-12 10:45:00 | 172.63 | 2025-11-12 11:05:00 | 172.12 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-18 11:15:00 | 173.93 | 2025-11-18 11:45:00 | 174.22 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-19 10:45:00 | 172.30 | 2025-11-19 14:05:00 | 171.70 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-19 10:45:00 | 172.30 | 2025-11-19 15:20:00 | 171.34 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-11-24 10:35:00 | 169.85 | 2025-11-24 11:20:00 | 169.21 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-11-24 10:35:00 | 169.85 | 2025-11-24 15:00:00 | 169.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 10:05:00 | 170.27 | 2025-11-27 10:30:00 | 170.77 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-05 10:15:00 | 176.60 | 2025-12-05 10:20:00 | 176.17 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-09 09:30:00 | 166.61 | 2025-12-09 09:50:00 | 165.58 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-12-09 09:30:00 | 166.61 | 2025-12-09 10:15:00 | 166.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 10:10:00 | 169.25 | 2025-12-11 10:20:00 | 169.89 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-11 10:10:00 | 169.25 | 2025-12-11 11:50:00 | 169.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-16 10:25:00 | 171.63 | 2025-12-16 11:05:00 | 171.29 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-29 11:05:00 | 165.11 | 2025-12-29 11:20:00 | 164.75 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-12-29 11:05:00 | 165.11 | 2025-12-29 11:50:00 | 165.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:30:00 | 165.40 | 2025-12-30 10:35:00 | 165.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-31 11:00:00 | 166.90 | 2025-12-31 11:20:00 | 166.61 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-01-02 10:25:00 | 167.97 | 2026-01-02 10:45:00 | 168.49 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-01-02 10:25:00 | 167.97 | 2026-01-02 15:20:00 | 175.66 | TARGET_HIT | 0.50 | 4.58% |
| SELL | retest1 | 2026-01-06 09:40:00 | 171.81 | 2026-01-06 09:45:00 | 172.34 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-08 11:10:00 | 170.15 | 2026-01-08 11:25:00 | 169.64 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-01-08 11:10:00 | 170.15 | 2026-01-08 15:20:00 | 166.92 | TARGET_HIT | 0.50 | 1.90% |
| SELL | retest1 | 2026-01-09 09:35:00 | 165.14 | 2026-01-09 09:40:00 | 165.68 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-13 10:00:00 | 161.92 | 2026-01-13 12:25:00 | 160.91 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-01-13 10:00:00 | 161.92 | 2026-01-13 15:20:00 | 160.00 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2026-01-14 09:45:00 | 157.80 | 2026-01-14 10:05:00 | 156.98 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-01-14 09:45:00 | 157.80 | 2026-01-14 10:20:00 | 157.53 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-01-19 11:10:00 | 153.80 | 2026-01-19 11:50:00 | 153.22 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-19 11:10:00 | 153.80 | 2026-01-19 15:20:00 | 152.14 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2026-01-20 09:45:00 | 149.83 | 2026-01-20 09:50:00 | 150.43 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-30 10:25:00 | 146.87 | 2026-01-30 11:25:00 | 147.71 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-01-30 10:25:00 | 146.87 | 2026-01-30 12:00:00 | 146.87 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-06 11:10:00 | 151.80 | 2026-02-06 11:30:00 | 152.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-10 10:25:00 | 155.35 | 2026-02-10 10:40:00 | 155.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-11 09:30:00 | 153.87 | 2026-02-11 09:40:00 | 153.35 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-11 09:30:00 | 153.87 | 2026-02-11 10:15:00 | 153.87 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 09:30:00 | 152.30 | 2026-02-16 09:45:00 | 151.79 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-20 09:40:00 | 153.50 | 2026-02-20 09:55:00 | 154.51 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-02-20 09:40:00 | 153.50 | 2026-02-20 12:10:00 | 154.56 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2026-02-26 09:50:00 | 161.20 | 2026-02-26 09:55:00 | 160.78 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-01 11:10:00 | 152.00 | 2026-04-01 11:25:00 | 152.49 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-09 09:50:00 | 155.99 | 2026-04-09 10:15:00 | 156.58 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-10 09:40:00 | 158.40 | 2026-04-10 09:55:00 | 157.91 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-13 09:40:00 | 157.14 | 2026-04-13 09:50:00 | 158.02 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-13 09:40:00 | 157.14 | 2026-04-13 14:05:00 | 161.42 | TARGET_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2026-04-15 10:30:00 | 164.40 | 2026-04-15 10:50:00 | 163.79 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-16 10:25:00 | 165.40 | 2026-04-16 10:40:00 | 164.85 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-28 09:40:00 | 190.50 | 2026-04-28 09:45:00 | 191.71 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-28 09:40:00 | 190.50 | 2026-04-28 09:50:00 | 190.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 11:15:00 | 185.50 | 2026-05-08 12:45:00 | 184.69 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-08 11:15:00 | 185.50 | 2026-05-08 15:20:00 | 184.65 | TARGET_HIT | 0.50 | 0.46% |
