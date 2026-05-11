# Federal Bank Ltd. (FEDERALBNK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-10-31 15:25:00 (27421 bars)
- **Last close:** 236.68
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
| ENTRY1 | 90 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 13 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 77
- **Target hits / Stop hits / Partials:** 13 / 77 / 30
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 14.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 28 | 41.2% | 10 | 40 | 18 | 0.21% | 14.3% |
| BUY @ 2nd Alert (retest1) | 68 | 28 | 41.2% | 10 | 40 | 18 | 0.21% | 14.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 52 | 15 | 28.8% | 3 | 37 | 12 | 0.00% | 0.1% |
| SELL @ 2nd Alert (retest1) | 52 | 15 | 28.8% | 3 | 37 | 12 | 0.00% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 120 | 43 | 35.8% | 13 | 77 | 30 | 0.12% | 14.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 11:10:00 | 162.85 | 163.09 | 0.00 | ORB-short ORB[163.40,164.40] vol=2.8x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 11:30:00 | 162.30 | 163.02 | 0.00 | T1 1.5R @ 162.30 |
| Stop hit — per-position SL triggered | 2024-05-15 11:55:00 | 162.85 | 162.97 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:05:00 | 162.45 | 162.76 | 0.00 | ORB-short ORB[162.80,163.95] vol=4.0x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 11:15:00 | 161.79 | 162.70 | 0.00 | T1 1.5R @ 161.79 |
| Stop hit — per-position SL triggered | 2024-05-16 11:25:00 | 162.45 | 162.67 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:05:00 | 164.55 | 163.95 | 0.00 | ORB-long ORB[163.45,164.20] vol=2.6x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-05-17 10:15:00 | 164.17 | 164.00 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:55:00 | 162.40 | 163.19 | 0.00 | ORB-short ORB[163.05,164.00] vol=4.1x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-05-21 10:20:00 | 162.82 | 163.12 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:45:00 | 162.45 | 163.02 | 0.00 | ORB-short ORB[162.50,163.70] vol=2.2x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 10:25:00 | 161.70 | 162.77 | 0.00 | T1 1.5R @ 161.70 |
| Stop hit — per-position SL triggered | 2024-05-22 10:30:00 | 162.45 | 162.74 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:05:00 | 162.60 | 163.07 | 0.00 | ORB-short ORB[162.65,164.15] vol=2.4x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-05-23 10:20:00 | 163.02 | 163.03 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:35:00 | 164.85 | 164.07 | 0.00 | ORB-long ORB[162.60,164.70] vol=2.1x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-05-27 09:55:00 | 164.30 | 164.38 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:30:00 | 160.50 | 159.55 | 0.00 | ORB-long ORB[158.50,159.80] vol=2.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-05-29 10:05:00 | 160.01 | 159.88 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-05-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 09:55:00 | 161.65 | 161.10 | 0.00 | ORB-long ORB[160.15,161.50] vol=1.5x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-05-31 10:40:00 | 161.15 | 161.33 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:40:00 | 166.00 | 165.48 | 0.00 | ORB-long ORB[164.25,165.90] vol=2.2x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:15:00 | 166.80 | 165.84 | 0.00 | T1 1.5R @ 166.80 |
| Stop hit — per-position SL triggered | 2024-06-07 10:20:00 | 166.00 | 165.86 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 10:40:00 | 163.00 | 164.73 | 0.00 | ORB-short ORB[164.81,166.65] vol=1.5x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-06-10 11:20:00 | 163.59 | 164.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:30:00 | 167.85 | 166.00 | 0.00 | ORB-long ORB[164.70,166.34] vol=1.9x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-06-11 11:50:00 | 167.29 | 167.00 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:45:00 | 169.58 | 167.79 | 0.00 | ORB-long ORB[166.52,167.80] vol=3.3x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:50:00 | 170.46 | 168.69 | 0.00 | T1 1.5R @ 170.46 |
| Target hit | 2024-06-12 15:20:00 | 173.95 | 172.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-06-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 11:00:00 | 175.20 | 174.18 | 0.00 | ORB-long ORB[173.25,174.90] vol=2.8x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:20:00 | 175.81 | 174.60 | 0.00 | T1 1.5R @ 175.81 |
| Stop hit — per-position SL triggered | 2024-06-18 12:40:00 | 175.20 | 174.91 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 177.40 | 177.86 | 0.00 | ORB-short ORB[177.96,179.73] vol=1.6x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-06-21 11:05:00 | 177.86 | 177.82 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-06-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 11:00:00 | 178.10 | 177.70 | 0.00 | ORB-long ORB[175.46,177.00] vol=1.5x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 12:00:00 | 178.88 | 177.93 | 0.00 | T1 1.5R @ 178.88 |
| Stop hit — per-position SL triggered | 2024-06-27 12:15:00 | 178.10 | 178.02 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:50:00 | 177.03 | 177.67 | 0.00 | ORB-short ORB[177.21,178.50] vol=2.0x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-07-02 11:00:00 | 177.45 | 177.65 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:50:00 | 182.45 | 181.41 | 0.00 | ORB-long ORB[179.94,181.89] vol=3.6x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:00:00 | 183.41 | 181.92 | 0.00 | T1 1.5R @ 183.41 |
| Target hit | 2024-07-05 15:20:00 | 186.21 | 184.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 185.69 | 186.68 | 0.00 | ORB-short ORB[186.18,187.50] vol=1.5x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-07-08 09:45:00 | 186.25 | 186.61 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 11:00:00 | 190.26 | 189.79 | 0.00 | ORB-long ORB[188.23,189.65] vol=10.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 189.67 | 189.80 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:55:00 | 194.05 | 194.82 | 0.00 | ORB-short ORB[194.10,195.85] vol=1.9x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-07-15 11:20:00 | 194.64 | 194.77 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 10:05:00 | 197.62 | 196.76 | 0.00 | ORB-long ORB[195.00,197.60] vol=1.5x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-07-18 10:30:00 | 197.03 | 196.88 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:15:00 | 193.80 | 194.80 | 0.00 | ORB-short ORB[194.55,197.00] vol=2.2x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-07-19 10:25:00 | 194.31 | 194.77 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:05:00 | 193.18 | 192.05 | 0.00 | ORB-long ORB[190.10,191.95] vol=1.8x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-07-22 10:20:00 | 192.55 | 192.28 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-07-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:35:00 | 200.45 | 199.31 | 0.00 | ORB-long ORB[198.39,200.06] vol=1.7x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 12:45:00 | 201.90 | 199.97 | 0.00 | T1 1.5R @ 201.90 |
| Target hit | 2024-07-25 15:20:00 | 204.73 | 201.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2024-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:55:00 | 201.24 | 200.00 | 0.00 | ORB-long ORB[198.27,200.48] vol=1.8x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-07-29 11:05:00 | 200.58 | 200.08 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:40:00 | 202.61 | 201.55 | 0.00 | ORB-long ORB[199.50,201.57] vol=1.9x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-07-30 09:45:00 | 201.89 | 201.66 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 11:05:00 | 202.43 | 202.42 | 0.00 | ORB-long ORB[201.13,202.00] vol=1.7x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-08-01 11:45:00 | 201.88 | 202.41 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 11:15:00 | 191.30 | 193.15 | 0.00 | ORB-short ORB[192.10,194.31] vol=3.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-08-05 11:35:00 | 192.09 | 192.94 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:40:00 | 194.79 | 193.60 | 0.00 | ORB-long ORB[192.60,193.76] vol=2.4x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-08-08 11:00:00 | 194.15 | 193.74 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:40:00 | 197.42 | 196.78 | 0.00 | ORB-long ORB[195.00,197.37] vol=1.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-08-12 10:45:00 | 196.77 | 196.79 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 10:45:00 | 204.21 | 202.65 | 0.00 | ORB-long ORB[202.03,203.04] vol=3.8x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 11:05:00 | 205.05 | 203.22 | 0.00 | T1 1.5R @ 205.05 |
| Stop hit — per-position SL triggered | 2024-08-13 11:10:00 | 204.21 | 203.26 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 11:10:00 | 204.10 | 202.57 | 0.00 | ORB-long ORB[201.80,203.50] vol=2.7x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-08-20 11:35:00 | 203.70 | 202.69 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-08-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 09:50:00 | 200.95 | 202.35 | 0.00 | ORB-short ORB[202.71,204.10] vol=2.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-08-21 09:55:00 | 201.45 | 202.27 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-08-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:35:00 | 201.35 | 202.04 | 0.00 | ORB-short ORB[201.79,203.14] vol=1.7x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-08-23 09:55:00 | 201.76 | 201.90 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-08-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 11:10:00 | 198.20 | 198.83 | 0.00 | ORB-short ORB[198.30,199.61] vol=1.8x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 12:35:00 | 197.64 | 198.46 | 0.00 | T1 1.5R @ 197.64 |
| Target hit | 2024-08-27 15:20:00 | 196.91 | 197.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 194.78 | 195.32 | 0.00 | ORB-short ORB[195.41,197.49] vol=6.5x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 09:35:00 | 194.08 | 195.10 | 0.00 | T1 1.5R @ 194.08 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 194.78 | 195.05 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-08-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:00:00 | 197.27 | 196.86 | 0.00 | ORB-long ORB[195.87,197.16] vol=2.3x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-08-30 10:05:00 | 196.83 | 196.85 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 10:30:00 | 189.68 | 190.52 | 0.00 | ORB-short ORB[189.90,191.75] vol=2.4x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-09-04 10:50:00 | 190.28 | 190.42 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-09-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:40:00 | 182.21 | 183.30 | 0.00 | ORB-short ORB[183.29,184.79] vol=1.9x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-09-12 13:05:00 | 182.70 | 182.77 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:10:00 | 186.25 | 185.23 | 0.00 | ORB-long ORB[183.51,185.88] vol=1.7x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 11:15:00 | 187.11 | 185.98 | 0.00 | T1 1.5R @ 187.11 |
| Target hit | 2024-09-13 13:25:00 | 186.41 | 186.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2024-09-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:40:00 | 187.68 | 187.06 | 0.00 | ORB-long ORB[185.86,186.98] vol=4.4x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-09-16 09:45:00 | 187.18 | 187.12 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-09-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:45:00 | 185.27 | 184.68 | 0.00 | ORB-long ORB[183.71,184.97] vol=1.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-09-18 10:50:00 | 184.83 | 184.69 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-09-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:40:00 | 186.27 | 186.85 | 0.00 | ORB-short ORB[186.97,187.75] vol=2.0x ATR=0.36 |
| Stop hit — per-position SL triggered | 2024-09-24 11:10:00 | 186.63 | 186.78 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:00:00 | 189.38 | 190.04 | 0.00 | ORB-short ORB[190.37,191.33] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-09-25 11:40:00 | 189.79 | 189.87 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-09-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:20:00 | 194.83 | 192.96 | 0.00 | ORB-long ORB[191.01,192.56] vol=1.5x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-09-27 10:30:00 | 194.25 | 193.15 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 11:00:00 | 196.77 | 195.17 | 0.00 | ORB-long ORB[193.30,196.00] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 196.14 | 195.32 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-10-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 09:40:00 | 195.00 | 194.27 | 0.00 | ORB-long ORB[192.36,194.91] vol=1.7x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:45:00 | 196.21 | 194.73 | 0.00 | T1 1.5R @ 196.21 |
| Target hit | 2024-10-04 12:25:00 | 196.12 | 196.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2024-10-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:40:00 | 186.14 | 184.82 | 0.00 | ORB-long ORB[182.35,184.66] vol=2.0x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 13:10:00 | 187.38 | 185.51 | 0.00 | T1 1.5R @ 187.38 |
| Target hit | 2024-10-08 14:40:00 | 186.49 | 186.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2024-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:30:00 | 187.58 | 186.92 | 0.00 | ORB-long ORB[185.70,187.40] vol=1.5x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-10-11 09:35:00 | 187.13 | 186.96 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 191.20 | 190.22 | 0.00 | ORB-long ORB[188.56,190.74] vol=4.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-10-14 09:35:00 | 190.55 | 190.38 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:55:00 | 193.80 | 193.83 | 0.00 | ORB-short ORB[193.92,195.20] vol=1.8x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:25:00 | 193.07 | 193.57 | 0.00 | T1 1.5R @ 193.07 |
| Target hit | 2024-10-17 11:35:00 | 193.70 | 193.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2024-10-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:45:00 | 185.70 | 187.88 | 0.00 | ORB-short ORB[187.62,190.32] vol=1.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:05:00 | 184.47 | 186.82 | 0.00 | T1 1.5R @ 184.47 |
| Stop hit — per-position SL triggered | 2024-10-25 11:35:00 | 185.70 | 185.32 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-10-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:55:00 | 202.04 | 200.25 | 0.00 | ORB-long ORB[199.00,201.30] vol=1.9x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 201.22 | 200.47 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-10-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:00:00 | 205.60 | 204.14 | 0.00 | ORB-long ORB[201.69,204.00] vol=2.8x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-10-31 10:05:00 | 204.93 | 204.23 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-11-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:30:00 | 206.78 | 205.84 | 0.00 | ORB-long ORB[204.51,206.48] vol=1.8x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-11-08 09:35:00 | 206.19 | 205.87 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-11-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 10:30:00 | 209.27 | 208.54 | 0.00 | ORB-long ORB[207.66,209.20] vol=1.6x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-11-12 10:35:00 | 208.69 | 208.53 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-11-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 11:00:00 | 207.52 | 205.95 | 0.00 | ORB-long ORB[204.21,207.03] vol=2.6x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 11:20:00 | 208.72 | 206.19 | 0.00 | T1 1.5R @ 208.72 |
| Stop hit — per-position SL triggered | 2024-11-21 11:25:00 | 207.52 | 206.22 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-11-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 11:10:00 | 212.67 | 210.76 | 0.00 | ORB-long ORB[210.25,212.00] vol=3.5x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-11-25 11:40:00 | 212.08 | 211.27 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-11-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 10:20:00 | 210.80 | 211.47 | 0.00 | ORB-short ORB[211.12,213.00] vol=2.3x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-11-26 10:35:00 | 211.41 | 211.41 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-11-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:45:00 | 210.74 | 211.85 | 0.00 | ORB-short ORB[211.20,213.01] vol=2.8x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-11-28 11:00:00 | 211.29 | 211.76 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-12-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:50:00 | 212.11 | 211.16 | 0.00 | ORB-long ORB[209.27,210.90] vol=2.8x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-12-04 09:55:00 | 211.64 | 211.25 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 212.00 | 212.97 | 0.00 | ORB-short ORB[212.28,215.10] vol=2.7x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 212.75 | 212.98 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 215.50 | 215.04 | 0.00 | ORB-long ORB[214.25,215.40] vol=1.9x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-12-12 09:40:00 | 215.08 | 214.99 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 209.98 | 211.27 | 0.00 | ORB-short ORB[211.70,213.50] vol=2.2x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 210.47 | 210.90 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:25:00 | 205.60 | 207.34 | 0.00 | ORB-short ORB[207.93,211.00] vol=2.0x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:50:00 | 204.62 | 206.59 | 0.00 | T1 1.5R @ 204.62 |
| Target hit | 2024-12-18 15:20:00 | 200.19 | 202.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2024-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 10:55:00 | 197.57 | 196.36 | 0.00 | ORB-long ORB[194.88,196.94] vol=2.0x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-12-23 12:35:00 | 196.92 | 196.64 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-12-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:00:00 | 198.49 | 197.10 | 0.00 | ORB-long ORB[195.55,197.88] vol=1.5x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 10:20:00 | 199.38 | 197.81 | 0.00 | T1 1.5R @ 199.38 |
| Target hit | 2024-12-30 12:45:00 | 199.84 | 200.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 69 — BUY (started 2025-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:35:00 | 201.89 | 200.90 | 0.00 | ORB-long ORB[199.56,201.12] vol=1.6x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:20:00 | 202.70 | 201.62 | 0.00 | T1 1.5R @ 202.70 |
| Target hit | 2025-01-02 15:20:00 | 206.27 | 203.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2025-01-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 09:50:00 | 197.60 | 198.89 | 0.00 | ORB-short ORB[197.76,200.49] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-01-07 10:10:00 | 198.47 | 198.44 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:10:00 | 193.54 | 195.44 | 0.00 | ORB-short ORB[195.36,197.60] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-01-08 11:20:00 | 194.10 | 195.12 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-01-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 11:00:00 | 193.54 | 194.49 | 0.00 | ORB-short ORB[194.38,196.00] vol=4.0x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:15:00 | 192.69 | 194.35 | 0.00 | T1 1.5R @ 192.69 |
| Stop hit — per-position SL triggered | 2025-01-09 12:35:00 | 193.54 | 193.71 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:50:00 | 195.68 | 196.59 | 0.00 | ORB-short ORB[195.99,197.84] vol=1.5x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-01-16 11:00:00 | 196.24 | 196.50 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-01-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:20:00 | 193.04 | 193.56 | 0.00 | ORB-short ORB[193.53,195.98] vol=3.4x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-01-17 10:25:00 | 193.60 | 193.59 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-01-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:50:00 | 190.28 | 191.34 | 0.00 | ORB-short ORB[191.56,193.44] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-01-22 09:55:00 | 190.89 | 191.30 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 11:15:00 | 182.54 | 183.37 | 0.00 | ORB-short ORB[183.55,185.83] vol=2.5x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 12:05:00 | 181.84 | 183.19 | 0.00 | T1 1.5R @ 181.84 |
| Stop hit — per-position SL triggered | 2025-02-21 12:55:00 | 182.54 | 183.01 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:10:00 | 179.85 | 177.34 | 0.00 | ORB-long ORB[175.80,177.90] vol=1.8x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-03-11 13:15:00 | 179.25 | 178.35 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:45:00 | 178.35 | 179.37 | 0.00 | ORB-short ORB[178.84,181.15] vol=3.2x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:25:00 | 177.56 | 179.11 | 0.00 | T1 1.5R @ 177.56 |
| Stop hit — per-position SL triggered | 2025-03-12 13:15:00 | 178.35 | 178.60 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:15:00 | 179.65 | 178.31 | 0.00 | ORB-long ORB[176.90,178.47] vol=2.3x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:55:00 | 180.30 | 178.72 | 0.00 | T1 1.5R @ 180.30 |
| Stop hit — per-position SL triggered | 2025-03-18 12:15:00 | 179.65 | 178.82 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:05:00 | 182.97 | 181.64 | 0.00 | ORB-long ORB[180.56,182.16] vol=1.8x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:15:00 | 183.79 | 181.93 | 0.00 | T1 1.5R @ 183.79 |
| Target hit | 2025-03-19 15:20:00 | 186.25 | 184.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2025-03-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:00:00 | 187.92 | 187.36 | 0.00 | ORB-long ORB[185.80,187.85] vol=3.4x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-03-21 10:10:00 | 187.44 | 187.43 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-03-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 09:45:00 | 194.10 | 193.00 | 0.00 | ORB-long ORB[191.56,193.35] vol=1.9x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 12:10:00 | 195.08 | 193.84 | 0.00 | T1 1.5R @ 195.08 |
| Target hit | 2025-03-27 15:20:00 | 199.07 | 196.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2025-04-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:45:00 | 189.86 | 190.92 | 0.00 | ORB-short ORB[190.13,191.99] vol=1.8x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 10:05:00 | 189.01 | 190.32 | 0.00 | T1 1.5R @ 189.01 |
| Stop hit — per-position SL triggered | 2025-04-03 10:10:00 | 189.86 | 190.29 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-04-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 10:55:00 | 192.00 | 192.52 | 0.00 | ORB-short ORB[192.57,194.64] vol=2.0x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-04-08 11:00:00 | 192.71 | 192.53 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-04-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:55:00 | 188.10 | 190.03 | 0.00 | ORB-short ORB[189.23,191.30] vol=2.0x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-04-09 11:20:00 | 188.92 | 189.34 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2025-04-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-11 10:25:00 | 190.48 | 191.14 | 0.00 | ORB-short ORB[190.52,192.95] vol=1.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-04-11 11:30:00 | 191.06 | 190.96 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-04-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:25:00 | 200.72 | 198.84 | 0.00 | ORB-long ORB[195.10,198.11] vol=2.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:20:00 | 201.81 | 199.46 | 0.00 | T1 1.5R @ 201.81 |
| Stop hit — per-position SL triggered | 2025-04-21 11:50:00 | 200.72 | 199.77 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 199.47 | 200.50 | 0.00 | ORB-short ORB[200.01,202.99] vol=1.8x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-04-23 09:40:00 | 200.15 | 200.47 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:35:00 | 202.10 | 200.81 | 0.00 | ORB-long ORB[199.29,200.81] vol=2.0x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:55:00 | 202.83 | 201.77 | 0.00 | T1 1.5R @ 202.83 |
| Stop hit — per-position SL triggered | 2025-04-24 11:20:00 | 202.10 | 201.91 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2025-05-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:40:00 | 191.85 | 191.52 | 0.00 | ORB-long ORB[189.92,191.19] vol=3.4x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-05-05 12:25:00 | 191.34 | 191.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 11:10:00 | 162.85 | 2024-05-15 11:30:00 | 162.30 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-05-15 11:10:00 | 162.85 | 2024-05-15 11:55:00 | 162.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 11:05:00 | 162.45 | 2024-05-16 11:15:00 | 161.79 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-16 11:05:00 | 162.45 | 2024-05-16 11:25:00 | 162.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 10:05:00 | 164.55 | 2024-05-17 10:15:00 | 164.17 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-05-21 09:55:00 | 162.40 | 2024-05-21 10:20:00 | 162.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-22 09:45:00 | 162.45 | 2024-05-22 10:25:00 | 161.70 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-05-22 09:45:00 | 162.45 | 2024-05-22 10:30:00 | 162.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-23 10:05:00 | 162.60 | 2024-05-23 10:20:00 | 163.02 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-27 09:35:00 | 164.85 | 2024-05-27 09:55:00 | 164.30 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-05-29 09:30:00 | 160.50 | 2024-05-29 10:05:00 | 160.01 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-31 09:55:00 | 161.65 | 2024-05-31 10:40:00 | 161.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-07 09:40:00 | 166.00 | 2024-06-07 10:15:00 | 166.80 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-06-07 09:40:00 | 166.00 | 2024-06-07 10:20:00 | 166.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-10 10:40:00 | 163.00 | 2024-06-10 11:20:00 | 163.59 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-11 10:30:00 | 167.85 | 2024-06-11 11:50:00 | 167.29 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-12 10:45:00 | 169.58 | 2024-06-12 10:50:00 | 170.46 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-06-12 10:45:00 | 169.58 | 2024-06-12 15:20:00 | 173.95 | TARGET_HIT | 0.50 | 2.58% |
| BUY | retest1 | 2024-06-18 11:00:00 | 175.20 | 2024-06-18 11:20:00 | 175.81 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-06-18 11:00:00 | 175.20 | 2024-06-18 12:40:00 | 175.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 10:45:00 | 177.40 | 2024-06-21 11:05:00 | 177.86 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-27 11:00:00 | 178.10 | 2024-06-27 12:00:00 | 178.88 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-06-27 11:00:00 | 178.10 | 2024-06-27 12:15:00 | 178.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 10:50:00 | 177.03 | 2024-07-02 11:00:00 | 177.45 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-05 09:50:00 | 182.45 | 2024-07-05 10:00:00 | 183.41 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-05 09:50:00 | 182.45 | 2024-07-05 15:20:00 | 186.21 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2024-07-08 09:40:00 | 185.69 | 2024-07-08 09:45:00 | 186.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-11 11:00:00 | 190.26 | 2024-07-11 11:15:00 | 189.67 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-15 10:55:00 | 194.05 | 2024-07-15 11:20:00 | 194.64 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-18 10:05:00 | 197.62 | 2024-07-18 10:30:00 | 197.03 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-19 10:15:00 | 193.80 | 2024-07-19 10:25:00 | 194.31 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-22 10:05:00 | 193.18 | 2024-07-22 10:20:00 | 192.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-25 10:35:00 | 200.45 | 2024-07-25 12:45:00 | 201.90 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-07-25 10:35:00 | 200.45 | 2024-07-25 15:20:00 | 204.73 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2024-07-29 10:55:00 | 201.24 | 2024-07-29 11:05:00 | 200.58 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-30 09:40:00 | 202.61 | 2024-07-30 09:45:00 | 201.89 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-01 11:05:00 | 202.43 | 2024-08-01 11:45:00 | 201.88 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-05 11:15:00 | 191.30 | 2024-08-05 11:35:00 | 192.09 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-08 10:40:00 | 194.79 | 2024-08-08 11:00:00 | 194.15 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-12 10:40:00 | 197.42 | 2024-08-12 10:45:00 | 196.77 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-13 10:45:00 | 204.21 | 2024-08-13 11:05:00 | 205.05 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-13 10:45:00 | 204.21 | 2024-08-13 11:10:00 | 204.21 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 11:10:00 | 204.10 | 2024-08-20 11:35:00 | 203.70 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-21 09:50:00 | 200.95 | 2024-08-21 09:55:00 | 201.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-23 09:35:00 | 201.35 | 2024-08-23 09:55:00 | 201.76 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-27 11:10:00 | 198.20 | 2024-08-27 12:35:00 | 197.64 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-08-27 11:10:00 | 198.20 | 2024-08-27 15:20:00 | 196.91 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-08-28 09:30:00 | 194.78 | 2024-08-28 09:35:00 | 194.08 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-28 09:30:00 | 194.78 | 2024-08-28 09:40:00 | 194.78 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 10:00:00 | 197.27 | 2024-08-30 10:05:00 | 196.83 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-04 10:30:00 | 189.68 | 2024-09-04 10:50:00 | 190.28 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-12 10:40:00 | 182.21 | 2024-09-12 13:05:00 | 182.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-13 10:10:00 | 186.25 | 2024-09-13 11:15:00 | 187.11 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-09-13 10:10:00 | 186.25 | 2024-09-13 13:25:00 | 186.41 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-09-16 09:40:00 | 187.68 | 2024-09-16 09:45:00 | 187.18 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-18 10:45:00 | 185.27 | 2024-09-18 10:50:00 | 184.83 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-24 10:40:00 | 186.27 | 2024-09-24 11:10:00 | 186.63 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-09-25 11:00:00 | 189.38 | 2024-09-25 11:40:00 | 189.79 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-27 10:20:00 | 194.83 | 2024-09-27 10:30:00 | 194.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-03 11:00:00 | 196.77 | 2024-10-03 11:15:00 | 196.14 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-04 09:40:00 | 195.00 | 2024-10-04 09:45:00 | 196.21 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-04 09:40:00 | 195.00 | 2024-10-04 12:25:00 | 196.12 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-08 10:40:00 | 186.14 | 2024-10-08 13:10:00 | 187.38 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-10-08 10:40:00 | 186.14 | 2024-10-08 14:40:00 | 186.49 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2024-10-11 09:30:00 | 187.58 | 2024-10-11 09:35:00 | 187.13 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-10-14 09:30:00 | 191.20 | 2024-10-14 09:35:00 | 190.55 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-17 10:55:00 | 193.80 | 2024-10-17 11:25:00 | 193.07 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-17 10:55:00 | 193.80 | 2024-10-17 11:35:00 | 193.70 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-10-25 09:45:00 | 185.70 | 2024-10-25 10:05:00 | 184.47 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-10-25 09:45:00 | 185.70 | 2024-10-25 11:35:00 | 185.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-30 10:55:00 | 202.04 | 2024-10-30 11:15:00 | 201.22 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-10-31 10:00:00 | 205.60 | 2024-10-31 10:05:00 | 204.93 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-08 09:30:00 | 206.78 | 2024-11-08 09:35:00 | 206.19 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-12 10:30:00 | 209.27 | 2024-11-12 10:35:00 | 208.69 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-21 11:00:00 | 207.52 | 2024-11-21 11:20:00 | 208.72 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-11-21 11:00:00 | 207.52 | 2024-11-21 11:25:00 | 207.52 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 11:10:00 | 212.67 | 2024-11-25 11:40:00 | 212.08 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-26 10:20:00 | 210.80 | 2024-11-26 10:35:00 | 211.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-28 10:45:00 | 210.74 | 2024-11-28 11:00:00 | 211.29 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-04 09:50:00 | 212.11 | 2024-12-04 09:55:00 | 211.64 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-06 10:05:00 | 212.00 | 2024-12-06 10:20:00 | 212.75 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-12 09:35:00 | 215.50 | 2024-12-12 09:40:00 | 215.08 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-13 10:30:00 | 209.98 | 2024-12-13 11:10:00 | 210.47 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-18 10:25:00 | 205.60 | 2024-12-18 10:50:00 | 204.62 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-18 10:25:00 | 205.60 | 2024-12-18 15:20:00 | 200.19 | TARGET_HIT | 0.50 | 2.63% |
| BUY | retest1 | 2024-12-23 10:55:00 | 197.57 | 2024-12-23 12:35:00 | 196.92 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-30 10:00:00 | 198.49 | 2024-12-30 10:20:00 | 199.38 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-12-30 10:00:00 | 198.49 | 2024-12-30 12:45:00 | 199.84 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2025-01-02 09:35:00 | 201.89 | 2025-01-02 10:20:00 | 202.70 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-02 09:35:00 | 201.89 | 2025-01-02 15:20:00 | 206.27 | TARGET_HIT | 0.50 | 2.17% |
| SELL | retest1 | 2025-01-07 09:50:00 | 197.60 | 2025-01-07 10:10:00 | 198.47 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-01-08 11:10:00 | 193.54 | 2025-01-08 11:20:00 | 194.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-09 11:00:00 | 193.54 | 2025-01-09 11:15:00 | 192.69 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-09 11:00:00 | 193.54 | 2025-01-09 12:35:00 | 193.54 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-16 10:50:00 | 195.68 | 2025-01-16 11:00:00 | 196.24 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-17 10:20:00 | 193.04 | 2025-01-17 10:25:00 | 193.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-22 09:50:00 | 190.28 | 2025-01-22 09:55:00 | 190.89 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-21 11:15:00 | 182.54 | 2025-02-21 12:05:00 | 181.84 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-02-21 11:15:00 | 182.54 | 2025-02-21 12:55:00 | 182.54 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-11 11:10:00 | 179.85 | 2025-03-11 13:15:00 | 179.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-12 10:45:00 | 178.35 | 2025-03-12 11:25:00 | 177.56 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-03-12 10:45:00 | 178.35 | 2025-03-12 13:15:00 | 178.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 11:15:00 | 179.65 | 2025-03-18 11:55:00 | 180.30 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-03-18 11:15:00 | 179.65 | 2025-03-18 12:15:00 | 179.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 10:05:00 | 182.97 | 2025-03-19 10:15:00 | 183.79 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-03-19 10:05:00 | 182.97 | 2025-03-19 15:20:00 | 186.25 | TARGET_HIT | 0.50 | 1.79% |
| BUY | retest1 | 2025-03-21 10:00:00 | 187.92 | 2025-03-21 10:10:00 | 187.44 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-03-27 09:45:00 | 194.10 | 2025-03-27 12:10:00 | 195.08 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-03-27 09:45:00 | 194.10 | 2025-03-27 15:20:00 | 199.07 | TARGET_HIT | 0.50 | 2.56% |
| SELL | retest1 | 2025-04-03 09:45:00 | 189.86 | 2025-04-03 10:05:00 | 189.01 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-04-03 09:45:00 | 189.86 | 2025-04-03 10:10:00 | 189.86 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-08 10:55:00 | 192.00 | 2025-04-08 11:00:00 | 192.71 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-09 10:55:00 | 188.10 | 2025-04-09 11:20:00 | 188.92 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-04-11 10:25:00 | 190.48 | 2025-04-11 11:30:00 | 191.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-21 10:25:00 | 200.72 | 2025-04-21 11:20:00 | 201.81 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-04-21 10:25:00 | 200.72 | 2025-04-21 11:50:00 | 200.72 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-23 09:35:00 | 199.47 | 2025-04-23 09:40:00 | 200.15 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-24 09:35:00 | 202.10 | 2025-04-24 10:55:00 | 202.83 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-04-24 09:35:00 | 202.10 | 2025-04-24 11:20:00 | 202.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 10:40:00 | 191.85 | 2025-05-05 12:25:00 | 191.34 | STOP_HIT | 1.00 | -0.26% |
