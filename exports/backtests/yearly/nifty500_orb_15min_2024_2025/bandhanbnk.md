# Bandhan Bank Ltd. (BANDHANBNK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 206.25
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
| ENTRY1 | 70 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 18 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 52
- **Target hits / Stop hits / Partials:** 18 / 52 / 35
- **Avg / median % per leg:** 0.17% / 0.05%
- **Sum % (uncompounded):** 17.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 21 | 43.8% | 8 | 27 | 13 | 0.07% | 3.3% |
| BUY @ 2nd Alert (retest1) | 48 | 21 | 43.8% | 8 | 27 | 13 | 0.07% | 3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 32 | 56.1% | 10 | 25 | 22 | 0.25% | 14.1% |
| SELL @ 2nd Alert (retest1) | 57 | 32 | 56.1% | 10 | 25 | 22 | 0.25% | 14.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 105 | 53 | 50.5% | 18 | 52 | 35 | 0.17% | 17.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:25:00 | 190.50 | 189.42 | 0.00 | ORB-long ORB[187.95,190.45] vol=1.6x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-05-14 10:35:00 | 189.79 | 189.45 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:40:00 | 189.00 | 187.91 | 0.00 | ORB-long ORB[186.65,188.30] vol=2.3x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-05-27 10:55:00 | 188.29 | 187.96 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:45:00 | 191.90 | 190.36 | 0.00 | ORB-long ORB[188.30,190.50] vol=2.2x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-05-29 09:50:00 | 191.25 | 190.64 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:10:00 | 188.80 | 190.62 | 0.00 | ORB-short ORB[190.60,191.90] vol=1.8x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:30:00 | 187.98 | 190.32 | 0.00 | T1 1.5R @ 187.98 |
| Stop hit — per-position SL triggered | 2024-05-30 13:00:00 | 188.80 | 189.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:55:00 | 194.30 | 192.71 | 0.00 | ORB-long ORB[190.90,193.45] vol=2.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-06-06 11:10:00 | 193.59 | 192.97 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:55:00 | 194.15 | 192.94 | 0.00 | ORB-long ORB[190.65,193.45] vol=2.6x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-06-07 10:15:00 | 193.45 | 193.20 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 11:10:00 | 198.63 | 196.98 | 0.00 | ORB-long ORB[195.54,198.25] vol=5.9x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 11:15:00 | 199.76 | 198.00 | 0.00 | T1 1.5R @ 199.76 |
| Target hit | 2024-06-10 14:15:00 | 199.41 | 199.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:35:00 | 200.08 | 198.75 | 0.00 | ORB-long ORB[197.06,198.94] vol=2.4x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-06-12 10:40:00 | 199.29 | 199.49 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 10:50:00 | 195.86 | 197.60 | 0.00 | ORB-short ORB[197.57,198.85] vol=1.9x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:30:00 | 195.04 | 196.91 | 0.00 | T1 1.5R @ 195.04 |
| Stop hit — per-position SL triggered | 2024-06-13 12:00:00 | 195.86 | 196.72 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 09:35:00 | 193.87 | 194.91 | 0.00 | ORB-short ORB[194.40,195.70] vol=2.0x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-06-14 09:45:00 | 194.43 | 194.78 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:55:00 | 196.00 | 194.69 | 0.00 | ORB-long ORB[193.51,195.00] vol=2.4x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:45:00 | 196.77 | 195.40 | 0.00 | T1 1.5R @ 196.77 |
| Target hit | 2024-06-18 15:20:00 | 198.21 | 196.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-06-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 10:05:00 | 200.45 | 199.46 | 0.00 | ORB-long ORB[198.10,200.22] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-06-19 10:10:00 | 199.77 | 199.46 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:45:00 | 206.39 | 204.86 | 0.00 | ORB-long ORB[203.05,205.36] vol=1.5x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:50:00 | 207.47 | 205.37 | 0.00 | T1 1.5R @ 207.47 |
| Target hit | 2024-06-26 12:40:00 | 206.82 | 206.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2024-06-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:50:00 | 202.85 | 204.19 | 0.00 | ORB-short ORB[204.20,205.95] vol=4.2x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 11:25:00 | 201.82 | 203.71 | 0.00 | T1 1.5R @ 201.82 |
| Target hit | 2024-06-27 15:20:00 | 200.35 | 202.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-07-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 11:10:00 | 205.38 | 204.67 | 0.00 | ORB-long ORB[203.77,205.30] vol=2.4x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 12:05:00 | 206.00 | 205.00 | 0.00 | T1 1.5R @ 206.00 |
| Stop hit — per-position SL triggered | 2024-07-01 12:30:00 | 205.38 | 205.06 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:40:00 | 192.00 | 192.86 | 0.00 | ORB-short ORB[192.50,193.88] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-07-15 10:05:00 | 192.51 | 192.70 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:00:00 | 192.66 | 194.05 | 0.00 | ORB-short ORB[193.11,195.90] vol=1.6x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 191.59 | 193.68 | 0.00 | T1 1.5R @ 191.59 |
| Stop hit — per-position SL triggered | 2024-07-19 10:20:00 | 192.66 | 193.65 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 197.50 | 196.67 | 0.00 | ORB-long ORB[195.87,197.39] vol=3.9x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:40:00 | 198.40 | 196.93 | 0.00 | T1 1.5R @ 198.40 |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 197.50 | 197.26 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:45:00 | 201.50 | 202.30 | 0.00 | ORB-short ORB[201.55,203.89] vol=1.6x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-08-08 11:10:00 | 202.21 | 202.25 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 194.68 | 193.90 | 0.00 | ORB-long ORB[192.60,194.38] vol=2.1x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-08-20 09:35:00 | 194.27 | 193.98 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 11:15:00 | 202.10 | 200.46 | 0.00 | ORB-long ORB[198.84,201.24] vol=1.8x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-08-27 11:30:00 | 201.71 | 200.58 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:55:00 | 195.35 | 195.90 | 0.00 | ORB-short ORB[195.50,197.44] vol=2.0x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:10:00 | 194.65 | 195.71 | 0.00 | T1 1.5R @ 194.65 |
| Target hit | 2024-08-29 15:05:00 | 194.34 | 193.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2024-09-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 11:10:00 | 203.94 | 202.44 | 0.00 | ORB-long ORB[200.71,202.40] vol=3.2x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 203.38 | 202.48 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:55:00 | 198.93 | 201.01 | 0.00 | ORB-short ORB[201.25,203.60] vol=1.9x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 197.76 | 200.06 | 0.00 | T1 1.5R @ 197.76 |
| Stop hit — per-position SL triggered | 2024-09-06 11:45:00 | 198.93 | 199.22 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:15:00 | 206.64 | 206.97 | 0.00 | ORB-short ORB[207.12,209.38] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-09-16 11:40:00 | 207.20 | 206.94 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 209.30 | 208.08 | 0.00 | ORB-long ORB[206.55,209.09] vol=3.8x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-09-17 10:15:00 | 208.46 | 208.49 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:40:00 | 208.59 | 211.90 | 0.00 | ORB-short ORB[213.00,215.40] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-09-19 10:50:00 | 209.50 | 211.65 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 11:05:00 | 207.97 | 210.05 | 0.00 | ORB-short ORB[210.24,212.82] vol=2.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-09-24 11:10:00 | 208.58 | 209.98 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:10:00 | 205.37 | 206.26 | 0.00 | ORB-short ORB[205.63,207.64] vol=1.8x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:35:00 | 204.46 | 206.07 | 0.00 | T1 1.5R @ 204.46 |
| Target hit | 2024-09-25 15:10:00 | 205.07 | 204.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — SELL (started 2024-09-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 11:05:00 | 203.15 | 203.68 | 0.00 | ORB-short ORB[203.86,206.00] vol=1.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-09-26 11:40:00 | 203.59 | 203.68 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 09:45:00 | 196.73 | 197.73 | 0.00 | ORB-short ORB[197.41,199.27] vol=1.7x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 10:10:00 | 195.81 | 197.16 | 0.00 | T1 1.5R @ 195.81 |
| Stop hit — per-position SL triggered | 2024-10-01 10:55:00 | 196.73 | 196.74 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:15:00 | 191.60 | 192.45 | 0.00 | ORB-short ORB[192.20,194.24] vol=2.9x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:20:00 | 191.05 | 192.37 | 0.00 | T1 1.5R @ 191.05 |
| Target hit | 2024-10-03 15:20:00 | 188.85 | 190.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 189.97 | 191.49 | 0.00 | ORB-short ORB[192.50,193.45] vol=9.6x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:15:00 | 189.16 | 191.08 | 0.00 | T1 1.5R @ 189.16 |
| Stop hit — per-position SL triggered | 2024-10-17 14:00:00 | 189.97 | 190.20 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:10:00 | 189.23 | 190.49 | 0.00 | ORB-short ORB[190.22,191.95] vol=1.7x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:35:00 | 188.44 | 190.27 | 0.00 | T1 1.5R @ 188.44 |
| Target hit | 2024-10-21 15:20:00 | 184.65 | 187.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-10-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:55:00 | 182.45 | 179.10 | 0.00 | ORB-long ORB[177.05,179.75] vol=2.2x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-10-31 10:00:00 | 181.74 | 179.42 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:00:00 | 183.75 | 182.94 | 0.00 | ORB-long ORB[182.00,183.50] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-11-06 10:10:00 | 183.21 | 183.06 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:35:00 | 169.10 | 170.83 | 0.00 | ORB-short ORB[170.60,172.40] vol=2.0x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 168.16 | 170.32 | 0.00 | T1 1.5R @ 168.16 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 169.10 | 169.83 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:40:00 | 164.34 | 165.43 | 0.00 | ORB-short ORB[165.30,167.65] vol=1.9x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-11-18 09:45:00 | 164.90 | 165.38 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 172.63 | 171.97 | 0.00 | ORB-long ORB[170.80,172.32] vol=2.4x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 09:40:00 | 173.11 | 172.30 | 0.00 | T1 1.5R @ 173.11 |
| Target hit | 2024-11-28 10:35:00 | 172.71 | 172.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2024-11-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:10:00 | 170.00 | 171.04 | 0.00 | ORB-short ORB[170.10,171.95] vol=1.7x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 10:55:00 | 169.24 | 170.54 | 0.00 | T1 1.5R @ 169.24 |
| Target hit | 2024-11-29 15:00:00 | 169.23 | 169.01 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — BUY (started 2024-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:35:00 | 176.61 | 176.00 | 0.00 | ORB-long ORB[175.00,176.20] vol=2.0x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-12-04 09:55:00 | 176.20 | 176.13 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 176.37 | 176.96 | 0.00 | ORB-short ORB[176.51,178.24] vol=2.2x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:35:00 | 175.70 | 176.54 | 0.00 | T1 1.5R @ 175.70 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 176.37 | 175.75 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:55:00 | 176.36 | 174.92 | 0.00 | ORB-long ORB[173.60,175.49] vol=1.7x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-12-10 10:10:00 | 175.91 | 175.14 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:00:00 | 174.99 | 175.74 | 0.00 | ORB-short ORB[175.50,176.68] vol=1.8x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 14:10:00 | 174.55 | 175.38 | 0.00 | T1 1.5R @ 174.55 |
| Target hit | 2024-12-11 15:20:00 | 174.35 | 175.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 172.95 | 173.45 | 0.00 | ORB-short ORB[173.08,174.32] vol=2.1x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:10:00 | 172.48 | 173.22 | 0.00 | T1 1.5R @ 172.48 |
| Target hit | 2024-12-12 15:20:00 | 170.38 | 171.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-12-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:00:00 | 167.09 | 167.94 | 0.00 | ORB-short ORB[167.62,169.80] vol=2.8x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:30:00 | 166.47 | 167.53 | 0.00 | T1 1.5R @ 166.47 |
| Stop hit — per-position SL triggered | 2024-12-17 10:45:00 | 167.09 | 167.45 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 09:30:00 | 159.03 | 159.87 | 0.00 | ORB-short ORB[159.45,160.72] vol=1.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-12-23 10:05:00 | 159.82 | 159.46 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:35:00 | 160.63 | 159.95 | 0.00 | ORB-long ORB[159.00,160.24] vol=2.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-01-01 09:40:00 | 160.17 | 160.00 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:10:00 | 158.81 | 159.42 | 0.00 | ORB-short ORB[159.15,160.97] vol=1.8x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:20:00 | 158.33 | 159.25 | 0.00 | T1 1.5R @ 158.33 |
| Stop hit — per-position SL triggered | 2025-01-02 10:35:00 | 158.81 | 159.02 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 152.09 | 153.39 | 0.00 | ORB-short ORB[153.19,154.56] vol=1.6x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:40:00 | 151.37 | 152.98 | 0.00 | T1 1.5R @ 151.37 |
| Target hit | 2025-01-08 15:20:00 | 151.73 | 151.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2025-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:30:00 | 151.04 | 150.60 | 0.00 | ORB-long ORB[150.13,150.99] vol=1.6x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-01-16 11:25:00 | 150.46 | 150.95 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:45:00 | 151.70 | 150.60 | 0.00 | ORB-long ORB[149.20,151.00] vol=1.7x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:35:00 | 152.42 | 151.25 | 0.00 | T1 1.5R @ 152.42 |
| Target hit | 2025-01-20 15:20:00 | 152.53 | 151.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2025-01-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:20:00 | 149.80 | 150.46 | 0.00 | ORB-short ORB[150.40,151.69] vol=1.6x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:40:00 | 149.19 | 150.18 | 0.00 | T1 1.5R @ 149.19 |
| Stop hit — per-position SL triggered | 2025-01-24 13:10:00 | 149.80 | 150.10 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:30:00 | 146.24 | 146.87 | 0.00 | ORB-short ORB[146.25,148.05] vol=1.8x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-01-27 09:40:00 | 146.79 | 146.79 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 151.79 | 153.15 | 0.00 | ORB-short ORB[152.51,154.74] vol=1.9x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-02-06 09:40:00 | 152.28 | 152.93 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:05:00 | 154.53 | 152.98 | 0.00 | ORB-long ORB[152.26,153.46] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-02-07 12:05:00 | 153.79 | 153.29 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:05:00 | 149.97 | 151.18 | 0.00 | ORB-short ORB[151.15,152.65] vol=1.7x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-02-10 10:35:00 | 150.49 | 150.96 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:30:00 | 147.02 | 147.63 | 0.00 | ORB-short ORB[147.05,148.89] vol=1.8x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-02-11 09:35:00 | 147.48 | 147.59 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 11:05:00 | 149.69 | 148.81 | 0.00 | ORB-long ORB[147.80,149.45] vol=3.3x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 11:15:00 | 150.40 | 149.01 | 0.00 | T1 1.5R @ 150.40 |
| Stop hit — per-position SL triggered | 2025-03-06 11:25:00 | 149.69 | 149.07 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:00:00 | 151.10 | 150.18 | 0.00 | ORB-long ORB[148.67,150.80] vol=2.4x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 10:35:00 | 151.92 | 150.54 | 0.00 | T1 1.5R @ 151.92 |
| Stop hit — per-position SL triggered | 2025-03-10 12:10:00 | 151.10 | 150.97 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 142.82 | 141.93 | 0.00 | ORB-long ORB[141.00,142.03] vol=1.6x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:20:00 | 143.32 | 142.38 | 0.00 | T1 1.5R @ 143.32 |
| Target hit | 2025-03-21 13:10:00 | 143.51 | 143.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — BUY (started 2025-03-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 09:55:00 | 149.34 | 148.29 | 0.00 | ORB-long ORB[146.96,149.10] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 10:05:00 | 150.14 | 148.67 | 0.00 | T1 1.5R @ 150.14 |
| Target hit | 2025-03-26 12:00:00 | 150.35 | 150.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — BUY (started 2025-03-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 09:45:00 | 148.97 | 147.71 | 0.00 | ORB-long ORB[146.36,148.40] vol=1.8x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-03-27 11:20:00 | 148.29 | 148.12 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 11:05:00 | 150.16 | 148.75 | 0.00 | ORB-long ORB[148.00,149.96] vol=1.8x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-04-08 11:15:00 | 149.48 | 148.88 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:35:00 | 152.20 | 151.44 | 0.00 | ORB-long ORB[150.55,152.00] vol=2.2x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 10:35:00 | 152.99 | 151.87 | 0.00 | T1 1.5R @ 152.99 |
| Target hit | 2025-04-15 15:20:00 | 154.10 | 153.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:10:00 | 168.90 | 168.35 | 0.00 | ORB-long ORB[165.75,167.99] vol=1.8x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 12:20:00 | 169.99 | 168.79 | 0.00 | T1 1.5R @ 169.99 |
| Stop hit — per-position SL triggered | 2025-04-22 14:45:00 | 168.90 | 169.30 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 11:10:00 | 171.11 | 170.23 | 0.00 | ORB-long ORB[169.05,171.00] vol=2.0x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-04-24 11:20:00 | 170.62 | 170.28 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-04-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:40:00 | 168.39 | 169.81 | 0.00 | ORB-short ORB[169.65,171.76] vol=1.7x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:55:00 | 167.35 | 169.05 | 0.00 | T1 1.5R @ 167.35 |
| Target hit | 2025-04-25 12:55:00 | 167.59 | 167.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — SELL (started 2025-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 10:55:00 | 163.45 | 164.55 | 0.00 | ORB-short ORB[164.66,166.86] vol=2.2x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 11:15:00 | 162.38 | 164.13 | 0.00 | T1 1.5R @ 162.38 |
| Stop hit — per-position SL triggered | 2025-04-28 11:40:00 | 163.45 | 164.01 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-05-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:45:00 | 165.52 | 163.78 | 0.00 | ORB-long ORB[161.31,163.19] vol=2.0x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-05-05 11:40:00 | 164.98 | 164.04 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:25:00 | 190.50 | 2024-05-14 10:35:00 | 189.79 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-05-27 10:40:00 | 189.00 | 2024-05-27 10:55:00 | 188.29 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-29 09:45:00 | 191.90 | 2024-05-29 09:50:00 | 191.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-30 11:10:00 | 188.80 | 2024-05-30 11:30:00 | 187.98 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-30 11:10:00 | 188.80 | 2024-05-30 13:00:00 | 188.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-06 10:55:00 | 194.30 | 2024-06-06 11:10:00 | 193.59 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-07 09:55:00 | 194.15 | 2024-06-07 10:15:00 | 193.45 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-10 11:10:00 | 198.63 | 2024-06-10 11:15:00 | 199.76 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-10 11:10:00 | 198.63 | 2024-06-10 14:15:00 | 199.41 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-12 09:35:00 | 200.08 | 2024-06-12 10:40:00 | 199.29 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-06-13 10:50:00 | 195.86 | 2024-06-13 11:30:00 | 195.04 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-06-13 10:50:00 | 195.86 | 2024-06-13 12:00:00 | 195.86 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-14 09:35:00 | 193.87 | 2024-06-14 09:45:00 | 194.43 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-18 09:55:00 | 196.00 | 2024-06-18 10:45:00 | 196.77 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-18 09:55:00 | 196.00 | 2024-06-18 15:20:00 | 198.21 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2024-06-19 10:05:00 | 200.45 | 2024-06-19 10:10:00 | 199.77 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-26 09:45:00 | 206.39 | 2024-06-26 09:50:00 | 207.47 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-06-26 09:45:00 | 206.39 | 2024-06-26 12:40:00 | 206.82 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2024-06-27 10:50:00 | 202.85 | 2024-06-27 11:25:00 | 201.82 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-06-27 10:50:00 | 202.85 | 2024-06-27 15:20:00 | 200.35 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2024-07-01 11:10:00 | 205.38 | 2024-07-01 12:05:00 | 206.00 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-07-01 11:10:00 | 205.38 | 2024-07-01 12:30:00 | 205.38 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-15 09:40:00 | 192.00 | 2024-07-15 10:05:00 | 192.51 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-19 10:00:00 | 192.66 | 2024-07-19 10:15:00 | 191.59 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-07-19 10:00:00 | 192.66 | 2024-07-19 10:20:00 | 192.66 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 11:15:00 | 197.50 | 2024-07-23 11:40:00 | 198.40 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-23 11:15:00 | 197.50 | 2024-07-23 12:15:00 | 197.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 10:45:00 | 201.50 | 2024-08-08 11:10:00 | 202.21 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-20 09:30:00 | 194.68 | 2024-08-20 09:35:00 | 194.27 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-27 11:15:00 | 202.10 | 2024-08-27 11:30:00 | 201.71 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-29 09:55:00 | 195.35 | 2024-08-29 10:10:00 | 194.65 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-29 09:55:00 | 195.35 | 2024-08-29 15:05:00 | 194.34 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-03 11:10:00 | 203.94 | 2024-09-03 11:15:00 | 203.38 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-06 09:55:00 | 198.93 | 2024-09-06 10:05:00 | 197.76 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-06 09:55:00 | 198.93 | 2024-09-06 11:45:00 | 198.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-16 11:15:00 | 206.64 | 2024-09-16 11:40:00 | 207.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-17 09:40:00 | 209.30 | 2024-09-17 10:15:00 | 208.46 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-19 10:40:00 | 208.59 | 2024-09-19 10:50:00 | 209.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-09-24 11:05:00 | 207.97 | 2024-09-24 11:10:00 | 208.58 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-25 10:10:00 | 205.37 | 2024-09-25 10:35:00 | 204.46 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-25 10:10:00 | 205.37 | 2024-09-25 15:10:00 | 205.07 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2024-09-26 11:05:00 | 203.15 | 2024-09-26 11:40:00 | 203.59 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-01 09:45:00 | 196.73 | 2024-10-01 10:10:00 | 195.81 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-10-01 09:45:00 | 196.73 | 2024-10-01 10:55:00 | 196.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-03 11:15:00 | 191.60 | 2024-10-03 11:20:00 | 191.05 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-10-03 11:15:00 | 191.60 | 2024-10-03 15:20:00 | 188.85 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2024-10-17 11:05:00 | 189.97 | 2024-10-17 11:15:00 | 189.16 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-17 11:05:00 | 189.97 | 2024-10-17 14:00:00 | 189.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 11:10:00 | 189.23 | 2024-10-21 11:35:00 | 188.44 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-21 11:10:00 | 189.23 | 2024-10-21 15:20:00 | 184.65 | TARGET_HIT | 0.50 | 2.42% |
| BUY | retest1 | 2024-10-31 09:55:00 | 182.45 | 2024-10-31 10:00:00 | 181.74 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-11-06 10:00:00 | 183.75 | 2024-11-06 10:10:00 | 183.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-13 09:35:00 | 169.10 | 2024-11-13 09:40:00 | 168.16 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-11-13 09:35:00 | 169.10 | 2024-11-13 09:50:00 | 169.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-18 09:40:00 | 164.34 | 2024-11-18 09:45:00 | 164.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-28 09:30:00 | 172.63 | 2024-11-28 09:40:00 | 173.11 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-11-28 09:30:00 | 172.63 | 2024-11-28 10:35:00 | 172.71 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-11-29 10:10:00 | 170.00 | 2024-11-29 10:55:00 | 169.24 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-11-29 10:10:00 | 170.00 | 2024-11-29 15:00:00 | 169.23 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-12-04 09:35:00 | 176.61 | 2024-12-04 09:55:00 | 176.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-06 09:30:00 | 176.37 | 2024-12-06 09:35:00 | 175.70 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-06 09:30:00 | 176.37 | 2024-12-06 10:20:00 | 176.37 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 09:55:00 | 176.36 | 2024-12-10 10:10:00 | 175.91 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-11 11:00:00 | 174.99 | 2024-12-11 14:10:00 | 174.55 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-12-11 11:00:00 | 174.99 | 2024-12-11 15:20:00 | 174.35 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-12 09:40:00 | 172.95 | 2024-12-12 10:10:00 | 172.48 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-12-12 09:40:00 | 172.95 | 2024-12-12 15:20:00 | 170.38 | TARGET_HIT | 0.50 | 1.49% |
| SELL | retest1 | 2024-12-17 10:00:00 | 167.09 | 2024-12-17 10:30:00 | 166.47 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-17 10:00:00 | 167.09 | 2024-12-17 10:45:00 | 167.09 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-23 09:30:00 | 159.03 | 2024-12-23 10:05:00 | 159.82 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-01-01 09:35:00 | 160.63 | 2025-01-01 09:40:00 | 160.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-02 10:10:00 | 158.81 | 2025-01-02 10:20:00 | 158.33 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-01-02 10:10:00 | 158.81 | 2025-01-02 10:35:00 | 158.81 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-08 11:15:00 | 152.09 | 2025-01-08 11:40:00 | 151.37 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-08 11:15:00 | 152.09 | 2025-01-08 15:20:00 | 151.73 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-01-16 09:30:00 | 151.04 | 2025-01-16 11:25:00 | 150.46 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-20 09:45:00 | 151.70 | 2025-01-20 11:35:00 | 152.42 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-01-20 09:45:00 | 151.70 | 2025-01-20 15:20:00 | 152.53 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-24 10:20:00 | 149.80 | 2025-01-24 12:40:00 | 149.19 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-01-24 10:20:00 | 149.80 | 2025-01-24 13:10:00 | 149.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 09:30:00 | 146.24 | 2025-01-27 09:40:00 | 146.79 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-06 09:30:00 | 151.79 | 2025-02-06 09:40:00 | 152.28 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-07 11:05:00 | 154.53 | 2025-02-07 12:05:00 | 153.79 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-02-10 10:05:00 | 149.97 | 2025-02-10 10:35:00 | 150.49 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-11 09:30:00 | 147.02 | 2025-02-11 09:35:00 | 147.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-06 11:05:00 | 149.69 | 2025-03-06 11:15:00 | 150.40 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-03-06 11:05:00 | 149.69 | 2025-03-06 11:25:00 | 149.69 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-10 10:00:00 | 151.10 | 2025-03-10 10:35:00 | 151.92 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-03-10 10:00:00 | 151.10 | 2025-03-10 12:10:00 | 151.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:40:00 | 142.82 | 2025-03-21 10:20:00 | 143.32 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-03-21 09:40:00 | 142.82 | 2025-03-21 13:10:00 | 143.51 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-03-26 09:55:00 | 149.34 | 2025-03-26 10:05:00 | 150.14 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-03-26 09:55:00 | 149.34 | 2025-03-26 12:00:00 | 150.35 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2025-03-27 09:45:00 | 148.97 | 2025-03-27 11:20:00 | 148.29 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-04-08 11:05:00 | 150.16 | 2025-04-08 11:15:00 | 149.48 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-15 09:35:00 | 152.20 | 2025-04-15 10:35:00 | 152.99 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-15 09:35:00 | 152.20 | 2025-04-15 15:20:00 | 154.10 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2025-04-22 10:10:00 | 168.90 | 2025-04-22 12:20:00 | 169.99 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-04-22 10:10:00 | 168.90 | 2025-04-22 14:45:00 | 168.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 11:10:00 | 171.11 | 2025-04-24 11:20:00 | 170.62 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-25 09:40:00 | 168.39 | 2025-04-25 09:55:00 | 167.35 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-04-25 09:40:00 | 168.39 | 2025-04-25 12:55:00 | 167.59 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2025-04-28 10:55:00 | 163.45 | 2025-04-28 11:15:00 | 162.38 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-04-28 10:55:00 | 163.45 | 2025-04-28 11:40:00 | 163.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 10:45:00 | 165.52 | 2025-05-05 11:40:00 | 164.98 | STOP_HIT | 1.00 | -0.32% |
