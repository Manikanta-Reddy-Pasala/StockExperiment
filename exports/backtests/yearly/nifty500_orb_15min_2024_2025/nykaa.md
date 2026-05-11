# FSN E-Commerce Ventures Ltd. (NYKAA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 273.00
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 14 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 108 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 60
- **Target hits / Stop hits / Partials:** 14 / 60 / 34
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 17.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 25 | 41.7% | 8 | 35 | 17 | 0.18% | 10.8% |
| BUY @ 2nd Alert (retest1) | 60 | 25 | 41.7% | 8 | 35 | 17 | 0.18% | 10.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 23 | 47.9% | 6 | 25 | 17 | 0.15% | 7.2% |
| SELL @ 2nd Alert (retest1) | 48 | 23 | 47.9% | 6 | 25 | 17 | 0.15% | 7.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 108 | 48 | 44.4% | 14 | 60 | 34 | 0.17% | 17.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:30:00 | 166.90 | 168.04 | 0.00 | ORB-short ORB[168.35,169.95] vol=2.0x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-05-14 11:05:00 | 167.54 | 167.84 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:35:00 | 170.80 | 169.76 | 0.00 | ORB-long ORB[167.70,169.80] vol=2.6x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 09:45:00 | 171.94 | 170.29 | 0.00 | T1 1.5R @ 171.94 |
| Stop hit — per-position SL triggered | 2024-05-15 10:35:00 | 170.80 | 171.36 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:15:00 | 177.40 | 176.07 | 0.00 | ORB-long ORB[175.05,176.75] vol=2.0x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-05-21 10:35:00 | 176.81 | 176.38 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:10:00 | 165.90 | 166.78 | 0.00 | ORB-short ORB[166.40,167.80] vol=1.7x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 13:05:00 | 164.96 | 166.15 | 0.00 | T1 1.5R @ 164.96 |
| Stop hit — per-position SL triggered | 2024-05-28 13:50:00 | 165.90 | 165.79 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:15:00 | 171.55 | 170.99 | 0.00 | ORB-long ORB[169.00,171.40] vol=2.7x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-06-07 12:00:00 | 171.02 | 171.01 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:20:00 | 170.50 | 169.47 | 0.00 | ORB-long ORB[168.30,170.06] vol=1.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-06-10 10:30:00 | 169.85 | 169.64 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:10:00 | 172.10 | 171.21 | 0.00 | ORB-long ORB[170.20,171.95] vol=2.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 171.61 | 171.24 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:40:00 | 174.40 | 173.13 | 0.00 | ORB-long ORB[172.20,173.89] vol=2.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-06-21 10:00:00 | 173.81 | 173.51 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:00:00 | 174.82 | 173.91 | 0.00 | ORB-long ORB[172.77,174.14] vol=1.6x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 10:20:00 | 175.85 | 174.84 | 0.00 | T1 1.5R @ 175.85 |
| Target hit | 2024-06-24 14:30:00 | 175.99 | 176.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2024-07-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 11:05:00 | 173.69 | 173.80 | 0.00 | ORB-short ORB[174.25,176.84] vol=6.1x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-07-04 11:15:00 | 174.18 | 173.82 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:55:00 | 174.29 | 173.15 | 0.00 | ORB-long ORB[170.90,172.90] vol=2.0x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-07-05 10:25:00 | 173.42 | 173.29 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:55:00 | 174.85 | 176.60 | 0.00 | ORB-short ORB[176.31,177.78] vol=2.3x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 12:15:00 | 173.71 | 175.66 | 0.00 | T1 1.5R @ 173.71 |
| Stop hit — per-position SL triggered | 2024-07-09 15:00:00 | 174.85 | 174.78 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:55:00 | 178.86 | 178.17 | 0.00 | ORB-long ORB[175.79,178.35] vol=2.0x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-07-11 10:10:00 | 178.03 | 178.20 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:40:00 | 180.38 | 179.32 | 0.00 | ORB-long ORB[177.30,179.34] vol=4.2x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-07-12 11:35:00 | 179.86 | 179.96 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:50:00 | 177.20 | 175.82 | 0.00 | ORB-long ORB[173.05,175.39] vol=1.8x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-07-22 11:00:00 | 176.54 | 175.87 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:55:00 | 179.34 | 177.29 | 0.00 | ORB-long ORB[176.51,178.00] vol=4.2x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-07-24 11:00:00 | 178.63 | 177.57 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:55:00 | 196.86 | 194.71 | 0.00 | ORB-long ORB[193.10,195.87] vol=5.3x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:00:00 | 198.41 | 195.03 | 0.00 | T1 1.5R @ 198.41 |
| Target hit | 2024-08-01 11:55:00 | 197.29 | 197.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2024-08-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:35:00 | 192.24 | 189.75 | 0.00 | ORB-long ORB[188.54,190.50] vol=2.4x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-08-07 10:50:00 | 191.17 | 189.99 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 09:50:00 | 190.34 | 192.64 | 0.00 | ORB-short ORB[191.90,194.70] vol=2.0x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-08-08 10:05:00 | 191.42 | 192.12 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 11:05:00 | 191.80 | 190.95 | 0.00 | ORB-long ORB[190.10,191.52] vol=2.5x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-08-09 12:00:00 | 191.22 | 191.21 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 189.55 | 190.78 | 0.00 | ORB-short ORB[190.57,192.80] vol=2.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-08-13 09:40:00 | 190.51 | 190.71 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:20:00 | 212.12 | 215.04 | 0.00 | ORB-short ORB[215.15,218.35] vol=1.6x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:35:00 | 210.48 | 214.08 | 0.00 | T1 1.5R @ 210.48 |
| Stop hit — per-position SL triggered | 2024-08-29 10:50:00 | 212.12 | 213.66 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 208.12 | 209.76 | 0.00 | ORB-short ORB[209.01,211.30] vol=1.7x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-08-30 09:35:00 | 209.16 | 209.66 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 206.44 | 207.18 | 0.00 | ORB-short ORB[206.60,209.18] vol=2.6x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 09:55:00 | 205.26 | 206.56 | 0.00 | T1 1.5R @ 205.26 |
| Stop hit — per-position SL triggered | 2024-09-03 10:10:00 | 206.44 | 206.51 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:30:00 | 218.85 | 217.21 | 0.00 | ORB-long ORB[215.00,217.88] vol=3.5x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:35:00 | 220.47 | 218.04 | 0.00 | T1 1.5R @ 220.47 |
| Stop hit — per-position SL triggered | 2024-09-06 09:50:00 | 218.85 | 218.82 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 09:55:00 | 199.38 | 200.19 | 0.00 | ORB-short ORB[200.03,202.82] vol=1.9x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-09-20 10:00:00 | 200.15 | 200.18 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:50:00 | 199.16 | 200.32 | 0.00 | ORB-short ORB[200.10,202.47] vol=3.0x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:00:00 | 198.07 | 199.74 | 0.00 | T1 1.5R @ 198.07 |
| Target hit | 2024-09-24 15:20:00 | 196.74 | 197.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-09-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:40:00 | 196.90 | 197.36 | 0.00 | ORB-short ORB[197.00,199.33] vol=2.0x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-09-25 10:45:00 | 197.64 | 197.37 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:25:00 | 204.17 | 202.55 | 0.00 | ORB-long ORB[200.30,203.11] vol=1.9x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:30:00 | 205.25 | 202.94 | 0.00 | T1 1.5R @ 205.25 |
| Stop hit — per-position SL triggered | 2024-09-27 10:35:00 | 204.17 | 203.06 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:30:00 | 196.23 | 197.20 | 0.00 | ORB-short ORB[197.20,199.79] vol=1.8x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 14:55:00 | 195.09 | 196.31 | 0.00 | T1 1.5R @ 195.09 |
| Target hit | 2024-09-30 15:20:00 | 195.90 | 196.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-10-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:45:00 | 196.01 | 195.40 | 0.00 | ORB-long ORB[194.42,196.00] vol=1.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-10-03 11:20:00 | 195.41 | 195.69 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-07 09:40:00 | 194.00 | 192.91 | 0.00 | ORB-long ORB[191.15,193.90] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-10-07 09:55:00 | 193.19 | 193.23 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:45:00 | 181.87 | 184.15 | 0.00 | ORB-short ORB[184.45,186.00] vol=1.5x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-10-21 10:55:00 | 182.64 | 182.37 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:30:00 | 178.74 | 179.65 | 0.00 | ORB-short ORB[178.91,181.30] vol=2.1x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-10-22 10:45:00 | 179.38 | 179.59 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 09:40:00 | 176.82 | 178.12 | 0.00 | ORB-short ORB[177.40,179.54] vol=2.2x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-10-24 10:00:00 | 177.71 | 177.86 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 10:15:00 | 179.16 | 178.11 | 0.00 | ORB-long ORB[177.26,178.88] vol=1.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-10-29 10:25:00 | 178.24 | 178.30 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:00:00 | 182.00 | 182.65 | 0.00 | ORB-short ORB[182.12,183.83] vol=1.6x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 15:10:00 | 181.12 | 182.02 | 0.00 | T1 1.5R @ 181.12 |
| Target hit | 2024-11-06 15:20:00 | 181.18 | 181.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-11-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 11:00:00 | 184.57 | 183.30 | 0.00 | ORB-long ORB[181.00,182.97] vol=3.2x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 11:05:00 | 185.56 | 183.55 | 0.00 | T1 1.5R @ 185.56 |
| Target hit | 2024-11-07 15:20:00 | 192.80 | 188.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-11-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 11:05:00 | 187.05 | 185.29 | 0.00 | ORB-long ORB[183.40,185.93] vol=2.6x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 11:15:00 | 188.09 | 185.97 | 0.00 | T1 1.5R @ 188.09 |
| Stop hit — per-position SL triggered | 2024-11-12 11:30:00 | 187.05 | 186.89 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 11:00:00 | 173.00 | 172.22 | 0.00 | ORB-long ORB[169.69,172.19] vol=1.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-11-19 11:30:00 | 172.45 | 172.32 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 09:40:00 | 170.52 | 171.02 | 0.00 | ORB-short ORB[170.53,172.40] vol=2.0x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 11:55:00 | 169.85 | 170.80 | 0.00 | T1 1.5R @ 169.85 |
| Target hit | 2024-11-25 14:15:00 | 170.00 | 169.63 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2024-11-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 10:40:00 | 166.91 | 167.52 | 0.00 | ORB-short ORB[167.55,169.67] vol=2.1x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 13:10:00 | 165.89 | 166.87 | 0.00 | T1 1.5R @ 165.89 |
| Stop hit — per-position SL triggered | 2024-11-26 13:55:00 | 166.91 | 166.80 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-11-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:10:00 | 172.20 | 173.13 | 0.00 | ORB-short ORB[172.90,174.89] vol=1.9x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 10:20:00 | 170.98 | 172.55 | 0.00 | T1 1.5R @ 170.98 |
| Stop hit — per-position SL triggered | 2024-11-29 10:30:00 | 172.20 | 172.45 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:05:00 | 173.15 | 171.42 | 0.00 | ORB-long ORB[169.39,171.50] vol=2.4x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-12-03 11:10:00 | 172.48 | 172.06 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:35:00 | 165.43 | 166.39 | 0.00 | ORB-short ORB[166.34,168.70] vol=2.4x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-12-09 11:50:00 | 166.03 | 165.91 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:00:00 | 170.84 | 172.75 | 0.00 | ORB-short ORB[172.56,174.00] vol=1.6x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:15:00 | 170.18 | 172.38 | 0.00 | T1 1.5R @ 170.18 |
| Target hit | 2024-12-11 15:20:00 | 169.90 | 170.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2024-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:35:00 | 167.08 | 168.03 | 0.00 | ORB-short ORB[167.37,169.40] vol=1.5x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-12-13 09:40:00 | 167.63 | 167.95 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:40:00 | 176.07 | 174.75 | 0.00 | ORB-long ORB[173.00,175.48] vol=1.6x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-12-17 09:45:00 | 174.94 | 174.82 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:55:00 | 170.62 | 169.43 | 0.00 | ORB-long ORB[167.71,169.92] vol=2.5x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-12-19 10:00:00 | 169.84 | 169.48 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:45:00 | 159.99 | 160.42 | 0.00 | ORB-short ORB[160.40,162.78] vol=1.9x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:05:00 | 159.06 | 160.22 | 0.00 | T1 1.5R @ 159.06 |
| Stop hit — per-position SL triggered | 2024-12-26 10:30:00 | 159.99 | 159.72 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:55:00 | 160.88 | 159.81 | 0.00 | ORB-long ORB[158.65,160.48] vol=2.9x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 11:15:00 | 161.62 | 160.55 | 0.00 | T1 1.5R @ 161.62 |
| Target hit | 2024-12-30 13:25:00 | 161.15 | 161.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — BUY (started 2024-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:50:00 | 162.08 | 161.03 | 0.00 | ORB-long ORB[160.31,161.80] vol=2.4x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-12-31 11:45:00 | 161.53 | 161.40 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:35:00 | 165.33 | 164.01 | 0.00 | ORB-long ORB[162.40,163.90] vol=1.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-01-01 09:45:00 | 164.84 | 164.49 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:35:00 | 168.00 | 167.45 | 0.00 | ORB-long ORB[165.32,167.15] vol=2.8x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:00:00 | 168.91 | 168.06 | 0.00 | T1 1.5R @ 168.91 |
| Stop hit — per-position SL triggered | 2025-01-03 10:25:00 | 168.00 | 168.26 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:00:00 | 173.12 | 171.47 | 0.00 | ORB-long ORB[168.30,170.30] vol=2.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-01-09 10:45:00 | 172.41 | 172.19 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:55:00 | 168.24 | 166.86 | 0.00 | ORB-long ORB[165.64,167.58] vol=1.9x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 11:05:00 | 169.01 | 167.18 | 0.00 | T1 1.5R @ 169.01 |
| Target hit | 2025-01-15 15:20:00 | 172.40 | 170.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-01-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:50:00 | 168.35 | 169.73 | 0.00 | ORB-short ORB[170.49,172.09] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 11:10:00 | 167.58 | 169.48 | 0.00 | T1 1.5R @ 167.58 |
| Stop hit — per-position SL triggered | 2025-01-21 11:15:00 | 168.35 | 169.44 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:00:00 | 168.65 | 166.93 | 0.00 | ORB-long ORB[164.39,166.87] vol=2.7x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:25:00 | 169.77 | 167.72 | 0.00 | T1 1.5R @ 169.77 |
| Stop hit — per-position SL triggered | 2025-01-23 11:40:00 | 168.65 | 168.66 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:15:00 | 162.89 | 163.71 | 0.00 | ORB-short ORB[164.05,165.99] vol=1.5x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-01-27 10:45:00 | 163.63 | 163.55 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-02-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:45:00 | 167.08 | 168.32 | 0.00 | ORB-short ORB[168.30,169.75] vol=4.6x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:00:00 | 166.37 | 167.72 | 0.00 | T1 1.5R @ 166.37 |
| Stop hit — per-position SL triggered | 2025-02-01 12:10:00 | 167.08 | 166.27 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 11:05:00 | 178.14 | 178.85 | 0.00 | ORB-short ORB[179.00,181.39] vol=1.9x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-02-04 11:40:00 | 178.85 | 178.76 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 10:45:00 | 176.85 | 177.99 | 0.00 | ORB-short ORB[177.38,179.49] vol=1.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 10:50:00 | 176.12 | 177.82 | 0.00 | T1 1.5R @ 176.12 |
| Stop hit — per-position SL triggered | 2025-02-05 11:45:00 | 176.85 | 176.71 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 165.40 | 164.51 | 0.00 | ORB-long ORB[163.51,164.90] vol=2.7x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:55:00 | 166.33 | 164.94 | 0.00 | T1 1.5R @ 166.33 |
| Stop hit — per-position SL triggered | 2025-02-25 10:10:00 | 165.40 | 165.06 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:40:00 | 163.44 | 162.70 | 0.00 | ORB-long ORB[161.06,163.00] vol=1.7x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-03-05 09:50:00 | 162.87 | 162.86 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 163.22 | 161.83 | 0.00 | ORB-long ORB[159.76,162.09] vol=2.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 11:15:00 | 164.16 | 162.07 | 0.00 | T1 1.5R @ 164.16 |
| Stop hit — per-position SL triggered | 2025-03-11 11:30:00 | 163.22 | 162.23 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-03-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 10:35:00 | 165.15 | 165.79 | 0.00 | ORB-short ORB[165.53,167.49] vol=1.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 12:10:00 | 164.56 | 165.16 | 0.00 | T1 1.5R @ 164.56 |
| Stop hit — per-position SL triggered | 2025-03-19 12:35:00 | 165.15 | 164.97 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:05:00 | 167.96 | 166.73 | 0.00 | ORB-long ORB[165.29,166.75] vol=1.7x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 10:15:00 | 168.72 | 167.69 | 0.00 | T1 1.5R @ 168.72 |
| Target hit | 2025-03-20 10:45:00 | 168.70 | 168.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — BUY (started 2025-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:15:00 | 169.32 | 168.88 | 0.00 | ORB-long ORB[167.01,169.05] vol=4.8x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:25:00 | 170.04 | 168.98 | 0.00 | T1 1.5R @ 170.04 |
| Target hit | 2025-03-21 13:50:00 | 171.80 | 172.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 69 — BUY (started 2025-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:35:00 | 176.41 | 175.13 | 0.00 | ORB-long ORB[173.22,174.90] vol=3.5x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-03-25 10:05:00 | 175.75 | 175.91 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:40:00 | 191.41 | 189.76 | 0.00 | ORB-long ORB[187.68,189.70] vol=2.3x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-04-17 10:45:00 | 190.73 | 189.88 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 191.36 | 190.15 | 0.00 | ORB-long ORB[188.86,190.88] vol=1.7x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:35:00 | 192.37 | 191.25 | 0.00 | T1 1.5R @ 192.37 |
| Target hit | 2025-04-21 10:50:00 | 193.91 | 194.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — SELL (started 2025-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-22 11:05:00 | 190.08 | 192.23 | 0.00 | ORB-short ORB[191.12,193.60] vol=2.0x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 11:10:00 | 189.11 | 192.07 | 0.00 | T1 1.5R @ 189.11 |
| Target hit | 2025-04-22 15:20:00 | 187.63 | 189.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-04-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:05:00 | 194.30 | 192.96 | 0.00 | ORB-long ORB[191.60,193.75] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-04-24 10:10:00 | 193.62 | 193.02 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 10:00:00 | 193.99 | 192.63 | 0.00 | ORB-long ORB[190.75,193.00] vol=2.7x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 10:15:00 | 194.95 | 193.40 | 0.00 | T1 1.5R @ 194.95 |
| Stop hit — per-position SL triggered | 2025-04-28 10:20:00 | 193.99 | 193.41 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:30:00 | 166.90 | 2024-05-14 11:05:00 | 167.54 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-15 09:35:00 | 170.80 | 2024-05-15 09:45:00 | 171.94 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-05-15 09:35:00 | 170.80 | 2024-05-15 10:35:00 | 170.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-21 10:15:00 | 177.40 | 2024-05-21 10:35:00 | 176.81 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-28 10:10:00 | 165.90 | 2024-05-28 13:05:00 | 164.96 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-05-28 10:10:00 | 165.90 | 2024-05-28 13:50:00 | 165.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 11:15:00 | 171.55 | 2024-06-07 12:00:00 | 171.02 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-10 10:20:00 | 170.50 | 2024-06-10 10:30:00 | 169.85 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-12 11:10:00 | 172.10 | 2024-06-12 11:15:00 | 171.61 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-21 09:40:00 | 174.40 | 2024-06-21 10:00:00 | 173.81 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-24 10:00:00 | 174.82 | 2024-06-24 10:20:00 | 175.85 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-06-24 10:00:00 | 174.82 | 2024-06-24 14:30:00 | 175.99 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2024-07-04 11:05:00 | 173.69 | 2024-07-04 11:15:00 | 174.18 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-05 09:55:00 | 174.29 | 2024-07-05 10:25:00 | 173.42 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-07-09 09:55:00 | 174.85 | 2024-07-09 12:15:00 | 173.71 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-07-09 09:55:00 | 174.85 | 2024-07-09 15:00:00 | 174.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 09:55:00 | 178.86 | 2024-07-11 10:10:00 | 178.03 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-07-12 10:40:00 | 180.38 | 2024-07-12 11:35:00 | 179.86 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-22 10:50:00 | 177.20 | 2024-07-22 11:00:00 | 176.54 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-24 10:55:00 | 179.34 | 2024-07-24 11:00:00 | 178.63 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-01 10:55:00 | 196.86 | 2024-08-01 11:00:00 | 198.41 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-08-01 10:55:00 | 196.86 | 2024-08-01 11:55:00 | 197.29 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-08-07 10:35:00 | 192.24 | 2024-08-07 10:50:00 | 191.17 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-08-08 09:50:00 | 190.34 | 2024-08-08 10:05:00 | 191.42 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-08-09 11:05:00 | 191.80 | 2024-08-09 12:00:00 | 191.22 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-13 09:35:00 | 189.55 | 2024-08-13 09:40:00 | 190.51 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-08-29 10:20:00 | 212.12 | 2024-08-29 10:35:00 | 210.48 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-08-29 10:20:00 | 212.12 | 2024-08-29 10:50:00 | 212.12 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 09:30:00 | 208.12 | 2024-08-30 09:35:00 | 209.16 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-09-03 09:30:00 | 206.44 | 2024-09-03 09:55:00 | 205.26 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-03 09:30:00 | 206.44 | 2024-09-03 10:10:00 | 206.44 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-06 09:30:00 | 218.85 | 2024-09-06 09:35:00 | 220.47 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-09-06 09:30:00 | 218.85 | 2024-09-06 09:50:00 | 218.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-20 09:55:00 | 199.38 | 2024-09-20 10:00:00 | 200.15 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-24 09:50:00 | 199.16 | 2024-09-24 10:00:00 | 198.07 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-09-24 09:50:00 | 199.16 | 2024-09-24 15:20:00 | 196.74 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2024-09-25 10:40:00 | 196.90 | 2024-09-25 10:45:00 | 197.64 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-27 10:25:00 | 204.17 | 2024-09-27 10:30:00 | 205.25 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-09-27 10:25:00 | 204.17 | 2024-09-27 10:35:00 | 204.17 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-30 10:30:00 | 196.23 | 2024-09-30 14:55:00 | 195.09 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-30 10:30:00 | 196.23 | 2024-09-30 15:20:00 | 195.90 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2024-10-03 09:45:00 | 196.01 | 2024-10-03 11:20:00 | 195.41 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-07 09:40:00 | 194.00 | 2024-10-07 09:55:00 | 193.19 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-21 09:45:00 | 181.87 | 2024-10-21 10:55:00 | 182.64 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-22 10:30:00 | 178.74 | 2024-10-22 10:45:00 | 179.38 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-24 09:40:00 | 176.82 | 2024-10-24 10:00:00 | 177.71 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-29 10:15:00 | 179.16 | 2024-10-29 10:25:00 | 178.24 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-11-06 10:00:00 | 182.00 | 2024-11-06 15:10:00 | 181.12 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-11-06 10:00:00 | 182.00 | 2024-11-06 15:20:00 | 181.18 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-11-07 11:00:00 | 184.57 | 2024-11-07 11:05:00 | 185.56 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-11-07 11:00:00 | 184.57 | 2024-11-07 15:20:00 | 192.80 | TARGET_HIT | 0.50 | 4.46% |
| BUY | retest1 | 2024-11-12 11:05:00 | 187.05 | 2024-11-12 11:15:00 | 188.09 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-11-12 11:05:00 | 187.05 | 2024-11-12 11:30:00 | 187.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 11:00:00 | 173.00 | 2024-11-19 11:30:00 | 172.45 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-25 09:40:00 | 170.52 | 2024-11-25 11:55:00 | 169.85 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-11-25 09:40:00 | 170.52 | 2024-11-25 14:15:00 | 170.00 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2024-11-26 10:40:00 | 166.91 | 2024-11-26 13:10:00 | 165.89 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-11-26 10:40:00 | 166.91 | 2024-11-26 13:55:00 | 166.91 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 10:10:00 | 172.20 | 2024-11-29 10:20:00 | 170.98 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-11-29 10:10:00 | 172.20 | 2024-11-29 10:30:00 | 172.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 10:05:00 | 173.15 | 2024-12-03 11:10:00 | 172.48 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-09 10:35:00 | 165.43 | 2024-12-09 11:50:00 | 166.03 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-11 11:00:00 | 170.84 | 2024-12-11 11:15:00 | 170.18 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-11 11:00:00 | 170.84 | 2024-12-11 15:20:00 | 169.90 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2024-12-13 09:35:00 | 167.08 | 2024-12-13 09:40:00 | 167.63 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-17 09:40:00 | 176.07 | 2024-12-17 09:45:00 | 174.94 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2024-12-19 09:55:00 | 170.62 | 2024-12-19 10:00:00 | 169.84 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-12-26 09:45:00 | 159.99 | 2024-12-26 10:05:00 | 159.06 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-12-26 09:45:00 | 159.99 | 2024-12-26 10:30:00 | 159.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:55:00 | 160.88 | 2024-12-30 11:15:00 | 161.62 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-30 10:55:00 | 160.88 | 2024-12-30 13:25:00 | 161.15 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2024-12-31 10:50:00 | 162.08 | 2024-12-31 11:45:00 | 161.53 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-01 09:35:00 | 165.33 | 2025-01-01 09:45:00 | 164.84 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-03 09:35:00 | 168.00 | 2025-01-03 10:00:00 | 168.91 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-03 09:35:00 | 168.00 | 2025-01-03 10:25:00 | 168.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 10:00:00 | 173.12 | 2025-01-09 10:45:00 | 172.41 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-01-15 10:55:00 | 168.24 | 2025-01-15 11:05:00 | 169.01 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-15 10:55:00 | 168.24 | 2025-01-15 15:20:00 | 172.40 | TARGET_HIT | 0.50 | 2.47% |
| SELL | retest1 | 2025-01-21 10:50:00 | 168.35 | 2025-01-21 11:10:00 | 167.58 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-21 10:50:00 | 168.35 | 2025-01-21 11:15:00 | 168.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 10:00:00 | 168.65 | 2025-01-23 10:25:00 | 169.77 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-23 10:00:00 | 168.65 | 2025-01-23 11:40:00 | 168.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 10:15:00 | 162.89 | 2025-01-27 10:45:00 | 163.63 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-02-01 10:45:00 | 167.08 | 2025-02-01 11:00:00 | 166.37 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-02-01 10:45:00 | 167.08 | 2025-02-01 12:10:00 | 167.08 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-04 11:05:00 | 178.14 | 2025-02-04 11:40:00 | 178.85 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-02-05 10:45:00 | 176.85 | 2025-02-05 10:50:00 | 176.12 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-02-05 10:45:00 | 176.85 | 2025-02-05 11:45:00 | 176.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-25 09:30:00 | 165.40 | 2025-02-25 09:55:00 | 166.33 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-02-25 09:30:00 | 165.40 | 2025-02-25 10:10:00 | 165.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 09:40:00 | 163.44 | 2025-03-05 09:50:00 | 162.87 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-11 11:00:00 | 163.22 | 2025-03-11 11:15:00 | 164.16 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-03-11 11:00:00 | 163.22 | 2025-03-11 11:30:00 | 163.22 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-19 10:35:00 | 165.15 | 2025-03-19 12:10:00 | 164.56 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-03-19 10:35:00 | 165.15 | 2025-03-19 12:35:00 | 165.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-20 10:05:00 | 167.96 | 2025-03-20 10:15:00 | 168.72 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-03-20 10:05:00 | 167.96 | 2025-03-20 10:45:00 | 168.70 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2025-03-21 10:15:00 | 169.32 | 2025-03-21 10:25:00 | 170.04 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-03-21 10:15:00 | 169.32 | 2025-03-21 13:50:00 | 171.80 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2025-03-25 09:35:00 | 176.41 | 2025-03-25 10:05:00 | 175.75 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-17 10:40:00 | 191.41 | 2025-04-17 10:45:00 | 190.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-21 09:30:00 | 191.36 | 2025-04-21 09:35:00 | 192.37 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-21 09:30:00 | 191.36 | 2025-04-21 10:50:00 | 193.91 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2025-04-22 11:05:00 | 190.08 | 2025-04-22 11:10:00 | 189.11 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-04-22 11:05:00 | 190.08 | 2025-04-22 15:20:00 | 187.63 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-04-24 10:05:00 | 194.30 | 2025-04-24 10:10:00 | 193.62 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-28 10:00:00 | 193.99 | 2025-04-28 10:15:00 | 194.95 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-28 10:00:00 | 193.99 | 2025-04-28 10:20:00 | 193.99 | STOP_HIT | 0.50 | 0.00% |
