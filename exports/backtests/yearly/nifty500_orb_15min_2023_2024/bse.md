# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-06-04 15:25:00 (19706 bars)
- **Last close:** 853.47
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
| ENTRY1 | 56 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 8 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 48
- **Target hits / Stop hits / Partials:** 8 / 48 / 22
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 11.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 17 | 37.0% | 5 | 29 | 12 | 0.17% | 7.9% |
| BUY @ 2nd Alert (retest1) | 46 | 17 | 37.0% | 5 | 29 | 12 | 0.17% | 7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 13 | 40.6% | 3 | 19 | 10 | 0.12% | 3.8% |
| SELL @ 2nd Alert (retest1) | 32 | 13 | 40.6% | 3 | 19 | 10 | 0.12% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 78 | 30 | 38.5% | 8 | 48 | 22 | 0.15% | 11.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:55:00 | 184.08 | 182.71 | 0.00 | ORB-long ORB[181.77,182.83] vol=1.9x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-05-16 10:00:00 | 183.51 | 182.79 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 10:05:00 | 181.00 | 179.77 | 0.00 | ORB-long ORB[178.85,180.33] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-05-17 10:15:00 | 180.39 | 179.87 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 174.93 | 175.51 | 0.00 | ORB-short ORB[175.33,176.65] vol=2.2x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 09:35:00 | 173.93 | 175.15 | 0.00 | T1 1.5R @ 173.93 |
| Stop hit — per-position SL triggered | 2023-05-19 10:30:00 | 174.93 | 174.78 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:20:00 | 175.48 | 175.21 | 0.00 | ORB-long ORB[173.67,175.28] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-05-22 14:45:00 | 174.85 | 175.26 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 10:10:00 | 178.30 | 177.35 | 0.00 | ORB-long ORB[176.18,177.80] vol=1.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-05-24 10:15:00 | 177.82 | 177.38 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 09:35:00 | 181.63 | 181.00 | 0.00 | ORB-long ORB[179.70,181.33] vol=4.6x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 09:45:00 | 182.67 | 181.53 | 0.00 | T1 1.5R @ 182.67 |
| Target hit | 2023-05-31 12:50:00 | 185.07 | 185.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2023-06-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 09:40:00 | 193.07 | 192.29 | 0.00 | ORB-long ORB[191.02,192.73] vol=4.1x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 09:45:00 | 194.08 | 192.53 | 0.00 | T1 1.5R @ 194.08 |
| Stop hit — per-position SL triggered | 2023-06-05 09:50:00 | 193.07 | 192.55 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:10:00 | 189.97 | 189.48 | 0.00 | ORB-long ORB[188.78,189.93] vol=1.8x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 10:35:00 | 190.70 | 189.86 | 0.00 | T1 1.5R @ 190.70 |
| Stop hit — per-position SL triggered | 2023-06-07 11:30:00 | 189.97 | 189.97 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:35:00 | 184.63 | 185.80 | 0.00 | ORB-short ORB[185.00,187.65] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-06-09 09:55:00 | 185.28 | 185.60 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 09:35:00 | 184.02 | 184.60 | 0.00 | ORB-short ORB[184.08,185.67] vol=1.6x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-06-12 09:40:00 | 184.44 | 184.57 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:40:00 | 192.10 | 191.12 | 0.00 | ORB-long ORB[189.93,191.53] vol=3.3x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 09:45:00 | 192.94 | 191.52 | 0.00 | T1 1.5R @ 192.94 |
| Stop hit — per-position SL triggered | 2023-06-15 10:40:00 | 192.10 | 192.53 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:40:00 | 193.35 | 192.52 | 0.00 | ORB-long ORB[191.67,193.17] vol=3.3x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-06-16 10:25:00 | 192.64 | 192.64 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 10:15:00 | 200.85 | 199.37 | 0.00 | ORB-long ORB[197.87,200.43] vol=2.1x ATR=0.70 |
| Stop hit — per-position SL triggered | 2023-06-27 10:25:00 | 200.15 | 199.59 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:15:00 | 227.03 | 224.57 | 0.00 | ORB-long ORB[222.07,224.33] vol=3.6x ATR=1.39 |
| Stop hit — per-position SL triggered | 2023-07-05 10:40:00 | 225.64 | 225.03 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:55:00 | 231.25 | 229.40 | 0.00 | ORB-long ORB[227.33,230.67] vol=4.4x ATR=1.10 |
| Stop hit — per-position SL triggered | 2023-07-06 10:15:00 | 230.15 | 229.77 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:40:00 | 234.00 | 233.41 | 0.00 | ORB-long ORB[232.47,233.97] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2023-07-11 10:45:00 | 233.21 | 233.54 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:30:00 | 248.55 | 250.20 | 0.00 | ORB-short ORB[249.33,252.57] vol=2.2x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 09:40:00 | 247.17 | 249.61 | 0.00 | T1 1.5R @ 247.17 |
| Stop hit — per-position SL triggered | 2023-07-18 09:50:00 | 248.55 | 249.39 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 10:40:00 | 245.90 | 247.66 | 0.00 | ORB-short ORB[246.17,248.62] vol=1.6x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 11:15:00 | 244.79 | 247.36 | 0.00 | T1 1.5R @ 244.79 |
| Stop hit — per-position SL triggered | 2023-07-19 15:10:00 | 245.90 | 246.38 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 11:15:00 | 244.67 | 246.48 | 0.00 | ORB-short ORB[244.93,248.32] vol=2.1x ATR=0.58 |
| Stop hit — per-position SL triggered | 2023-07-20 11:30:00 | 245.25 | 246.41 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 11:10:00 | 244.82 | 245.90 | 0.00 | ORB-short ORB[244.95,247.17] vol=2.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-07-24 11:35:00 | 245.30 | 245.83 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 10:05:00 | 246.30 | 244.13 | 0.00 | ORB-long ORB[242.67,244.83] vol=3.2x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-07-25 10:15:00 | 245.36 | 244.41 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 10:05:00 | 270.38 | 268.49 | 0.00 | ORB-long ORB[266.37,269.30] vol=2.7x ATR=1.21 |
| Stop hit — per-position SL triggered | 2023-07-31 10:15:00 | 269.17 | 268.71 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 09:30:00 | 270.88 | 269.79 | 0.00 | ORB-long ORB[268.00,270.77] vol=2.0x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 09:40:00 | 272.28 | 271.27 | 0.00 | T1 1.5R @ 272.28 |
| Target hit | 2023-08-01 12:45:00 | 274.27 | 274.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2023-08-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 10:00:00 | 293.72 | 291.91 | 0.00 | ORB-long ORB[289.42,292.93] vol=2.8x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 10:25:00 | 295.45 | 292.55 | 0.00 | T1 1.5R @ 295.45 |
| Stop hit — per-position SL triggered | 2023-08-08 10:30:00 | 293.72 | 292.57 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 10:20:00 | 291.63 | 289.67 | 0.00 | ORB-long ORB[288.00,291.33] vol=1.5x ATR=1.05 |
| Stop hit — per-position SL triggered | 2023-08-21 10:45:00 | 290.58 | 290.00 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 09:45:00 | 301.55 | 299.50 | 0.00 | ORB-long ORB[297.52,299.80] vol=3.1x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-23 09:50:00 | 303.18 | 300.45 | 0.00 | T1 1.5R @ 303.18 |
| Stop hit — per-position SL triggered | 2023-08-23 10:30:00 | 301.55 | 301.71 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:45:00 | 311.52 | 308.66 | 0.00 | ORB-long ORB[305.77,308.33] vol=5.4x ATR=1.20 |
| Stop hit — per-position SL triggered | 2023-08-24 09:55:00 | 310.32 | 309.22 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:40:00 | 305.23 | 304.38 | 0.00 | ORB-long ORB[302.82,305.18] vol=2.7x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 09:45:00 | 306.54 | 305.09 | 0.00 | T1 1.5R @ 306.54 |
| Target hit | 2023-08-30 15:20:00 | 322.07 | 320.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2023-10-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 09:30:00 | 450.75 | 447.42 | 0.00 | ORB-long ORB[444.00,448.43] vol=3.7x ATR=2.08 |
| Stop hit — per-position SL triggered | 2023-10-06 09:35:00 | 448.67 | 448.87 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-10-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 10:30:00 | 493.08 | 496.88 | 0.00 | ORB-short ORB[495.42,502.07] vol=2.1x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 10:50:00 | 490.86 | 496.00 | 0.00 | T1 1.5R @ 490.86 |
| Stop hit — per-position SL triggered | 2023-10-17 11:35:00 | 493.08 | 494.35 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-10-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 10:50:00 | 497.33 | 494.57 | 0.00 | ORB-long ORB[492.33,495.70] vol=6.3x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 11:05:00 | 499.80 | 495.28 | 0.00 | T1 1.5R @ 499.80 |
| Stop hit — per-position SL triggered | 2023-10-19 11:10:00 | 497.33 | 495.47 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-10-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 10:05:00 | 523.70 | 518.54 | 0.00 | ORB-long ORB[513.92,521.67] vol=1.8x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 13:55:00 | 528.04 | 521.90 | 0.00 | T1 1.5R @ 528.04 |
| Target hit | 2023-10-20 15:20:00 | 528.42 | 523.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2023-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 11:00:00 | 592.10 | 598.65 | 0.00 | ORB-short ORB[598.67,606.33] vol=2.1x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-11-06 11:05:00 | 594.12 | 598.51 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-12-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 10:25:00 | 784.80 | 803.69 | 0.00 | ORB-short ORB[808.10,819.60] vol=4.5x ATR=5.01 |
| Stop hit — per-position SL triggered | 2023-12-11 10:30:00 | 789.81 | 802.31 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:20:00 | 773.33 | 783.02 | 0.00 | ORB-short ORB[784.67,794.33] vol=1.6x ATR=3.46 |
| Stop hit — per-position SL triggered | 2023-12-12 10:25:00 | 776.79 | 782.56 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-12-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:50:00 | 790.33 | 785.72 | 0.00 | ORB-long ORB[780.72,789.98] vol=1.5x ATR=3.22 |
| Stop hit — per-position SL triggered | 2023-12-22 09:55:00 | 787.11 | 786.07 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:15:00 | 780.92 | 773.94 | 0.00 | ORB-long ORB[768.33,777.67] vol=2.3x ATR=2.76 |
| Stop hit — per-position SL triggered | 2023-12-26 10:55:00 | 778.16 | 775.72 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-12-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:35:00 | 763.95 | 761.66 | 0.00 | ORB-long ORB[756.02,762.47] vol=2.2x ATR=2.76 |
| Stop hit — per-position SL triggered | 2023-12-28 10:05:00 | 761.19 | 761.98 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 10:15:00 | 749.95 | 743.09 | 0.00 | ORB-long ORB[738.75,747.67] vol=1.7x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 10:25:00 | 754.62 | 745.37 | 0.00 | T1 1.5R @ 754.62 |
| Target hit | 2024-01-03 15:20:00 | 750.33 | 749.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2024-01-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 11:00:00 | 757.92 | 752.89 | 0.00 | ORB-long ORB[747.25,755.18] vol=5.9x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-01-04 11:10:00 | 755.30 | 753.16 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-01-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 10:35:00 | 753.33 | 764.62 | 0.00 | ORB-short ORB[762.67,773.30] vol=2.2x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-01-10 10:40:00 | 756.06 | 764.33 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-01-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:45:00 | 758.17 | 755.14 | 0.00 | ORB-long ORB[751.37,756.50] vol=2.3x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-01-11 10:55:00 | 755.43 | 755.89 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-01-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 10:20:00 | 748.50 | 749.38 | 0.00 | ORB-short ORB[749.67,755.65] vol=5.7x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 10:45:00 | 745.76 | 749.25 | 0.00 | T1 1.5R @ 745.76 |
| Stop hit — per-position SL triggered | 2024-01-12 14:40:00 | 748.50 | 747.76 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-01-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 10:00:00 | 751.00 | 748.17 | 0.00 | ORB-long ORB[743.70,748.33] vol=3.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-01-15 10:55:00 | 749.12 | 748.52 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-01-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 10:55:00 | 759.33 | 769.24 | 0.00 | ORB-short ORB[768.72,776.67] vol=2.0x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 11:15:00 | 756.08 | 767.26 | 0.00 | T1 1.5R @ 756.08 |
| Target hit | 2024-01-19 15:20:00 | 758.63 | 759.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:15:00 | 719.40 | 716.16 | 0.00 | ORB-long ORB[710.77,718.32] vol=1.6x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-01-29 10:25:00 | 717.11 | 716.48 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-02-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 09:55:00 | 786.00 | 787.94 | 0.00 | ORB-short ORB[786.35,794.00] vol=1.8x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 10:40:00 | 780.40 | 784.37 | 0.00 | T1 1.5R @ 780.40 |
| Target hit | 2024-02-15 15:20:00 | 777.00 | 779.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2024-02-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 10:20:00 | 776.52 | 773.69 | 0.00 | ORB-long ORB[769.00,772.83] vol=1.7x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-02-20 10:35:00 | 773.69 | 774.29 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-02-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:30:00 | 757.85 | 763.05 | 0.00 | ORB-short ORB[760.48,770.30] vol=2.6x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 10:40:00 | 754.96 | 761.77 | 0.00 | T1 1.5R @ 754.96 |
| Stop hit — per-position SL triggered | 2024-02-21 10:50:00 | 757.85 | 761.26 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 11:15:00 | 775.00 | 766.64 | 0.00 | ORB-long ORB[761.33,771.67] vol=6.3x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 11:20:00 | 779.25 | 769.26 | 0.00 | T1 1.5R @ 779.25 |
| Stop hit — per-position SL triggered | 2024-02-27 11:30:00 | 775.00 | 769.90 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-03-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 11:00:00 | 783.25 | 786.64 | 0.00 | ORB-short ORB[784.25,793.33] vol=5.3x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-03-01 12:40:00 | 786.10 | 785.82 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:00:00 | 754.90 | 756.63 | 0.00 | ORB-short ORB[755.40,763.13] vol=2.0x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-03-06 13:40:00 | 758.29 | 754.15 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 09:30:00 | 675.68 | 684.26 | 0.00 | ORB-short ORB[683.67,692.33] vol=1.9x ATR=4.78 |
| Stop hit — per-position SL triggered | 2024-03-18 09:40:00 | 680.46 | 682.17 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 11:05:00 | 958.27 | 959.92 | 0.00 | ORB-short ORB[958.83,967.57] vol=1.6x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 13:15:00 | 953.14 | 959.31 | 0.00 | T1 1.5R @ 953.14 |
| Target hit | 2024-04-08 15:20:00 | 933.57 | 945.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2024-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 11:10:00 | 930.77 | 942.29 | 0.00 | ORB-short ORB[938.67,951.67] vol=2.5x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 925.93 | 940.60 | 0.00 | T1 1.5R @ 925.93 |
| Stop hit — per-position SL triggered | 2024-05-07 11:20:00 | 930.77 | 940.09 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-05-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-10 09:35:00 | 885.33 | 903.28 | 0.00 | ORB-short ORB[904.17,916.38] vol=2.0x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-05-10 09:40:00 | 890.90 | 900.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-16 09:55:00 | 184.08 | 2023-05-16 10:00:00 | 183.51 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-05-17 10:05:00 | 181.00 | 2023-05-17 10:15:00 | 180.39 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-05-19 09:30:00 | 174.93 | 2023-05-19 09:35:00 | 173.93 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2023-05-19 09:30:00 | 174.93 | 2023-05-19 10:30:00 | 174.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-22 10:20:00 | 175.48 | 2023-05-22 14:45:00 | 174.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-05-24 10:10:00 | 178.30 | 2023-05-24 10:15:00 | 177.82 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-05-31 09:35:00 | 181.63 | 2023-05-31 09:45:00 | 182.67 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2023-05-31 09:35:00 | 181.63 | 2023-05-31 12:50:00 | 185.07 | TARGET_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2023-06-05 09:40:00 | 193.07 | 2023-06-05 09:45:00 | 194.08 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-06-05 09:40:00 | 193.07 | 2023-06-05 09:50:00 | 193.07 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-07 10:10:00 | 189.97 | 2023-06-07 10:35:00 | 190.70 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-07 10:10:00 | 189.97 | 2023-06-07 11:30:00 | 189.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-09 09:35:00 | 184.63 | 2023-06-09 09:55:00 | 185.28 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-06-12 09:35:00 | 184.02 | 2023-06-12 09:40:00 | 184.44 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-15 09:40:00 | 192.10 | 2023-06-15 09:45:00 | 192.94 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-06-15 09:40:00 | 192.10 | 2023-06-15 10:40:00 | 192.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-16 09:40:00 | 193.35 | 2023-06-16 10:25:00 | 192.64 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-06-27 10:15:00 | 200.85 | 2023-06-27 10:25:00 | 200.15 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-07-05 10:15:00 | 227.03 | 2023-07-05 10:40:00 | 225.64 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2023-07-06 09:55:00 | 231.25 | 2023-07-06 10:15:00 | 230.15 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2023-07-11 09:40:00 | 234.00 | 2023-07-11 10:45:00 | 233.21 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-18 09:30:00 | 248.55 | 2023-07-18 09:40:00 | 247.17 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-07-18 09:30:00 | 248.55 | 2023-07-18 09:50:00 | 248.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-19 10:40:00 | 245.90 | 2023-07-19 11:15:00 | 244.79 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-07-19 10:40:00 | 245.90 | 2023-07-19 15:10:00 | 245.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-20 11:15:00 | 244.67 | 2023-07-20 11:30:00 | 245.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-24 11:10:00 | 244.82 | 2023-07-24 11:35:00 | 245.30 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-25 10:05:00 | 246.30 | 2023-07-25 10:15:00 | 245.36 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-07-31 10:05:00 | 270.38 | 2023-07-31 10:15:00 | 269.17 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-08-01 09:30:00 | 270.88 | 2023-08-01 09:40:00 | 272.28 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-08-01 09:30:00 | 270.88 | 2023-08-01 12:45:00 | 274.27 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2023-08-08 10:00:00 | 293.72 | 2023-08-08 10:25:00 | 295.45 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-08-08 10:00:00 | 293.72 | 2023-08-08 10:30:00 | 293.72 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-21 10:20:00 | 291.63 | 2023-08-21 10:45:00 | 290.58 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-23 09:45:00 | 301.55 | 2023-08-23 09:50:00 | 303.18 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-08-23 09:45:00 | 301.55 | 2023-08-23 10:30:00 | 301.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-24 09:45:00 | 311.52 | 2023-08-24 09:55:00 | 310.32 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-08-30 09:40:00 | 305.23 | 2023-08-30 09:45:00 | 306.54 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-08-30 09:40:00 | 305.23 | 2023-08-30 15:20:00 | 322.07 | TARGET_HIT | 0.50 | 5.52% |
| BUY | retest1 | 2023-10-06 09:30:00 | 450.75 | 2023-10-06 09:35:00 | 448.67 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-10-17 10:30:00 | 493.08 | 2023-10-17 10:50:00 | 490.86 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-10-17 10:30:00 | 493.08 | 2023-10-17 11:35:00 | 493.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-19 10:50:00 | 497.33 | 2023-10-19 11:05:00 | 499.80 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-10-19 10:50:00 | 497.33 | 2023-10-19 11:10:00 | 497.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-20 10:05:00 | 523.70 | 2023-10-20 13:55:00 | 528.04 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2023-10-20 10:05:00 | 523.70 | 2023-10-20 15:20:00 | 528.42 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2023-11-06 11:00:00 | 592.10 | 2023-11-06 11:05:00 | 594.12 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-12-11 10:25:00 | 784.80 | 2023-12-11 10:30:00 | 789.81 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2023-12-12 10:20:00 | 773.33 | 2023-12-12 10:25:00 | 776.79 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-12-22 09:50:00 | 790.33 | 2023-12-22 09:55:00 | 787.11 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-12-26 10:15:00 | 780.92 | 2023-12-26 10:55:00 | 778.16 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-28 09:35:00 | 763.95 | 2023-12-28 10:05:00 | 761.19 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-01-03 10:15:00 | 749.95 | 2024-01-03 10:25:00 | 754.62 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-01-03 10:15:00 | 749.95 | 2024-01-03 15:20:00 | 750.33 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-01-04 11:00:00 | 757.92 | 2024-01-04 11:10:00 | 755.30 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-01-10 10:35:00 | 753.33 | 2024-01-10 10:40:00 | 756.06 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-01-11 09:45:00 | 758.17 | 2024-01-11 10:55:00 | 755.43 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-01-12 10:20:00 | 748.50 | 2024-01-12 10:45:00 | 745.76 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-01-12 10:20:00 | 748.50 | 2024-01-12 14:40:00 | 748.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-15 10:00:00 | 751.00 | 2024-01-15 10:55:00 | 749.12 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-19 10:55:00 | 759.33 | 2024-01-19 11:15:00 | 756.08 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-01-19 10:55:00 | 759.33 | 2024-01-19 15:20:00 | 758.63 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-01-29 10:15:00 | 719.40 | 2024-01-29 10:25:00 | 717.11 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-02-15 09:55:00 | 786.00 | 2024-02-15 10:40:00 | 780.40 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-02-15 09:55:00 | 786.00 | 2024-02-15 15:20:00 | 777.00 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2024-02-20 10:20:00 | 776.52 | 2024-02-20 10:35:00 | 773.69 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-02-21 10:30:00 | 757.85 | 2024-02-21 10:40:00 | 754.96 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-02-21 10:30:00 | 757.85 | 2024-02-21 10:50:00 | 757.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-27 11:15:00 | 775.00 | 2024-02-27 11:20:00 | 779.25 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-02-27 11:15:00 | 775.00 | 2024-02-27 11:30:00 | 775.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-01 11:00:00 | 783.25 | 2024-03-01 12:40:00 | 786.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-03-06 10:00:00 | 754.90 | 2024-03-06 13:40:00 | 758.29 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-03-18 09:30:00 | 675.68 | 2024-03-18 09:40:00 | 680.46 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2024-04-08 11:05:00 | 958.27 | 2024-04-08 13:15:00 | 953.14 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-04-08 11:05:00 | 958.27 | 2024-04-08 15:20:00 | 933.57 | TARGET_HIT | 0.50 | 2.58% |
| SELL | retest1 | 2024-05-07 11:10:00 | 930.77 | 2024-05-07 11:15:00 | 925.93 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-05-07 11:10:00 | 930.77 | 2024-05-07 11:20:00 | 930.77 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-10 09:35:00 | 885.33 | 2024-05-10 09:40:00 | 890.90 | STOP_HIT | 1.00 | -0.63% |
