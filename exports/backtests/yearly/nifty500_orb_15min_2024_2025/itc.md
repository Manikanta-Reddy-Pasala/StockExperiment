# ITC Ltd. (ITC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-09-06 15:25:00 (6096 bars)
- **Last close:** 502.25
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
| ENTRY1 | 42 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 6 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 36
- **Target hits / Stop hits / Partials:** 6 / 36 / 13
- **Avg / median % per leg:** 0.03% / -0.17%
- **Sum % (uncompounded):** 1.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 13 | 33.3% | 3 | 26 | 10 | 0.04% | 1.7% |
| BUY @ 2nd Alert (retest1) | 39 | 13 | 33.3% | 3 | 26 | 10 | 0.04% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 6 | 37.5% | 3 | 10 | 3 | -0.00% | -0.1% |
| SELL @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 3 | 10 | 3 | -0.00% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 55 | 19 | 34.5% | 6 | 36 | 13 | 0.03% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:15:00 | 432.60 | 430.10 | 0.00 | ORB-long ORB[429.00,431.80] vol=1.9x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-05-14 11:20:00 | 431.71 | 430.15 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 426.65 | 428.64 | 0.00 | ORB-short ORB[428.10,430.45] vol=2.3x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-05-16 10:15:00 | 427.72 | 427.52 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:45:00 | 435.55 | 432.47 | 0.00 | ORB-long ORB[428.90,433.45] vol=2.0x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-05-17 11:50:00 | 434.47 | 433.04 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 439.30 | 437.57 | 0.00 | ORB-long ORB[435.40,437.75] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-05-22 09:40:00 | 438.42 | 438.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 426.70 | 428.50 | 0.00 | ORB-short ORB[427.20,430.95] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-05-30 09:35:00 | 427.57 | 428.26 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 428.50 | 427.76 | 0.00 | ORB-long ORB[425.15,428.15] vol=2.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-05-31 09:55:00 | 427.60 | 427.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-03 11:15:00 | 429.35 | 431.63 | 0.00 | ORB-short ORB[432.00,434.90] vol=1.8x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-06-03 12:05:00 | 430.26 | 431.38 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 10:00:00 | 429.65 | 423.75 | 0.00 | ORB-long ORB[418.05,423.90] vol=1.6x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 10:35:00 | 433.84 | 426.66 | 0.00 | T1 1.5R @ 433.84 |
| Stop hit — per-position SL triggered | 2024-06-05 11:55:00 | 429.65 | 429.29 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:00:00 | 436.90 | 433.54 | 0.00 | ORB-long ORB[431.10,436.30] vol=1.9x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-06-06 10:40:00 | 435.27 | 434.23 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:05:00 | 436.60 | 434.21 | 0.00 | ORB-long ORB[431.10,436.15] vol=1.7x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-06-07 11:25:00 | 435.44 | 434.43 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:40:00 | 421.90 | 423.38 | 0.00 | ORB-short ORB[422.30,425.30] vol=1.6x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 12:40:00 | 420.60 | 422.48 | 0.00 | T1 1.5R @ 420.60 |
| Target hit | 2024-06-21 15:20:00 | 419.35 | 420.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-06-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:45:00 | 422.55 | 423.26 | 0.00 | ORB-short ORB[422.65,424.00] vol=2.1x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:05:00 | 421.75 | 423.01 | 0.00 | T1 1.5R @ 421.75 |
| Target hit | 2024-06-25 12:20:00 | 421.70 | 421.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2024-06-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:40:00 | 425.40 | 423.63 | 0.00 | ORB-long ORB[422.55,424.05] vol=3.2x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-06-26 11:55:00 | 424.69 | 424.39 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 11:10:00 | 429.00 | 427.81 | 0.00 | ORB-long ORB[425.50,428.70] vol=4.2x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-07-03 11:20:00 | 428.30 | 427.86 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:50:00 | 430.15 | 428.75 | 0.00 | ORB-long ORB[427.05,430.00] vol=3.5x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 11:10:00 | 431.08 | 429.08 | 0.00 | T1 1.5R @ 431.08 |
| Stop hit — per-position SL triggered | 2024-07-04 11:30:00 | 430.15 | 429.22 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:50:00 | 437.05 | 435.22 | 0.00 | ORB-long ORB[433.65,436.00] vol=1.8x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 09:55:00 | 438.47 | 435.69 | 0.00 | T1 1.5R @ 438.47 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 437.05 | 435.81 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:15:00 | 449.15 | 446.42 | 0.00 | ORB-long ORB[444.50,448.00] vol=2.2x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-07-09 10:35:00 | 447.97 | 446.91 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 447.05 | 451.46 | 0.00 | ORB-short ORB[451.10,455.35] vol=1.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 448.11 | 450.93 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 10:15:00 | 452.70 | 450.90 | 0.00 | ORB-long ORB[448.70,451.45] vol=1.8x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:20:00 | 454.31 | 451.38 | 0.00 | T1 1.5R @ 454.31 |
| Target hit | 2024-07-11 15:20:00 | 459.00 | 455.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-07-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:50:00 | 459.45 | 459.23 | 0.00 | ORB-long ORB[456.50,459.00] vol=1.7x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:55:00 | 461.19 | 459.43 | 0.00 | T1 1.5R @ 461.19 |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 459.45 | 459.87 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:00:00 | 462.25 | 460.13 | 0.00 | ORB-long ORB[457.20,459.70] vol=1.7x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 14:25:00 | 464.06 | 462.71 | 0.00 | T1 1.5R @ 464.06 |
| Stop hit — per-position SL triggered | 2024-07-15 15:15:00 | 462.25 | 462.89 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 466.00 | 464.28 | 0.00 | ORB-long ORB[461.40,465.20] vol=1.6x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-07-16 10:50:00 | 465.07 | 464.54 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 11:10:00 | 467.25 | 466.00 | 0.00 | ORB-long ORB[462.65,466.95] vol=4.2x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 12:05:00 | 468.48 | 466.36 | 0.00 | T1 1.5R @ 468.48 |
| Target hit | 2024-07-18 15:20:00 | 470.60 | 467.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-07-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 10:45:00 | 475.50 | 473.81 | 0.00 | ORB-long ORB[469.40,475.35] vol=1.7x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:55:00 | 477.32 | 474.11 | 0.00 | T1 1.5R @ 477.32 |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 475.50 | 475.29 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-07-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:50:00 | 471.00 | 470.34 | 0.00 | ORB-long ORB[466.40,470.20] vol=1.9x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-07-23 10:00:00 | 469.68 | 470.31 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-07-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 11:05:00 | 485.50 | 491.10 | 0.00 | ORB-short ORB[489.15,496.00] vol=2.0x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 487.04 | 490.54 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-07-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:45:00 | 495.15 | 491.86 | 0.00 | ORB-long ORB[488.05,494.75] vol=1.9x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-07-26 11:35:00 | 493.99 | 493.18 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:15:00 | 491.85 | 493.36 | 0.00 | ORB-short ORB[493.15,496.75] vol=2.2x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-08-01 11:40:00 | 492.91 | 493.24 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 10:50:00 | 487.50 | 487.25 | 0.00 | ORB-long ORB[479.55,486.85] vol=1.5x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-08-05 10:55:00 | 485.67 | 487.19 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:50:00 | 486.15 | 487.46 | 0.00 | ORB-short ORB[486.25,489.95] vol=2.3x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-08-06 10:55:00 | 487.25 | 487.43 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:35:00 | 495.45 | 492.67 | 0.00 | ORB-long ORB[488.10,494.30] vol=1.6x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-08-07 10:45:00 | 494.11 | 492.84 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:10:00 | 497.95 | 495.38 | 0.00 | ORB-long ORB[492.00,496.65] vol=2.4x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 11:25:00 | 499.27 | 495.91 | 0.00 | T1 1.5R @ 499.27 |
| Stop hit — per-position SL triggered | 2024-08-12 11:45:00 | 497.95 | 496.35 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:25:00 | 493.00 | 491.93 | 0.00 | ORB-long ORB[490.05,492.95] vol=2.3x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 11:00:00 | 494.53 | 492.40 | 0.00 | T1 1.5R @ 494.53 |
| Target hit | 2024-08-14 12:10:00 | 494.90 | 495.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2024-08-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:30:00 | 508.10 | 506.69 | 0.00 | ORB-long ORB[504.55,507.55] vol=1.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-08-19 09:50:00 | 506.65 | 507.28 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-08-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:25:00 | 504.50 | 502.12 | 0.00 | ORB-long ORB[499.05,501.25] vol=4.4x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-08-21 11:10:00 | 503.58 | 502.76 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 500.45 | 502.99 | 0.00 | ORB-short ORB[502.65,506.95] vol=1.8x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-08-23 09:40:00 | 501.54 | 502.61 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-08-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:05:00 | 504.00 | 505.64 | 0.00 | ORB-short ORB[505.00,507.60] vol=1.8x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:45:00 | 502.65 | 504.25 | 0.00 | T1 1.5R @ 502.65 |
| Target hit | 2024-08-27 15:20:00 | 500.70 | 502.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2024-08-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:30:00 | 498.70 | 500.62 | 0.00 | ORB-short ORB[499.05,502.30] vol=1.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-08-28 10:35:00 | 499.56 | 500.54 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 501.20 | 499.13 | 0.00 | ORB-long ORB[497.35,499.60] vol=2.2x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-08-29 11:05:00 | 500.28 | 499.41 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:00:00 | 512.00 | 508.65 | 0.00 | ORB-long ORB[503.30,509.00] vol=3.0x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-09-02 10:05:00 | 510.52 | 508.92 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 515.65 | 512.93 | 0.00 | ORB-long ORB[508.45,514.00] vol=3.1x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-09-03 09:45:00 | 514.38 | 513.41 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-09-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:05:00 | 502.15 | 506.42 | 0.00 | ORB-short ORB[508.75,512.00] vol=1.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-09-06 11:35:00 | 503.42 | 505.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 11:15:00 | 432.60 | 2024-05-14 11:20:00 | 431.71 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-16 09:30:00 | 426.65 | 2024-05-16 10:15:00 | 427.72 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-17 10:45:00 | 435.55 | 2024-05-17 11:50:00 | 434.47 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-22 09:35:00 | 439.30 | 2024-05-22 09:40:00 | 438.42 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-05-30 09:30:00 | 426.70 | 2024-05-30 09:35:00 | 427.57 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-05-31 09:45:00 | 428.50 | 2024-05-31 09:55:00 | 427.60 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-06-03 11:15:00 | 429.35 | 2024-06-03 12:05:00 | 430.26 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-06-05 10:00:00 | 429.65 | 2024-06-05 10:35:00 | 433.84 | PARTIAL | 0.50 | 0.97% |
| BUY | retest1 | 2024-06-05 10:00:00 | 429.65 | 2024-06-05 11:55:00 | 429.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-06 10:00:00 | 436.90 | 2024-06-06 10:40:00 | 435.27 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-07 11:05:00 | 436.60 | 2024-06-07 11:25:00 | 435.44 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-21 10:40:00 | 421.90 | 2024-06-21 12:40:00 | 420.60 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-06-21 10:40:00 | 421.90 | 2024-06-21 15:20:00 | 419.35 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-06-25 10:45:00 | 422.55 | 2024-06-25 11:05:00 | 421.75 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2024-06-25 10:45:00 | 422.55 | 2024-06-25 12:20:00 | 421.70 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-06-26 10:40:00 | 425.40 | 2024-06-26 11:55:00 | 424.69 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-07-03 11:10:00 | 429.00 | 2024-07-03 11:20:00 | 428.30 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-07-04 10:50:00 | 430.15 | 2024-07-04 11:10:00 | 431.08 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2024-07-04 10:50:00 | 430.15 | 2024-07-04 11:30:00 | 430.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 09:50:00 | 437.05 | 2024-07-08 09:55:00 | 438.47 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-07-08 09:50:00 | 437.05 | 2024-07-08 10:00:00 | 437.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 10:15:00 | 449.15 | 2024-07-09 10:35:00 | 447.97 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-10 10:35:00 | 447.05 | 2024-07-10 10:45:00 | 448.11 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-11 10:15:00 | 452.70 | 2024-07-11 10:20:00 | 454.31 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-07-11 10:15:00 | 452.70 | 2024-07-11 15:20:00 | 459.00 | TARGET_HIT | 0.50 | 1.39% |
| BUY | retest1 | 2024-07-12 10:50:00 | 459.45 | 2024-07-12 10:55:00 | 461.19 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-07-12 10:50:00 | 459.45 | 2024-07-12 11:15:00 | 459.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-15 10:00:00 | 462.25 | 2024-07-15 14:25:00 | 464.06 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-15 10:00:00 | 462.25 | 2024-07-15 15:15:00 | 462.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 10:30:00 | 466.00 | 2024-07-16 10:50:00 | 465.07 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-18 11:10:00 | 467.25 | 2024-07-18 12:05:00 | 468.48 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-07-18 11:10:00 | 467.25 | 2024-07-18 15:20:00 | 470.60 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2024-07-19 10:45:00 | 475.50 | 2024-07-19 10:55:00 | 477.32 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-07-19 10:45:00 | 475.50 | 2024-07-19 12:15:00 | 475.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 09:50:00 | 471.00 | 2024-07-23 10:00:00 | 469.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-25 11:05:00 | 485.50 | 2024-07-25 11:15:00 | 487.04 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-26 10:45:00 | 495.15 | 2024-07-26 11:35:00 | 493.99 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-01 11:15:00 | 491.85 | 2024-08-01 11:40:00 | 492.91 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-05 10:50:00 | 487.50 | 2024-08-05 10:55:00 | 485.67 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-06 10:50:00 | 486.15 | 2024-08-06 10:55:00 | 487.25 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-07 10:35:00 | 495.45 | 2024-08-07 10:45:00 | 494.11 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-12 11:10:00 | 497.95 | 2024-08-12 11:25:00 | 499.27 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-08-12 11:10:00 | 497.95 | 2024-08-12 11:45:00 | 497.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-14 10:25:00 | 493.00 | 2024-08-14 11:00:00 | 494.53 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-08-14 10:25:00 | 493.00 | 2024-08-14 12:10:00 | 494.90 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-08-19 09:30:00 | 508.10 | 2024-08-19 09:50:00 | 506.65 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-21 10:25:00 | 504.50 | 2024-08-21 11:10:00 | 503.58 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-08-23 09:30:00 | 500.45 | 2024-08-23 09:40:00 | 501.54 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-08-27 10:05:00 | 504.00 | 2024-08-27 10:45:00 | 502.65 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-08-27 10:05:00 | 504.00 | 2024-08-27 15:20:00 | 500.70 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-08-28 10:30:00 | 498.70 | 2024-08-28 10:35:00 | 499.56 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-29 10:55:00 | 501.20 | 2024-08-29 11:05:00 | 500.28 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-09-02 10:00:00 | 512.00 | 2024-09-02 10:05:00 | 510.52 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-03 09:35:00 | 515.65 | 2024-09-03 09:45:00 | 514.38 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-06 11:05:00 | 502.15 | 2024-09-06 11:35:00 | 503.42 | STOP_HIT | 1.00 | -0.25% |
