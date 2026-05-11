# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35296 bars)
- **Last close:** 1100.55
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
| ENTRY1 | 47 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 11 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 36
- **Target hits / Stop hits / Partials:** 11 / 36 / 18
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 12.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 8 | 29.6% | 2 | 19 | 6 | 0.09% | 2.3% |
| BUY @ 2nd Alert (retest1) | 27 | 8 | 29.6% | 2 | 19 | 6 | 0.09% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 21 | 55.3% | 9 | 17 | 12 | 0.27% | 10.3% |
| SELL @ 2nd Alert (retest1) | 38 | 21 | 55.3% | 9 | 17 | 12 | 0.27% | 10.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 65 | 29 | 44.6% | 11 | 36 | 18 | 0.19% | 12.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 11:15:00 | 478.95 | 475.03 | 0.00 | ORB-long ORB[472.05,477.45] vol=5.6x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:20:00 | 481.80 | 475.48 | 0.00 | T1 1.5R @ 481.80 |
| Stop hit — per-position SL triggered | 2024-05-30 11:25:00 | 478.95 | 475.64 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:45:00 | 475.00 | 477.49 | 0.00 | ORB-short ORB[476.00,480.45] vol=11.2x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-05-31 11:00:00 | 476.91 | 476.71 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:45:00 | 466.35 | 470.01 | 0.00 | ORB-short ORB[468.50,473.85] vol=1.8x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:05:00 | 462.96 | 468.36 | 0.00 | T1 1.5R @ 462.96 |
| Target hit | 2024-06-10 15:20:00 | 460.80 | 463.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-06-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:55:00 | 478.10 | 477.50 | 0.00 | ORB-long ORB[471.25,478.05] vol=9.9x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-06-12 10:00:00 | 475.95 | 477.30 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 11:00:00 | 478.60 | 471.82 | 0.00 | ORB-long ORB[467.00,471.00] vol=8.2x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-06-14 11:05:00 | 476.87 | 473.39 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:30:00 | 490.20 | 493.89 | 0.00 | ORB-short ORB[492.00,497.70] vol=1.7x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-06-26 10:40:00 | 492.35 | 493.61 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 509.05 | 503.38 | 0.00 | ORB-long ORB[497.85,503.10] vol=4.0x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:00:00 | 512.55 | 505.76 | 0.00 | T1 1.5R @ 512.55 |
| Stop hit — per-position SL triggered | 2024-06-27 10:45:00 | 509.05 | 510.44 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 09:30:00 | 505.50 | 507.88 | 0.00 | ORB-short ORB[506.95,509.90] vol=2.2x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 10:20:00 | 502.89 | 506.29 | 0.00 | T1 1.5R @ 502.89 |
| Stop hit — per-position SL triggered | 2024-06-28 10:30:00 | 505.50 | 505.84 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 10:45:00 | 491.95 | 495.52 | 0.00 | ORB-short ORB[496.05,499.70] vol=1.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 493.24 | 495.06 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:50:00 | 489.65 | 494.45 | 0.00 | ORB-short ORB[494.05,499.40] vol=1.5x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:00:00 | 487.51 | 493.73 | 0.00 | T1 1.5R @ 487.51 |
| Target hit | 2024-07-02 15:20:00 | 486.35 | 488.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-07-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:25:00 | 495.50 | 492.59 | 0.00 | ORB-long ORB[488.50,493.65] vol=4.9x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:15:00 | 497.99 | 495.45 | 0.00 | T1 1.5R @ 497.99 |
| Target hit | 2024-07-03 14:50:00 | 496.40 | 497.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 502.05 | 500.69 | 0.00 | ORB-long ORB[497.30,501.90] vol=2.9x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-07-04 10:25:00 | 499.98 | 501.26 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:15:00 | 510.00 | 505.24 | 0.00 | ORB-long ORB[502.00,507.60] vol=6.6x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-07-05 11:30:00 | 507.94 | 506.31 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 495.35 | 501.36 | 0.00 | ORB-short ORB[500.20,506.00] vol=3.0x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 497.65 | 499.16 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:30:00 | 490.95 | 495.22 | 0.00 | ORB-short ORB[495.10,501.40] vol=4.6x ATR=1.99 |
| Target hit | 2024-07-09 15:20:00 | 490.80 | 492.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:40:00 | 490.05 | 491.09 | 0.00 | ORB-short ORB[490.10,493.30] vol=1.8x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:55:00 | 488.15 | 490.43 | 0.00 | T1 1.5R @ 488.15 |
| Target hit | 2024-07-10 10:55:00 | 487.40 | 487.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — SELL (started 2024-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:30:00 | 487.70 | 490.79 | 0.00 | ORB-short ORB[489.75,494.15] vol=2.1x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:15:00 | 485.21 | 488.47 | 0.00 | T1 1.5R @ 485.21 |
| Target hit | 2024-07-11 15:20:00 | 483.00 | 485.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-07-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:05:00 | 464.90 | 467.10 | 0.00 | ORB-short ORB[465.00,469.65] vol=2.2x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 466.61 | 467.03 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:40:00 | 485.85 | 480.74 | 0.00 | ORB-long ORB[472.50,478.65] vol=3.7x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-07-26 09:45:00 | 483.82 | 482.44 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:10:00 | 497.10 | 492.87 | 0.00 | ORB-long ORB[491.00,494.90] vol=4.5x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-07-31 11:25:00 | 495.31 | 493.86 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:05:00 | 408.25 | 410.56 | 0.00 | ORB-short ORB[412.10,418.00] vol=7.1x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:05:00 | 405.28 | 410.02 | 0.00 | T1 1.5R @ 405.28 |
| Target hit | 2024-08-08 15:20:00 | 402.80 | 406.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-08-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:30:00 | 420.30 | 418.14 | 0.00 | ORB-long ORB[414.20,419.75] vol=2.1x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-08-19 09:35:00 | 418.63 | 419.39 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:15:00 | 431.50 | 427.55 | 0.00 | ORB-long ORB[423.85,428.10] vol=1.9x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-08-20 10:50:00 | 429.79 | 428.86 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:50:00 | 445.45 | 441.77 | 0.00 | ORB-long ORB[438.00,442.90] vol=5.5x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-08-23 10:55:00 | 444.05 | 441.94 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 11:15:00 | 436.80 | 435.25 | 0.00 | ORB-long ORB[430.95,436.00] vol=2.2x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 11:25:00 | 438.81 | 435.41 | 0.00 | T1 1.5R @ 438.81 |
| Stop hit — per-position SL triggered | 2024-08-27 11:30:00 | 436.80 | 435.42 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:40:00 | 442.60 | 439.32 | 0.00 | ORB-long ORB[435.60,441.95] vol=1.7x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-08-28 09:45:00 | 441.01 | 439.59 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 435.55 | 438.99 | 0.00 | ORB-short ORB[438.45,442.30] vol=1.8x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-08-29 11:00:00 | 436.57 | 438.91 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 09:35:00 | 438.90 | 441.07 | 0.00 | ORB-short ORB[439.85,444.00] vol=1.9x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:15:00 | 436.45 | 440.13 | 0.00 | T1 1.5R @ 436.45 |
| Stop hit — per-position SL triggered | 2024-09-02 14:20:00 | 438.90 | 438.16 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:15:00 | 424.95 | 427.12 | 0.00 | ORB-short ORB[425.05,429.00] vol=2.2x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:00:00 | 422.79 | 426.16 | 0.00 | T1 1.5R @ 422.79 |
| Target hit | 2024-09-10 15:20:00 | 419.95 | 421.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2024-09-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:45:00 | 422.70 | 424.05 | 0.00 | ORB-short ORB[422.85,426.00] vol=1.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-09-12 11:15:00 | 423.71 | 423.85 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 11:00:00 | 440.00 | 442.92 | 0.00 | ORB-short ORB[441.95,446.15] vol=2.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-09-24 11:25:00 | 441.23 | 442.57 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:35:00 | 456.75 | 452.47 | 0.00 | ORB-long ORB[449.20,453.85] vol=2.0x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:40:00 | 459.22 | 454.66 | 0.00 | T1 1.5R @ 459.22 |
| Stop hit — per-position SL triggered | 2024-09-27 10:45:00 | 456.75 | 454.86 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:30:00 | 424.65 | 421.64 | 0.00 | ORB-long ORB[417.30,422.95] vol=1.6x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-10-03 09:55:00 | 422.83 | 422.33 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:35:00 | 416.70 | 412.40 | 0.00 | ORB-long ORB[409.20,414.40] vol=2.3x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-10-10 10:40:00 | 415.29 | 412.58 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:50:00 | 416.10 | 414.47 | 0.00 | ORB-long ORB[411.65,415.80] vol=1.6x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-10-11 10:00:00 | 414.67 | 414.51 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:15:00 | 413.40 | 416.24 | 0.00 | ORB-short ORB[415.05,419.05] vol=3.7x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 10:25:00 | 411.34 | 414.95 | 0.00 | T1 1.5R @ 411.34 |
| Stop hit — per-position SL triggered | 2024-10-14 10:35:00 | 413.40 | 414.68 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:05:00 | 410.55 | 411.60 | 0.00 | ORB-short ORB[411.00,413.90] vol=2.2x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:45:00 | 408.87 | 411.22 | 0.00 | T1 1.5R @ 408.87 |
| Stop hit — per-position SL triggered | 2024-10-16 11:05:00 | 410.55 | 411.09 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:50:00 | 404.90 | 407.29 | 0.00 | ORB-short ORB[406.20,410.90] vol=2.2x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-10-17 14:50:00 | 405.98 | 405.11 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 10:55:00 | 405.10 | 407.15 | 0.00 | ORB-short ORB[407.20,410.00] vol=3.0x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 12:35:00 | 403.06 | 406.88 | 0.00 | T1 1.5R @ 403.06 |
| Target hit | 2024-10-21 15:20:00 | 400.80 | 405.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2024-11-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 11:10:00 | 553.75 | 549.58 | 0.00 | ORB-long ORB[542.85,551.00] vol=1.8x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-11-12 11:15:00 | 552.13 | 549.66 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 09:50:00 | 527.85 | 534.88 | 0.00 | ORB-short ORB[535.50,540.90] vol=1.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-11-22 10:00:00 | 531.09 | 534.20 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:30:00 | 590.25 | 597.15 | 0.00 | ORB-short ORB[595.00,603.65] vol=1.7x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-12-26 10:40:00 | 592.95 | 596.89 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:10:00 | 623.45 | 628.56 | 0.00 | ORB-short ORB[625.20,634.45] vol=2.7x ATR=2.54 |
| Stop hit — per-position SL triggered | 2025-01-02 11:20:00 | 625.99 | 628.44 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:55:00 | 607.50 | 610.63 | 0.00 | ORB-short ORB[607.85,612.55] vol=1.7x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:30:00 | 603.94 | 609.56 | 0.00 | T1 1.5R @ 603.94 |
| Target hit | 2025-01-09 15:20:00 | 595.75 | 605.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-01-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 11:05:00 | 466.85 | 467.01 | 0.00 | ORB-short ORB[467.25,473.30] vol=1.9x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 469.18 | 467.03 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 425.35 | 422.67 | 0.00 | ORB-long ORB[419.60,424.90] vol=2.1x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:50:00 | 428.70 | 423.95 | 0.00 | T1 1.5R @ 428.70 |
| Target hit | 2025-03-18 15:20:00 | 443.55 | 438.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 474.75 | 472.58 | 0.00 | ORB-long ORB[468.40,473.95] vol=1.7x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-03-21 10:00:00 | 472.65 | 472.92 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-30 11:15:00 | 478.95 | 2024-05-30 11:20:00 | 481.80 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-05-30 11:15:00 | 478.95 | 2024-05-30 11:25:00 | 478.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 10:45:00 | 475.00 | 2024-05-31 11:00:00 | 476.91 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-06-10 09:45:00 | 466.35 | 2024-06-10 10:05:00 | 462.96 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-06-10 09:45:00 | 466.35 | 2024-06-10 15:20:00 | 460.80 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2024-06-12 09:55:00 | 478.10 | 2024-06-12 10:00:00 | 475.95 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-06-14 11:00:00 | 478.60 | 2024-06-14 11:05:00 | 476.87 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-26 10:30:00 | 490.20 | 2024-06-26 10:40:00 | 492.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-06-27 09:45:00 | 509.05 | 2024-06-27 10:00:00 | 512.55 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-06-27 09:45:00 | 509.05 | 2024-06-27 10:45:00 | 509.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-28 09:30:00 | 505.50 | 2024-06-28 10:20:00 | 502.89 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-06-28 09:30:00 | 505.50 | 2024-06-28 10:30:00 | 505.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-01 10:45:00 | 491.95 | 2024-07-01 11:15:00 | 493.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-02 10:50:00 | 489.65 | 2024-07-02 11:00:00 | 487.51 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-02 10:50:00 | 489.65 | 2024-07-02 15:20:00 | 486.35 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2024-07-03 10:25:00 | 495.50 | 2024-07-03 11:15:00 | 497.99 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-03 10:25:00 | 495.50 | 2024-07-03 14:50:00 | 496.40 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2024-07-04 09:30:00 | 502.05 | 2024-07-04 10:25:00 | 499.98 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-07-05 11:15:00 | 510.00 | 2024-07-05 11:30:00 | 507.94 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-08 09:40:00 | 495.35 | 2024-07-08 10:00:00 | 497.65 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-07-09 10:30:00 | 490.95 | 2024-07-09 15:20:00 | 490.80 | TARGET_HIT | 1.00 | 0.03% |
| SELL | retest1 | 2024-07-10 09:40:00 | 490.05 | 2024-07-10 09:55:00 | 488.15 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-10 09:40:00 | 490.05 | 2024-07-10 10:55:00 | 487.40 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2024-07-11 09:30:00 | 487.70 | 2024-07-11 10:15:00 | 485.21 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-11 09:30:00 | 487.70 | 2024-07-11 15:20:00 | 483.00 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2024-07-25 10:05:00 | 464.90 | 2024-07-25 10:15:00 | 466.61 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-26 09:40:00 | 485.85 | 2024-07-26 09:45:00 | 483.82 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-07-31 11:10:00 | 497.10 | 2024-07-31 11:25:00 | 495.31 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-08 10:05:00 | 408.25 | 2024-08-08 11:05:00 | 405.28 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-08-08 10:05:00 | 408.25 | 2024-08-08 15:20:00 | 402.80 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2024-08-19 09:30:00 | 420.30 | 2024-08-19 09:35:00 | 418.63 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-20 10:15:00 | 431.50 | 2024-08-20 10:50:00 | 429.79 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-23 10:50:00 | 445.45 | 2024-08-23 10:55:00 | 444.05 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-27 11:15:00 | 436.80 | 2024-08-27 11:25:00 | 438.81 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-27 11:15:00 | 436.80 | 2024-08-27 11:30:00 | 436.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-28 09:40:00 | 442.60 | 2024-08-28 09:45:00 | 441.01 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-29 10:55:00 | 435.55 | 2024-08-29 11:00:00 | 436.57 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-02 09:35:00 | 438.90 | 2024-09-02 10:15:00 | 436.45 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-09-02 09:35:00 | 438.90 | 2024-09-02 14:20:00 | 438.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-10 10:15:00 | 424.95 | 2024-09-10 11:00:00 | 422.79 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-10 10:15:00 | 424.95 | 2024-09-10 15:20:00 | 419.95 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2024-09-12 10:45:00 | 422.70 | 2024-09-12 11:15:00 | 423.71 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-24 11:00:00 | 440.00 | 2024-09-24 11:25:00 | 441.23 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-27 10:35:00 | 456.75 | 2024-09-27 10:40:00 | 459.22 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-09-27 10:35:00 | 456.75 | 2024-09-27 10:45:00 | 456.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-03 09:30:00 | 424.65 | 2024-10-03 09:55:00 | 422.83 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-10-10 10:35:00 | 416.70 | 2024-10-10 10:40:00 | 415.29 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-11 09:50:00 | 416.10 | 2024-10-11 10:00:00 | 414.67 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-14 10:15:00 | 413.40 | 2024-10-14 10:25:00 | 411.34 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-14 10:15:00 | 413.40 | 2024-10-14 10:35:00 | 413.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 10:05:00 | 410.55 | 2024-10-16 10:45:00 | 408.87 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-16 10:05:00 | 410.55 | 2024-10-16 11:05:00 | 410.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:50:00 | 404.90 | 2024-10-17 14:50:00 | 405.98 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-21 10:55:00 | 405.10 | 2024-10-21 12:35:00 | 403.06 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-21 10:55:00 | 405.10 | 2024-10-21 15:20:00 | 400.80 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2024-11-12 11:10:00 | 553.75 | 2024-11-12 11:15:00 | 552.13 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-22 09:50:00 | 527.85 | 2024-11-22 10:00:00 | 531.09 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2024-12-26 10:30:00 | 590.25 | 2024-12-26 10:40:00 | 592.95 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-01-02 11:10:00 | 623.45 | 2025-01-02 11:20:00 | 625.99 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-09 10:55:00 | 607.50 | 2025-01-09 11:30:00 | 603.94 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-09 10:55:00 | 607.50 | 2025-01-09 15:20:00 | 595.75 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2025-01-24 11:05:00 | 466.85 | 2025-01-24 11:15:00 | 469.18 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-03-18 09:35:00 | 425.35 | 2025-03-18 09:50:00 | 428.70 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-03-18 09:35:00 | 425.35 | 2025-03-18 15:20:00 | 443.55 | TARGET_HIT | 0.50 | 4.28% |
| BUY | retest1 | 2025-03-21 09:40:00 | 474.75 | 2025-03-21 10:00:00 | 472.65 | STOP_HIT | 1.00 | -0.44% |
