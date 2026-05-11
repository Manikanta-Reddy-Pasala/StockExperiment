# LT Foods Ltd. (LTFOODS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 427.00
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
| TARGET_HIT | 9 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 47
- **Target hits / Stop hits / Partials:** 9 / 47 / 22
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 7.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 12 | 38.7% | 3 | 19 | 9 | 0.05% | 1.6% |
| BUY @ 2nd Alert (retest1) | 31 | 12 | 38.7% | 3 | 19 | 9 | 0.05% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 19 | 40.4% | 6 | 28 | 13 | 0.12% | 5.7% |
| SELL @ 2nd Alert (retest1) | 47 | 19 | 40.4% | 6 | 28 | 13 | 0.12% | 5.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 78 | 31 | 39.7% | 9 | 47 | 22 | 0.09% | 7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-14 09:35:00 | 370.10 | 372.50 | 0.00 | ORB-short ORB[371.00,375.50] vol=1.5x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-05-14 10:00:00 | 371.65 | 371.77 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:40:00 | 418.55 | 422.57 | 0.00 | ORB-short ORB[421.05,427.00] vol=1.8x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-05-29 10:45:00 | 419.90 | 422.51 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:30:00 | 428.40 | 424.05 | 0.00 | ORB-long ORB[420.00,425.70] vol=5.2x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:35:00 | 431.17 | 428.67 | 0.00 | T1 1.5R @ 431.17 |
| Stop hit — per-position SL triggered | 2025-05-30 10:05:00 | 428.40 | 429.72 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:15:00 | 432.35 | 437.82 | 0.00 | ORB-short ORB[437.40,441.40] vol=3.5x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:25:00 | 429.67 | 435.35 | 0.00 | T1 1.5R @ 429.67 |
| Target hit | 2025-06-19 15:20:00 | 425.35 | 430.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 449.20 | 445.00 | 0.00 | ORB-long ORB[441.30,447.50] vol=1.6x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-06-27 09:35:00 | 447.23 | 445.59 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-08-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:50:00 | 479.80 | 484.46 | 0.00 | ORB-short ORB[485.00,490.05] vol=1.5x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-08-01 10:20:00 | 481.83 | 482.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-08-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:55:00 | 473.65 | 479.01 | 0.00 | ORB-short ORB[481.00,487.75] vol=3.2x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-08-05 11:40:00 | 475.30 | 477.66 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:35:00 | 467.80 | 474.73 | 0.00 | ORB-short ORB[473.70,480.45] vol=3.4x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:35:00 | 464.16 | 471.80 | 0.00 | T1 1.5R @ 464.16 |
| Stop hit — per-position SL triggered | 2025-08-06 13:05:00 | 467.80 | 467.69 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-08-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:40:00 | 455.90 | 458.60 | 0.00 | ORB-short ORB[457.05,462.90] vol=1.9x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 09:55:00 | 453.16 | 457.62 | 0.00 | T1 1.5R @ 453.16 |
| Stop hit — per-position SL triggered | 2025-08-14 10:25:00 | 455.90 | 456.94 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-08-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:45:00 | 444.90 | 447.63 | 0.00 | ORB-short ORB[446.55,452.60] vol=1.6x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-08-18 09:55:00 | 446.97 | 447.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:35:00 | 457.80 | 455.70 | 0.00 | ORB-long ORB[452.35,457.00] vol=2.6x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-08-21 09:40:00 | 456.15 | 455.86 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:35:00 | 443.20 | 446.78 | 0.00 | ORB-short ORB[446.80,452.10] vol=1.7x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 09:40:00 | 440.57 | 444.55 | 0.00 | T1 1.5R @ 440.57 |
| Stop hit — per-position SL triggered | 2025-08-22 09:45:00 | 443.20 | 444.32 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-09-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 11:05:00 | 463.70 | 461.42 | 0.00 | ORB-long ORB[459.60,463.00] vol=2.1x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 11:10:00 | 465.41 | 462.04 | 0.00 | T1 1.5R @ 465.41 |
| Stop hit — per-position SL triggered | 2025-09-17 11:55:00 | 463.70 | 462.82 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-09-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:05:00 | 461.60 | 463.76 | 0.00 | ORB-short ORB[463.00,467.80] vol=5.0x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-09-18 11:15:00 | 462.65 | 463.71 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 11:05:00 | 390.90 | 393.01 | 0.00 | ORB-short ORB[393.00,396.70] vol=1.8x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:55:00 | 389.38 | 392.52 | 0.00 | T1 1.5R @ 389.38 |
| Target hit | 2025-10-09 15:20:00 | 386.75 | 389.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 392.00 | 388.59 | 0.00 | ORB-long ORB[384.00,389.50] vol=1.8x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-10-10 09:35:00 | 390.55 | 388.87 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 403.60 | 400.47 | 0.00 | ORB-long ORB[399.05,401.60] vol=4.5x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:45:00 | 406.02 | 402.14 | 0.00 | T1 1.5R @ 406.02 |
| Stop hit — per-position SL triggered | 2025-10-14 09:50:00 | 403.60 | 402.35 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:40:00 | 418.80 | 415.31 | 0.00 | ORB-long ORB[412.20,415.00] vol=2.6x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:45:00 | 420.92 | 417.48 | 0.00 | T1 1.5R @ 420.92 |
| Stop hit — per-position SL triggered | 2025-10-17 09:50:00 | 418.80 | 417.70 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-11-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:05:00 | 404.00 | 406.20 | 0.00 | ORB-short ORB[404.10,408.90] vol=1.5x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-11-06 12:10:00 | 405.83 | 405.69 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 400.30 | 402.58 | 0.00 | ORB-short ORB[402.15,408.00] vol=2.8x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-11-07 09:50:00 | 401.89 | 401.95 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-11-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:50:00 | 402.55 | 404.05 | 0.00 | ORB-short ORB[403.00,406.65] vol=1.9x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:15:00 | 400.52 | 402.95 | 0.00 | T1 1.5R @ 400.52 |
| Stop hit — per-position SL triggered | 2025-11-10 13:10:00 | 402.55 | 400.76 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:35:00 | 409.25 | 411.63 | 0.00 | ORB-short ORB[410.50,416.15] vol=2.2x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-11-13 12:35:00 | 410.60 | 409.78 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-11-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:55:00 | 407.00 | 410.29 | 0.00 | ORB-short ORB[409.80,415.25] vol=1.6x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:05:00 | 405.43 | 409.47 | 0.00 | T1 1.5R @ 405.43 |
| Target hit | 2025-11-18 11:40:00 | 406.25 | 406.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:15:00 | 419.40 | 414.24 | 0.00 | ORB-long ORB[407.95,412.00] vol=6.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-11-19 10:30:00 | 417.99 | 415.41 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-11-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:10:00 | 417.15 | 415.91 | 0.00 | ORB-long ORB[414.60,416.60] vol=1.9x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-11-20 11:10:00 | 415.89 | 415.99 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-11-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:55:00 | 404.80 | 405.52 | 0.00 | ORB-short ORB[405.80,408.95] vol=2.2x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-11-24 12:20:00 | 405.63 | 405.37 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-12-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:00:00 | 408.75 | 410.69 | 0.00 | ORB-short ORB[410.15,416.10] vol=1.9x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 409.76 | 410.39 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-12-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:20:00 | 406.60 | 403.03 | 0.00 | ORB-long ORB[400.25,402.90] vol=2.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 405.38 | 404.11 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-12-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:55:00 | 391.15 | 393.95 | 0.00 | ORB-short ORB[392.75,395.90] vol=2.7x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:10:00 | 389.15 | 393.53 | 0.00 | T1 1.5R @ 389.15 |
| Stop hit — per-position SL triggered | 2025-12-08 11:35:00 | 391.15 | 392.70 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 11:15:00 | 379.60 | 382.17 | 0.00 | ORB-short ORB[381.30,384.10] vol=1.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-12-12 11:20:00 | 380.45 | 382.02 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-12-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:55:00 | 387.45 | 384.54 | 0.00 | ORB-long ORB[381.05,385.95] vol=3.9x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:25:00 | 389.29 | 385.67 | 0.00 | T1 1.5R @ 389.29 |
| Stop hit — per-position SL triggered | 2025-12-16 11:30:00 | 387.45 | 385.72 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:50:00 | 391.00 | 392.27 | 0.00 | ORB-short ORB[391.10,394.60] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-12-18 10:05:00 | 392.21 | 391.93 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 397.85 | 396.47 | 0.00 | ORB-long ORB[393.25,397.70] vol=1.9x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-12-19 09:50:00 | 396.46 | 396.72 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-12-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:55:00 | 402.00 | 402.45 | 0.00 | ORB-short ORB[402.30,405.25] vol=2.2x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:10:00 | 400.36 | 401.99 | 0.00 | T1 1.5R @ 400.36 |
| Stop hit — per-position SL triggered | 2025-12-24 12:45:00 | 402.00 | 402.25 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:15:00 | 398.05 | 400.46 | 0.00 | ORB-short ORB[399.00,404.80] vol=1.7x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-12-29 11:45:00 | 398.82 | 400.21 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:45:00 | 383.70 | 388.69 | 0.00 | ORB-short ORB[390.15,394.85] vol=1.5x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:10:00 | 381.85 | 386.32 | 0.00 | T1 1.5R @ 381.85 |
| Stop hit — per-position SL triggered | 2025-12-30 11:25:00 | 383.70 | 385.15 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2026-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:05:00 | 387.45 | 388.11 | 0.00 | ORB-short ORB[388.00,389.95] vol=1.8x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 12:05:00 | 385.96 | 387.81 | 0.00 | T1 1.5R @ 385.96 |
| Target hit | 2026-01-01 15:05:00 | 386.65 | 386.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2026-01-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:40:00 | 378.15 | 381.63 | 0.00 | ORB-short ORB[381.00,385.50] vol=3.5x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:10:00 | 376.46 | 380.67 | 0.00 | T1 1.5R @ 376.46 |
| Target hit | 2026-01-08 15:20:00 | 367.65 | 373.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2026-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:55:00 | 367.10 | 368.79 | 0.00 | ORB-short ORB[369.35,374.85] vol=9.7x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-01-14 11:40:00 | 368.60 | 368.62 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-01-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:20:00 | 369.20 | 364.57 | 0.00 | ORB-long ORB[355.95,358.15] vol=2.3x ATR=1.81 |
| Stop hit — per-position SL triggered | 2026-01-30 13:05:00 | 367.39 | 366.68 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 429.30 | 427.50 | 0.00 | ORB-long ORB[423.35,429.00] vol=1.6x ATR=1.19 |
| Stop hit — per-position SL triggered | 2026-02-17 13:55:00 | 428.11 | 428.18 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 424.45 | 426.54 | 0.00 | ORB-short ORB[426.35,429.40] vol=2.3x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:05:00 | 422.90 | 425.57 | 0.00 | T1 1.5R @ 422.90 |
| Target hit | 2026-02-18 15:05:00 | 423.00 | 422.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 417.20 | 418.94 | 0.00 | ORB-short ORB[418.20,422.05] vol=1.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 418.30 | 418.76 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 415.35 | 413.48 | 0.00 | ORB-long ORB[410.15,415.10] vol=1.9x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 13:30:00 | 417.54 | 414.95 | 0.00 | T1 1.5R @ 417.54 |
| Stop hit — per-position SL triggered | 2026-02-20 15:05:00 | 415.35 | 415.24 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 405.00 | 406.86 | 0.00 | ORB-short ORB[405.65,411.65] vol=1.6x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 406.43 | 406.57 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:55:00 | 390.10 | 386.27 | 0.00 | ORB-long ORB[383.15,388.00] vol=1.9x ATR=2.29 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 387.81 | 387.53 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-03-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:35:00 | 390.50 | 385.90 | 0.00 | ORB-long ORB[379.95,385.25] vol=1.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-03-25 10:50:00 | 388.80 | 386.34 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 418.75 | 415.50 | 0.00 | ORB-long ORB[410.50,416.15] vol=8.4x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-04-15 10:00:00 | 416.75 | 416.29 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:05:00 | 417.50 | 414.92 | 0.00 | ORB-long ORB[414.00,417.45] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 416.32 | 415.06 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 420.45 | 419.04 | 0.00 | ORB-long ORB[415.65,419.75] vol=1.6x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 422.62 | 420.87 | 0.00 | T1 1.5R @ 422.62 |
| Target hit | 2026-04-21 11:15:00 | 421.80 | 422.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — SELL (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 413.80 | 416.33 | 0.00 | ORB-short ORB[415.30,421.00] vol=1.8x ATR=1.59 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 415.39 | 415.21 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 425.40 | 422.70 | 0.00 | ORB-long ORB[418.40,423.55] vol=3.0x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:35:00 | 427.93 | 423.63 | 0.00 | T1 1.5R @ 427.93 |
| Target hit | 2026-04-27 15:20:00 | 429.05 | 428.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 430.50 | 429.02 | 0.00 | ORB-long ORB[424.00,429.35] vol=2.9x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:45:00 | 432.94 | 430.19 | 0.00 | T1 1.5R @ 432.94 |
| Target hit | 2026-04-29 10:30:00 | 433.05 | 434.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 426.00 | 427.80 | 0.00 | ORB-short ORB[428.00,431.00] vol=2.1x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-05-05 12:50:00 | 427.48 | 426.69 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 421.85 | 424.97 | 0.00 | ORB-short ORB[425.25,429.80] vol=1.9x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 423.02 | 424.76 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 424.95 | 426.95 | 0.00 | ORB-short ORB[425.90,432.00] vol=1.8x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-05-07 10:00:00 | 426.36 | 426.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-14 09:35:00 | 370.10 | 2025-05-14 10:00:00 | 371.65 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-05-29 10:40:00 | 418.55 | 2025-05-29 10:45:00 | 419.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-30 09:30:00 | 428.40 | 2025-05-30 09:35:00 | 431.17 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-05-30 09:30:00 | 428.40 | 2025-05-30 10:05:00 | 428.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-19 10:15:00 | 432.35 | 2025-06-19 11:25:00 | 429.67 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-06-19 10:15:00 | 432.35 | 2025-06-19 15:20:00 | 425.35 | TARGET_HIT | 0.50 | 1.62% |
| BUY | retest1 | 2025-06-27 09:30:00 | 449.20 | 2025-06-27 09:35:00 | 447.23 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-08-01 09:50:00 | 479.80 | 2025-08-01 10:20:00 | 481.83 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-08-05 10:55:00 | 473.65 | 2025-08-05 11:40:00 | 475.30 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-06 09:35:00 | 467.80 | 2025-08-06 10:35:00 | 464.16 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2025-08-06 09:35:00 | 467.80 | 2025-08-06 13:05:00 | 467.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 09:40:00 | 455.90 | 2025-08-14 09:55:00 | 453.16 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-08-14 09:40:00 | 455.90 | 2025-08-14 10:25:00 | 455.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-18 09:45:00 | 444.90 | 2025-08-18 09:55:00 | 446.97 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-08-21 09:35:00 | 457.80 | 2025-08-21 09:40:00 | 456.15 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-08-22 09:35:00 | 443.20 | 2025-08-22 09:40:00 | 440.57 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-08-22 09:35:00 | 443.20 | 2025-08-22 09:45:00 | 443.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 11:05:00 | 463.70 | 2025-09-17 11:10:00 | 465.41 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-17 11:05:00 | 463.70 | 2025-09-17 11:55:00 | 463.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-18 11:05:00 | 461.60 | 2025-09-18 11:15:00 | 462.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-09 11:05:00 | 390.90 | 2025-10-09 11:55:00 | 389.38 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-09 11:05:00 | 390.90 | 2025-10-09 15:20:00 | 386.75 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2025-10-10 09:30:00 | 392.00 | 2025-10-10 09:35:00 | 390.55 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-14 09:40:00 | 403.60 | 2025-10-14 09:45:00 | 406.02 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-10-14 09:40:00 | 403.60 | 2025-10-14 09:50:00 | 403.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 09:40:00 | 418.80 | 2025-10-17 09:45:00 | 420.92 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-17 09:40:00 | 418.80 | 2025-10-17 09:50:00 | 418.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 11:05:00 | 404.00 | 2025-11-06 12:10:00 | 405.83 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-11-07 09:30:00 | 400.30 | 2025-11-07 09:50:00 | 401.89 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-11-10 09:50:00 | 402.55 | 2025-11-10 10:15:00 | 400.52 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-11-10 09:50:00 | 402.55 | 2025-11-10 13:10:00 | 402.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-13 09:35:00 | 409.25 | 2025-11-13 12:35:00 | 410.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-11-18 09:55:00 | 407.00 | 2025-11-18 10:05:00 | 405.43 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-18 09:55:00 | 407.00 | 2025-11-18 11:40:00 | 406.25 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-11-19 10:15:00 | 419.40 | 2025-11-19 10:30:00 | 417.99 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-20 10:10:00 | 417.15 | 2025-11-20 11:10:00 | 415.89 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-24 10:55:00 | 404.80 | 2025-11-24 12:20:00 | 405.63 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-03 10:00:00 | 408.75 | 2025-12-03 10:15:00 | 409.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-04 10:20:00 | 406.60 | 2025-12-04 11:15:00 | 405.38 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-08 10:55:00 | 391.15 | 2025-12-08 11:10:00 | 389.15 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-08 10:55:00 | 391.15 | 2025-12-08 11:35:00 | 391.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-12 11:15:00 | 379.60 | 2025-12-12 11:20:00 | 380.45 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-16 10:55:00 | 387.45 | 2025-12-16 11:25:00 | 389.29 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-12-16 10:55:00 | 387.45 | 2025-12-16 11:30:00 | 387.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-18 09:50:00 | 391.00 | 2025-12-18 10:05:00 | 392.21 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-19 09:30:00 | 397.85 | 2025-12-19 09:50:00 | 396.46 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-24 09:55:00 | 402.00 | 2025-12-24 11:10:00 | 400.36 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-24 09:55:00 | 402.00 | 2025-12-24 12:45:00 | 402.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 11:15:00 | 398.05 | 2025-12-29 11:45:00 | 398.82 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-30 10:45:00 | 383.70 | 2025-12-30 11:10:00 | 381.85 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-12-30 10:45:00 | 383.70 | 2025-12-30 11:25:00 | 383.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-01 11:05:00 | 387.45 | 2026-01-01 12:05:00 | 385.96 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-01 11:05:00 | 387.45 | 2026-01-01 15:05:00 | 386.65 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-01-08 10:40:00 | 378.15 | 2026-01-08 11:10:00 | 376.46 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-08 10:40:00 | 378.15 | 2026-01-08 15:20:00 | 367.65 | TARGET_HIT | 0.50 | 2.78% |
| SELL | retest1 | 2026-01-14 10:55:00 | 367.10 | 2026-01-14 11:40:00 | 368.60 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-01-30 10:20:00 | 369.20 | 2026-01-30 13:05:00 | 367.39 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-02-17 11:15:00 | 429.30 | 2026-02-17 13:55:00 | 428.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-18 10:55:00 | 424.45 | 2026-02-18 11:05:00 | 422.90 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-18 10:55:00 | 424.45 | 2026-02-18 15:05:00 | 423.00 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 09:30:00 | 417.20 | 2026-02-19 09:35:00 | 418.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-20 10:35:00 | 415.35 | 2026-02-20 13:30:00 | 417.54 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-20 10:35:00 | 415.35 | 2026-02-20 15:05:00 | 415.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 09:30:00 | 405.00 | 2026-02-25 09:45:00 | 406.43 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-17 09:55:00 | 390.10 | 2026-03-17 10:30:00 | 387.81 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2026-03-25 10:35:00 | 390.50 | 2026-03-25 10:50:00 | 388.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-15 09:40:00 | 418.75 | 2026-04-15 10:00:00 | 416.75 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-16 11:05:00 | 417.50 | 2026-04-16 11:25:00 | 416.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 09:35:00 | 420.45 | 2026-04-21 09:45:00 | 422.62 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-21 09:35:00 | 420.45 | 2026-04-21 11:15:00 | 421.80 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2026-04-24 09:40:00 | 413.80 | 2026-04-24 10:00:00 | 415.39 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-27 09:30:00 | 425.40 | 2026-04-27 09:35:00 | 427.93 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-27 09:30:00 | 425.40 | 2026-04-27 15:20:00 | 429.05 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2026-04-29 09:35:00 | 430.50 | 2026-04-29 09:45:00 | 432.94 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-29 09:35:00 | 430.50 | 2026-04-29 10:30:00 | 433.05 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2026-05-05 10:10:00 | 426.00 | 2026-05-05 12:50:00 | 427.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-05-06 11:00:00 | 421.85 | 2026-05-06 11:15:00 | 423.02 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-07 09:45:00 | 424.95 | 2026-05-07 10:00:00 | 426.36 | STOP_HIT | 1.00 | -0.33% |
