# Sumitomo Chemical India Ltd. (SUMICHEM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-11-06 15:25:00 (9183 bars)
- **Last close:** 563.70
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
| ENTRY1 | 45 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 41
- **Target hits / Stop hits / Partials:** 4 / 41 / 11
- **Avg / median % per leg:** -0.08% / -0.32%
- **Sum % (uncompounded):** -4.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 8 | 27.6% | 1 | 21 | 7 | -0.05% | -1.3% |
| BUY @ 2nd Alert (retest1) | 29 | 8 | 27.6% | 1 | 21 | 7 | -0.05% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 7 | 25.9% | 3 | 20 | 4 | -0.11% | -3.0% |
| SELL @ 2nd Alert (retest1) | 27 | 7 | 25.9% | 3 | 20 | 4 | -0.11% | -3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 56 | 15 | 26.8% | 4 | 41 | 11 | -0.08% | -4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:30:00 | 380.95 | 383.77 | 0.00 | ORB-short ORB[382.80,386.80] vol=1.6x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-05-13 11:05:00 | 382.87 | 383.40 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:15:00 | 400.75 | 394.45 | 0.00 | ORB-long ORB[389.80,394.40] vol=11.9x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-05-14 11:25:00 | 399.02 | 397.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:45:00 | 397.95 | 399.33 | 0.00 | ORB-short ORB[398.95,401.05] vol=1.9x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-05-15 11:10:00 | 399.23 | 399.04 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 392.90 | 394.25 | 0.00 | ORB-short ORB[394.00,396.50] vol=2.5x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-05-16 11:25:00 | 393.89 | 394.20 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:55:00 | 397.40 | 394.78 | 0.00 | ORB-long ORB[393.30,395.10] vol=3.9x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:10:00 | 399.26 | 396.19 | 0.00 | T1 1.5R @ 399.26 |
| Stop hit — per-position SL triggered | 2024-05-17 10:15:00 | 397.40 | 396.19 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:00:00 | 408.10 | 404.20 | 0.00 | ORB-long ORB[400.50,405.00] vol=3.1x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-05-21 10:20:00 | 406.30 | 405.02 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:35:00 | 469.45 | 472.24 | 0.00 | ORB-short ORB[471.90,476.95] vol=3.7x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-05-31 09:50:00 | 471.92 | 471.32 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 485.40 | 481.84 | 0.00 | ORB-long ORB[477.00,483.70] vol=1.8x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-06-07 10:15:00 | 482.99 | 483.83 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:50:00 | 486.40 | 488.74 | 0.00 | ORB-short ORB[487.00,493.45] vol=1.6x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-06-12 09:55:00 | 488.26 | 488.75 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 480.40 | 484.77 | 0.00 | ORB-short ORB[484.60,491.25] vol=1.6x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-06-13 09:55:00 | 482.89 | 484.04 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:05:00 | 497.20 | 493.00 | 0.00 | ORB-long ORB[489.95,496.45] vol=2.4x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:15:00 | 500.84 | 494.96 | 0.00 | T1 1.5R @ 500.84 |
| Stop hit — per-position SL triggered | 2024-06-14 11:55:00 | 497.20 | 498.20 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:45:00 | 479.70 | 478.51 | 0.00 | ORB-long ORB[470.30,476.00] vol=1.7x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 11:40:00 | 482.99 | 479.38 | 0.00 | T1 1.5R @ 482.99 |
| Stop hit — per-position SL triggered | 2024-06-20 11:50:00 | 479.70 | 479.60 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:50:00 | 496.35 | 488.74 | 0.00 | ORB-long ORB[484.85,492.00] vol=2.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-06-24 11:10:00 | 494.34 | 489.74 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:10:00 | 486.75 | 488.74 | 0.00 | ORB-short ORB[487.70,492.45] vol=1.7x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:30:00 | 485.03 | 488.39 | 0.00 | T1 1.5R @ 485.03 |
| Target hit | 2024-06-25 15:20:00 | 481.60 | 485.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-06-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:25:00 | 485.25 | 482.23 | 0.00 | ORB-long ORB[479.20,483.40] vol=1.7x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-06-26 14:25:00 | 483.45 | 484.46 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-06-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 10:40:00 | 479.05 | 480.49 | 0.00 | ORB-short ORB[479.20,483.10] vol=1.8x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-06-28 10:50:00 | 480.69 | 480.48 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 479.10 | 481.05 | 0.00 | ORB-short ORB[480.05,485.15] vol=1.7x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-07-01 09:35:00 | 480.81 | 480.79 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 11:00:00 | 499.25 | 496.76 | 0.00 | ORB-long ORB[492.10,497.50] vol=3.2x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-07-04 12:00:00 | 497.52 | 497.86 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 10:45:00 | 490.00 | 493.14 | 0.00 | ORB-short ORB[491.15,496.70] vol=1.7x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-07-05 11:20:00 | 491.65 | 492.52 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:20:00 | 497.40 | 492.62 | 0.00 | ORB-long ORB[488.15,492.20] vol=1.9x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-07-08 10:40:00 | 495.56 | 494.29 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 486.20 | 492.16 | 0.00 | ORB-short ORB[492.20,497.30] vol=3.0x ATR=2.32 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 488.52 | 492.01 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 504.40 | 500.70 | 0.00 | ORB-long ORB[493.00,498.00] vol=2.8x ATR=2.77 |
| Stop hit — per-position SL triggered | 2024-07-12 09:35:00 | 501.63 | 501.14 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:55:00 | 508.80 | 506.04 | 0.00 | ORB-long ORB[502.55,506.90] vol=2.3x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-07-16 10:00:00 | 507.04 | 506.15 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:50:00 | 508.30 | 504.24 | 0.00 | ORB-long ORB[499.05,505.00] vol=1.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 12:40:00 | 511.00 | 506.62 | 0.00 | T1 1.5R @ 511.00 |
| Stop hit — per-position SL triggered | 2024-07-26 13:25:00 | 508.30 | 509.17 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-07-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:30:00 | 543.10 | 537.39 | 0.00 | ORB-long ORB[517.80,525.95] vol=2.6x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-07-31 10:40:00 | 539.71 | 538.16 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:50:00 | 519.80 | 521.46 | 0.00 | ORB-short ORB[519.85,525.60] vol=2.2x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-08-01 11:20:00 | 521.27 | 521.34 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:50:00 | 520.45 | 517.14 | 0.00 | ORB-long ORB[512.10,519.65] vol=3.1x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:55:00 | 524.03 | 518.10 | 0.00 | T1 1.5R @ 524.03 |
| Stop hit — per-position SL triggered | 2024-08-09 10:25:00 | 520.45 | 519.73 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:05:00 | 529.85 | 529.98 | 0.00 | ORB-short ORB[532.55,540.20] vol=4.3x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-08-16 10:10:00 | 532.42 | 530.10 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 09:30:00 | 534.75 | 537.03 | 0.00 | ORB-short ORB[535.25,541.90] vol=1.6x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-08-19 10:00:00 | 537.18 | 536.44 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:50:00 | 539.75 | 535.86 | 0.00 | ORB-long ORB[530.70,536.55] vol=3.0x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-08-20 09:55:00 | 537.63 | 537.11 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 09:35:00 | 536.55 | 538.56 | 0.00 | ORB-short ORB[537.20,540.00] vol=1.7x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-08-21 09:45:00 | 538.20 | 538.07 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 544.60 | 549.77 | 0.00 | ORB-short ORB[547.00,554.15] vol=2.4x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-08-22 09:35:00 | 547.45 | 548.07 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-08-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 11:00:00 | 526.05 | 531.00 | 0.00 | ORB-short ORB[530.00,534.75] vol=1.7x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-08-26 11:10:00 | 527.60 | 530.87 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:15:00 | 538.90 | 533.86 | 0.00 | ORB-long ORB[530.50,536.40] vol=2.0x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:30:00 | 541.90 | 536.23 | 0.00 | T1 1.5R @ 541.90 |
| Stop hit — per-position SL triggered | 2024-08-28 11:10:00 | 538.90 | 539.76 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:40:00 | 513.30 | 515.20 | 0.00 | ORB-short ORB[513.40,518.45] vol=6.6x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-09-03 11:25:00 | 514.82 | 515.09 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 590.70 | 595.18 | 0.00 | ORB-short ORB[592.00,600.00] vol=2.0x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:05:00 | 585.78 | 593.29 | 0.00 | T1 1.5R @ 585.78 |
| Target hit | 2024-09-16 13:25:00 | 590.05 | 588.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — SELL (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 584.95 | 588.59 | 0.00 | ORB-short ORB[586.30,594.00] vol=1.6x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-09-17 11:45:00 | 587.56 | 585.60 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:55:00 | 593.00 | 600.00 | 0.00 | ORB-short ORB[599.20,606.00] vol=1.7x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 589.58 | 597.87 | 0.00 | T1 1.5R @ 589.58 |
| Target hit | 2024-09-19 15:00:00 | 586.75 | 585.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — SELL (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 550.55 | 552.41 | 0.00 | ORB-short ORB[551.25,557.05] vol=1.5x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:55:00 | 547.62 | 550.48 | 0.00 | T1 1.5R @ 547.62 |
| Stop hit — per-position SL triggered | 2024-09-26 12:30:00 | 550.55 | 549.13 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:05:00 | 554.00 | 550.81 | 0.00 | ORB-long ORB[547.05,552.55] vol=7.9x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-09-27 11:25:00 | 551.93 | 551.27 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:35:00 | 566.10 | 563.17 | 0.00 | ORB-long ORB[557.00,564.00] vol=3.7x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:55:00 | 569.60 | 565.78 | 0.00 | T1 1.5R @ 569.60 |
| Target hit | 2024-10-01 10:45:00 | 569.85 | 569.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2024-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:35:00 | 562.50 | 560.53 | 0.00 | ORB-long ORB[556.70,561.45] vol=1.6x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-10-03 09:55:00 | 559.98 | 560.86 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:40:00 | 545.30 | 543.87 | 0.00 | ORB-long ORB[540.15,544.85] vol=3.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2024-10-15 09:50:00 | 543.41 | 543.92 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:15:00 | 559.40 | 557.11 | 0.00 | ORB-long ORB[550.60,557.40] vol=2.2x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-10-16 10:20:00 | 557.48 | 557.13 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:40:00 | 561.80 | 568.45 | 0.00 | ORB-short ORB[566.90,573.05] vol=1.5x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-11-06 10:50:00 | 564.17 | 567.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:30:00 | 380.95 | 2024-05-13 11:05:00 | 382.87 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-05-14 11:15:00 | 400.75 | 2024-05-14 11:25:00 | 399.02 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-05-15 10:45:00 | 397.95 | 2024-05-15 11:10:00 | 399.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-16 11:15:00 | 392.90 | 2024-05-16 11:25:00 | 393.89 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-17 09:55:00 | 397.40 | 2024-05-17 10:10:00 | 399.26 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-05-17 09:55:00 | 397.40 | 2024-05-17 10:15:00 | 397.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-21 10:00:00 | 408.10 | 2024-05-21 10:20:00 | 406.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-05-31 09:35:00 | 469.45 | 2024-05-31 09:50:00 | 471.92 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-06-07 09:30:00 | 485.40 | 2024-06-07 10:15:00 | 482.99 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-06-12 09:50:00 | 486.40 | 2024-06-12 09:55:00 | 488.26 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-13 09:35:00 | 480.40 | 2024-06-13 09:55:00 | 482.89 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-06-14 10:05:00 | 497.20 | 2024-06-14 10:15:00 | 500.84 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-06-14 10:05:00 | 497.20 | 2024-06-14 11:55:00 | 497.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 10:45:00 | 479.70 | 2024-06-20 11:40:00 | 482.99 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-06-20 10:45:00 | 479.70 | 2024-06-20 11:50:00 | 479.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-24 10:50:00 | 496.35 | 2024-06-24 11:10:00 | 494.34 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-06-25 11:10:00 | 486.75 | 2024-06-25 11:30:00 | 485.03 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-06-25 11:10:00 | 486.75 | 2024-06-25 15:20:00 | 481.60 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2024-06-26 10:25:00 | 485.25 | 2024-06-26 14:25:00 | 483.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-06-28 10:40:00 | 479.05 | 2024-06-28 10:50:00 | 480.69 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-01 09:30:00 | 479.10 | 2024-07-01 09:35:00 | 480.81 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-04 11:00:00 | 499.25 | 2024-07-04 12:00:00 | 497.52 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-05 10:45:00 | 490.00 | 2024-07-05 11:20:00 | 491.65 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-08 10:20:00 | 497.40 | 2024-07-08 10:40:00 | 495.56 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-10 10:10:00 | 486.20 | 2024-07-10 10:15:00 | 488.52 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-07-12 09:30:00 | 504.40 | 2024-07-12 09:35:00 | 501.63 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-07-16 09:55:00 | 508.80 | 2024-07-16 10:00:00 | 507.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-26 09:50:00 | 508.30 | 2024-07-26 12:40:00 | 511.00 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-26 09:50:00 | 508.30 | 2024-07-26 13:25:00 | 508.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 10:30:00 | 543.10 | 2024-07-31 10:40:00 | 539.71 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2024-08-01 10:50:00 | 519.80 | 2024-08-01 11:20:00 | 521.27 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-09 09:50:00 | 520.45 | 2024-08-09 09:55:00 | 524.03 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-08-09 09:50:00 | 520.45 | 2024-08-09 10:25:00 | 520.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-16 10:05:00 | 529.85 | 2024-08-16 10:10:00 | 532.42 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-08-19 09:30:00 | 534.75 | 2024-08-19 10:00:00 | 537.18 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-08-20 09:50:00 | 539.75 | 2024-08-20 09:55:00 | 537.63 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-21 09:35:00 | 536.55 | 2024-08-21 09:45:00 | 538.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-22 09:30:00 | 544.60 | 2024-08-22 09:35:00 | 547.45 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-08-26 11:00:00 | 526.05 | 2024-08-26 11:10:00 | 527.60 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-28 10:15:00 | 538.90 | 2024-08-28 10:30:00 | 541.90 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-28 10:15:00 | 538.90 | 2024-08-28 11:10:00 | 538.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-03 10:40:00 | 513.30 | 2024-09-03 11:25:00 | 514.82 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-16 09:30:00 | 590.70 | 2024-09-16 10:05:00 | 585.78 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2024-09-16 09:30:00 | 590.70 | 2024-09-16 13:25:00 | 590.05 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2024-09-17 09:30:00 | 584.95 | 2024-09-17 11:45:00 | 587.56 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-09-19 09:55:00 | 593.00 | 2024-09-19 10:15:00 | 589.58 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-19 09:55:00 | 593.00 | 2024-09-19 15:00:00 | 586.75 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2024-09-26 09:30:00 | 550.55 | 2024-09-26 09:55:00 | 547.62 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-09-26 09:30:00 | 550.55 | 2024-09-26 12:30:00 | 550.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 11:05:00 | 554.00 | 2024-09-27 11:25:00 | 551.93 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-01 09:35:00 | 566.10 | 2024-10-01 09:55:00 | 569.60 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-01 09:35:00 | 566.10 | 2024-10-01 10:45:00 | 569.85 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-10-03 09:35:00 | 562.50 | 2024-10-03 09:55:00 | 559.98 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-10-15 09:40:00 | 545.30 | 2024-10-15 09:50:00 | 543.41 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-16 10:15:00 | 559.40 | 2024-10-16 10:20:00 | 557.48 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-11-06 10:40:00 | 561.80 | 2024-11-06 10:50:00 | 564.17 | STOP_HIT | 1.00 | -0.42% |
