# Indian Bank (INDIANB)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 862.50
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
| PARTIAL | 31 |
| TARGET_HIT | 15 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 101 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 55
- **Target hits / Stop hits / Partials:** 15 / 55 / 31
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 22.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 22 | 40.7% | 8 | 32 | 14 | 0.18% | 9.8% |
| BUY @ 2nd Alert (retest1) | 54 | 22 | 40.7% | 8 | 32 | 14 | 0.18% | 9.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 24 | 51.1% | 7 | 23 | 17 | 0.28% | 13.1% |
| SELL @ 2nd Alert (retest1) | 47 | 24 | 51.1% | 7 | 23 | 17 | 0.28% | 13.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 101 | 46 | 45.5% | 15 | 55 | 31 | 0.23% | 22.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:40:00 | 530.55 | 531.49 | 0.00 | ORB-short ORB[530.65,535.85] vol=1.5x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-05-16 12:05:00 | 532.59 | 530.89 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:35:00 | 539.80 | 536.01 | 0.00 | ORB-long ORB[534.00,539.25] vol=1.7x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 11:00:00 | 542.58 | 537.44 | 0.00 | T1 1.5R @ 542.58 |
| Target hit | 2024-05-17 12:50:00 | 541.45 | 541.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2024-05-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:00:00 | 584.90 | 577.28 | 0.00 | ORB-long ORB[569.65,577.00] vol=4.4x ATR=2.91 |
| Stop hit — per-position SL triggered | 2024-05-23 10:05:00 | 581.99 | 577.86 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:05:00 | 577.30 | 571.66 | 0.00 | ORB-long ORB[568.05,576.00] vol=2.7x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-05-24 11:25:00 | 575.16 | 572.63 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:05:00 | 537.55 | 533.33 | 0.00 | ORB-long ORB[531.35,537.00] vol=1.9x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 11:10:00 | 541.04 | 534.89 | 0.00 | T1 1.5R @ 541.04 |
| Stop hit — per-position SL triggered | 2024-06-07 11:25:00 | 537.55 | 536.51 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:10:00 | 537.70 | 541.35 | 0.00 | ORB-short ORB[540.50,544.20] vol=1.8x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-06-18 10:30:00 | 539.21 | 540.88 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 10:00:00 | 543.65 | 540.05 | 0.00 | ORB-long ORB[537.10,542.80] vol=4.3x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-06-19 10:05:00 | 541.48 | 540.31 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:10:00 | 545.05 | 541.81 | 0.00 | ORB-long ORB[537.00,542.90] vol=2.1x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-06-24 14:25:00 | 543.77 | 542.96 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:05:00 | 537.85 | 537.29 | 0.00 | ORB-long ORB[531.50,536.00] vol=9.5x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:20:00 | 540.66 | 537.40 | 0.00 | T1 1.5R @ 540.66 |
| Target hit | 2024-06-26 15:20:00 | 540.25 | 538.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-06-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:50:00 | 553.40 | 549.01 | 0.00 | ORB-long ORB[540.55,544.50] vol=4.3x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-06-28 10:05:00 | 550.76 | 550.49 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:35:00 | 543.60 | 545.19 | 0.00 | ORB-short ORB[544.55,547.95] vol=1.8x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 09:45:00 | 541.07 | 544.43 | 0.00 | T1 1.5R @ 541.07 |
| Target hit | 2024-07-02 14:25:00 | 537.75 | 537.63 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2024-07-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:50:00 | 537.15 | 531.81 | 0.00 | ORB-long ORB[525.55,531.50] vol=1.7x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-07-09 10:05:00 | 535.08 | 534.63 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:45:00 | 556.55 | 552.13 | 0.00 | ORB-long ORB[543.90,551.70] vol=1.9x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-07-12 09:55:00 | 554.45 | 553.73 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:55:00 | 562.50 | 556.36 | 0.00 | ORB-long ORB[552.40,557.95] vol=1.8x ATR=3.02 |
| Stop hit — per-position SL triggered | 2024-07-24 10:05:00 | 559.48 | 558.23 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:10:00 | 610.00 | 606.63 | 0.00 | ORB-long ORB[602.20,608.80] vol=1.8x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:30:00 | 613.00 | 608.61 | 0.00 | T1 1.5R @ 613.00 |
| Stop hit — per-position SL triggered | 2024-08-01 10:35:00 | 610.00 | 608.72 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:55:00 | 562.10 | 563.78 | 0.00 | ORB-short ORB[564.00,570.00] vol=3.0x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-08-08 11:10:00 | 563.73 | 563.52 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:55:00 | 555.70 | 559.75 | 0.00 | ORB-short ORB[558.00,563.75] vol=2.5x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 11:00:00 | 553.24 | 559.07 | 0.00 | T1 1.5R @ 553.24 |
| Stop hit — per-position SL triggered | 2024-08-09 11:25:00 | 555.70 | 558.51 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 546.30 | 549.34 | 0.00 | ORB-short ORB[548.05,554.00] vol=1.7x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:40:00 | 543.32 | 548.24 | 0.00 | T1 1.5R @ 543.32 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 546.30 | 548.12 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:55:00 | 558.35 | 554.22 | 0.00 | ORB-long ORB[550.05,554.80] vol=1.8x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-08-20 11:30:00 | 556.64 | 555.08 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:35:00 | 553.80 | 555.21 | 0.00 | ORB-short ORB[554.30,560.95] vol=4.7x ATR=1.33 |
| Stop hit — per-position SL triggered | 2024-08-21 11:15:00 | 555.13 | 555.17 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:25:00 | 549.45 | 550.34 | 0.00 | ORB-short ORB[549.50,554.90] vol=3.8x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 12:00:00 | 547.18 | 549.36 | 0.00 | T1 1.5R @ 547.18 |
| Stop hit — per-position SL triggered | 2024-08-26 12:10:00 | 549.45 | 550.16 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 11:05:00 | 565.75 | 562.76 | 0.00 | ORB-long ORB[561.90,565.30] vol=2.4x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-09-02 11:40:00 | 564.02 | 563.57 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:55:00 | 531.85 | 533.87 | 0.00 | ORB-short ORB[534.15,541.90] vol=2.2x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:25:00 | 529.51 | 532.43 | 0.00 | T1 1.5R @ 529.51 |
| Stop hit — per-position SL triggered | 2024-09-05 15:00:00 | 531.85 | 529.19 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:15:00 | 519.85 | 523.81 | 0.00 | ORB-short ORB[522.15,526.40] vol=1.5x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:40:00 | 517.35 | 520.81 | 0.00 | T1 1.5R @ 517.35 |
| Stop hit — per-position SL triggered | 2024-09-10 12:30:00 | 519.85 | 520.11 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:30:00 | 516.15 | 519.56 | 0.00 | ORB-short ORB[518.50,522.65] vol=1.7x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:30:00 | 514.13 | 517.80 | 0.00 | T1 1.5R @ 514.13 |
| Target hit | 2024-09-18 15:20:00 | 505.00 | 509.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 527.90 | 521.70 | 0.00 | ORB-long ORB[510.00,514.95] vol=2.3x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:05:00 | 530.50 | 523.69 | 0.00 | T1 1.5R @ 530.50 |
| Target hit | 2024-09-23 15:20:00 | 534.25 | 530.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2024-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:55:00 | 527.40 | 530.85 | 0.00 | ORB-short ORB[529.15,536.85] vol=1.7x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 529.00 | 530.66 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:35:00 | 522.80 | 527.28 | 0.00 | ORB-short ORB[525.65,531.30] vol=2.2x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 11:50:00 | 520.66 | 525.01 | 0.00 | T1 1.5R @ 520.66 |
| Stop hit — per-position SL triggered | 2024-09-25 13:25:00 | 522.80 | 523.63 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:15:00 | 538.00 | 533.34 | 0.00 | ORB-long ORB[529.25,536.90] vol=3.5x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-09-27 10:20:00 | 535.50 | 533.79 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 10:55:00 | 524.80 | 522.17 | 0.00 | ORB-long ORB[515.85,521.50] vol=1.9x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:05:00 | 527.65 | 522.81 | 0.00 | T1 1.5R @ 527.65 |
| Stop hit — per-position SL triggered | 2024-10-03 11:20:00 | 524.80 | 523.06 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:05:00 | 521.50 | 515.73 | 0.00 | ORB-long ORB[511.70,519.00] vol=1.7x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 10:30:00 | 525.25 | 518.63 | 0.00 | T1 1.5R @ 525.25 |
| Target hit | 2024-10-08 15:20:00 | 528.80 | 524.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2024-10-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:20:00 | 530.05 | 525.98 | 0.00 | ORB-long ORB[519.10,525.80] vol=1.8x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-10-16 10:30:00 | 528.05 | 526.17 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:05:00 | 512.55 | 517.03 | 0.00 | ORB-short ORB[519.30,522.90] vol=1.5x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-10-17 10:10:00 | 514.15 | 516.87 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:50:00 | 502.70 | 509.01 | 0.00 | ORB-short ORB[510.95,516.00] vol=1.9x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-10-22 12:30:00 | 504.83 | 505.46 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:25:00 | 506.50 | 504.41 | 0.00 | ORB-long ORB[501.00,506.45] vol=3.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-10-24 10:55:00 | 504.76 | 504.76 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:40:00 | 496.65 | 499.92 | 0.00 | ORB-short ORB[499.90,504.80] vol=1.8x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:50:00 | 493.61 | 497.50 | 0.00 | T1 1.5R @ 493.61 |
| Target hit | 2024-10-25 10:20:00 | 494.80 | 494.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — SELL (started 2024-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 09:45:00 | 576.20 | 578.25 | 0.00 | ORB-short ORB[576.50,582.40] vol=2.8x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:20:00 | 572.21 | 576.13 | 0.00 | T1 1.5R @ 572.21 |
| Stop hit — per-position SL triggered | 2024-11-06 12:35:00 | 576.20 | 574.77 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 551.40 | 555.47 | 0.00 | ORB-short ORB[554.05,559.70] vol=2.1x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 547.74 | 553.67 | 0.00 | T1 1.5R @ 547.74 |
| Stop hit — per-position SL triggered | 2024-11-13 10:30:00 | 551.40 | 550.58 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:50:00 | 565.25 | 560.57 | 0.00 | ORB-long ORB[557.15,562.05] vol=1.5x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-11-27 09:55:00 | 563.10 | 560.76 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:40:00 | 599.30 | 592.12 | 0.00 | ORB-long ORB[578.95,584.55] vol=8.9x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-12-04 09:50:00 | 596.60 | 594.73 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:50:00 | 602.30 | 598.16 | 0.00 | ORB-long ORB[593.00,599.60] vol=1.8x ATR=2.79 |
| Stop hit — per-position SL triggered | 2024-12-10 10:20:00 | 599.51 | 599.89 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 09:45:00 | 595.20 | 599.69 | 0.00 | ORB-short ORB[596.10,602.55] vol=2.3x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:05:00 | 592.30 | 598.83 | 0.00 | T1 1.5R @ 592.30 |
| Target hit | 2024-12-11 15:20:00 | 583.30 | 589.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-01-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:55:00 | 526.10 | 527.04 | 0.00 | ORB-short ORB[528.10,531.30] vol=3.3x ATR=2.15 |
| Stop hit — per-position SL triggered | 2025-01-01 11:20:00 | 528.25 | 526.89 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 528.60 | 527.10 | 0.00 | ORB-long ORB[523.35,528.15] vol=1.5x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 09:35:00 | 531.43 | 528.45 | 0.00 | T1 1.5R @ 531.43 |
| Stop hit — per-position SL triggered | 2025-01-03 10:10:00 | 528.60 | 529.13 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:50:00 | 502.75 | 505.34 | 0.00 | ORB-short ORB[504.50,509.45] vol=2.1x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:55:00 | 500.87 | 504.55 | 0.00 | T1 1.5R @ 500.87 |
| Target hit | 2025-01-09 12:15:00 | 501.95 | 501.94 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — BUY (started 2025-01-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 10:35:00 | 490.00 | 485.26 | 0.00 | ORB-long ORB[480.05,486.80] vol=1.8x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 10:45:00 | 492.95 | 486.79 | 0.00 | T1 1.5R @ 492.95 |
| Target hit | 2025-01-14 15:20:00 | 509.10 | 498.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2025-01-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 11:05:00 | 510.15 | 508.29 | 0.00 | ORB-long ORB[503.45,508.10] vol=2.0x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-01-15 11:30:00 | 508.52 | 508.47 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:40:00 | 516.55 | 513.78 | 0.00 | ORB-long ORB[509.40,514.40] vol=1.6x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 09:55:00 | 519.33 | 515.00 | 0.00 | T1 1.5R @ 519.33 |
| Target hit | 2025-01-16 15:20:00 | 527.55 | 525.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-01-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 11:00:00 | 520.20 | 524.95 | 0.00 | ORB-short ORB[521.05,527.90] vol=1.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-01-17 11:05:00 | 521.81 | 524.78 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 528.05 | 530.64 | 0.00 | ORB-short ORB[529.00,533.80] vol=1.5x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 11:20:00 | 525.53 | 528.68 | 0.00 | T1 1.5R @ 525.53 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 528.05 | 528.10 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:50:00 | 487.35 | 492.89 | 0.00 | ORB-short ORB[493.30,500.25] vol=5.1x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-01-27 10:55:00 | 489.35 | 492.30 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-02-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:40:00 | 552.00 | 545.73 | 0.00 | ORB-long ORB[542.35,549.90] vol=2.8x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-02-07 10:45:00 | 549.00 | 545.88 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-02-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:20:00 | 525.70 | 524.74 | 0.00 | ORB-long ORB[518.50,524.90] vol=3.7x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:30:00 | 528.70 | 525.77 | 0.00 | T1 1.5R @ 528.70 |
| Stop hit — per-position SL triggered | 2025-02-20 12:40:00 | 525.70 | 526.64 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 10:50:00 | 521.85 | 524.27 | 0.00 | ORB-short ORB[522.25,528.15] vol=2.2x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-02-25 11:00:00 | 523.42 | 524.19 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:35:00 | 529.00 | 527.34 | 0.00 | ORB-long ORB[523.05,528.55] vol=2.8x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-03-07 09:45:00 | 527.24 | 527.58 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-03-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:45:00 | 532.85 | 529.73 | 0.00 | ORB-long ORB[525.30,530.70] vol=2.2x ATR=1.73 |
| Stop hit — per-position SL triggered | 2025-03-10 09:55:00 | 531.12 | 530.16 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:40:00 | 512.35 | 508.80 | 0.00 | ORB-long ORB[502.05,509.50] vol=1.6x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 510.25 | 509.79 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-03-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:50:00 | 504.55 | 511.80 | 0.00 | ORB-short ORB[510.00,516.95] vol=2.0x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:25:00 | 502.04 | 508.38 | 0.00 | T1 1.5R @ 502.04 |
| Target hit | 2025-03-12 15:20:00 | 491.60 | 499.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:30:00 | 506.10 | 504.02 | 0.00 | ORB-long ORB[500.00,504.65] vol=1.9x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-03-18 09:35:00 | 504.74 | 504.21 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:40:00 | 520.70 | 518.85 | 0.00 | ORB-long ORB[514.00,520.10] vol=1.5x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:00:00 | 523.29 | 519.68 | 0.00 | T1 1.5R @ 523.29 |
| Target hit | 2025-03-19 15:20:00 | 525.20 | 522.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:10:00 | 525.95 | 529.59 | 0.00 | ORB-short ORB[527.60,534.25] vol=1.7x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-03-20 10:25:00 | 527.70 | 529.39 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-03-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 11:10:00 | 537.40 | 535.70 | 0.00 | ORB-long ORB[530.75,534.95] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-03-21 11:15:00 | 536.28 | 535.99 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 586.05 | 581.97 | 0.00 | ORB-long ORB[577.10,583.55] vol=1.6x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-04-21 09:35:00 | 583.49 | 582.43 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:20:00 | 591.50 | 585.39 | 0.00 | ORB-long ORB[576.45,584.15] vol=2.0x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:35:00 | 594.78 | 587.59 | 0.00 | T1 1.5R @ 594.78 |
| Stop hit — per-position SL triggered | 2025-04-22 10:40:00 | 591.50 | 587.92 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:00:00 | 580.90 | 581.86 | 0.00 | ORB-short ORB[581.15,587.70] vol=1.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:05:00 | 577.82 | 581.60 | 0.00 | T1 1.5R @ 577.82 |
| Stop hit — per-position SL triggered | 2025-04-23 10:25:00 | 580.90 | 581.07 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-04-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 10:40:00 | 574.85 | 579.51 | 0.00 | ORB-short ORB[579.75,584.60] vol=2.1x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-04-24 11:00:00 | 576.57 | 578.37 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 569.75 | 572.70 | 0.00 | ORB-short ORB[571.00,577.95] vol=1.5x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 566.55 | 570.67 | 0.00 | T1 1.5R @ 566.55 |
| Target hit | 2025-04-25 11:10:00 | 566.55 | 566.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 68 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 11:15:00 | 576.40 | 575.96 | 0.00 | ORB-long ORB[568.30,575.75] vol=2.3x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 12:15:00 | 579.37 | 577.32 | 0.00 | T1 1.5R @ 579.37 |
| Target hit | 2025-04-28 15:20:00 | 586.05 | 582.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2025-05-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 10:00:00 | 574.40 | 572.56 | 0.00 | ORB-long ORB[566.00,573.40] vol=1.6x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-05-02 10:05:00 | 572.49 | 571.16 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:50:00 | 566.85 | 564.24 | 0.00 | ORB-long ORB[557.45,565.80] vol=2.8x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-05-08 11:10:00 | 565.01 | 565.10 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 10:40:00 | 530.55 | 2024-05-16 12:05:00 | 532.59 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-17 10:35:00 | 539.80 | 2024-05-17 11:00:00 | 542.58 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-05-17 10:35:00 | 539.80 | 2024-05-17 12:50:00 | 541.45 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-05-23 10:00:00 | 584.90 | 2024-05-23 10:05:00 | 581.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-05-24 11:05:00 | 577.30 | 2024-05-24 11:25:00 | 575.16 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-07 11:05:00 | 537.55 | 2024-06-07 11:10:00 | 541.04 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-06-07 11:05:00 | 537.55 | 2024-06-07 11:25:00 | 537.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-18 10:10:00 | 537.70 | 2024-06-18 10:30:00 | 539.21 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-19 10:00:00 | 543.65 | 2024-06-19 10:05:00 | 541.48 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-24 11:10:00 | 545.05 | 2024-06-24 14:25:00 | 543.77 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-26 10:05:00 | 537.85 | 2024-06-26 10:20:00 | 540.66 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-06-26 10:05:00 | 537.85 | 2024-06-26 15:20:00 | 540.25 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-28 09:50:00 | 553.40 | 2024-06-28 10:05:00 | 550.76 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-07-02 09:35:00 | 543.60 | 2024-07-02 09:45:00 | 541.07 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-07-02 09:35:00 | 543.60 | 2024-07-02 14:25:00 | 537.75 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2024-07-09 09:50:00 | 537.15 | 2024-07-09 10:05:00 | 535.08 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-12 09:45:00 | 556.55 | 2024-07-12 09:55:00 | 554.45 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-24 09:55:00 | 562.50 | 2024-07-24 10:05:00 | 559.48 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-08-01 10:10:00 | 610.00 | 2024-08-01 10:30:00 | 613.00 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-08-01 10:10:00 | 610.00 | 2024-08-01 10:35:00 | 610.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 10:55:00 | 562.10 | 2024-08-08 11:10:00 | 563.73 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-09 10:55:00 | 555.70 | 2024-08-09 11:00:00 | 553.24 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-09 10:55:00 | 555.70 | 2024-08-09 11:25:00 | 555.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 09:30:00 | 546.30 | 2024-08-14 09:40:00 | 543.32 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-08-14 09:30:00 | 546.30 | 2024-08-14 09:45:00 | 546.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 10:55:00 | 558.35 | 2024-08-20 11:30:00 | 556.64 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-21 10:35:00 | 553.80 | 2024-08-21 11:15:00 | 555.13 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-26 10:25:00 | 549.45 | 2024-08-26 12:00:00 | 547.18 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-08-26 10:25:00 | 549.45 | 2024-08-26 12:10:00 | 549.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-02 11:05:00 | 565.75 | 2024-09-02 11:40:00 | 564.02 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-05 09:55:00 | 531.85 | 2024-09-05 11:25:00 | 529.51 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-05 09:55:00 | 531.85 | 2024-09-05 15:00:00 | 531.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-10 10:15:00 | 519.85 | 2024-09-10 11:40:00 | 517.35 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-09-10 10:15:00 | 519.85 | 2024-09-10 12:30:00 | 519.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 10:30:00 | 516.15 | 2024-09-18 11:30:00 | 514.13 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-09-18 10:30:00 | 516.15 | 2024-09-18 15:20:00 | 505.00 | TARGET_HIT | 0.50 | 2.16% |
| BUY | retest1 | 2024-09-23 11:00:00 | 527.90 | 2024-09-23 11:05:00 | 530.50 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-09-23 11:00:00 | 527.90 | 2024-09-23 15:20:00 | 534.25 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2024-09-24 10:55:00 | 527.40 | 2024-09-24 11:15:00 | 529.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-25 10:35:00 | 522.80 | 2024-09-25 11:50:00 | 520.66 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-09-25 10:35:00 | 522.80 | 2024-09-25 13:25:00 | 522.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:15:00 | 538.00 | 2024-09-27 10:20:00 | 535.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-10-03 10:55:00 | 524.80 | 2024-10-03 11:05:00 | 527.65 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-10-03 10:55:00 | 524.80 | 2024-10-03 11:20:00 | 524.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 10:05:00 | 521.50 | 2024-10-08 10:30:00 | 525.25 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-10-08 10:05:00 | 521.50 | 2024-10-08 15:20:00 | 528.80 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2024-10-16 10:20:00 | 530.05 | 2024-10-16 10:30:00 | 528.05 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-17 10:05:00 | 512.55 | 2024-10-17 10:10:00 | 514.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-22 10:50:00 | 502.70 | 2024-10-22 12:30:00 | 504.83 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-24 10:25:00 | 506.50 | 2024-10-24 10:55:00 | 504.76 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-25 09:40:00 | 496.65 | 2024-10-25 09:50:00 | 493.61 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-10-25 09:40:00 | 496.65 | 2024-10-25 10:20:00 | 494.80 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2024-11-06 09:45:00 | 576.20 | 2024-11-06 11:20:00 | 572.21 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-11-06 09:45:00 | 576.20 | 2024-11-06 12:35:00 | 576.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 09:30:00 | 551.40 | 2024-11-13 09:40:00 | 547.74 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-11-13 09:30:00 | 551.40 | 2024-11-13 10:30:00 | 551.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:50:00 | 565.25 | 2024-11-27 09:55:00 | 563.10 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-04 09:40:00 | 599.30 | 2024-12-04 09:50:00 | 596.60 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-12-10 09:50:00 | 602.30 | 2024-12-10 10:20:00 | 599.51 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-12-11 09:45:00 | 595.20 | 2024-12-11 10:05:00 | 592.30 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-12-11 09:45:00 | 595.20 | 2024-12-11 15:20:00 | 583.30 | TARGET_HIT | 0.50 | 2.00% |
| SELL | retest1 | 2025-01-01 10:55:00 | 526.10 | 2025-01-01 11:20:00 | 528.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-01-03 09:30:00 | 528.60 | 2025-01-03 09:35:00 | 531.43 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-03 09:30:00 | 528.60 | 2025-01-03 10:10:00 | 528.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 10:50:00 | 502.75 | 2025-01-09 10:55:00 | 500.87 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-01-09 10:50:00 | 502.75 | 2025-01-09 12:15:00 | 501.95 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-01-14 10:35:00 | 490.00 | 2025-01-14 10:45:00 | 492.95 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-14 10:35:00 | 490.00 | 2025-01-14 15:20:00 | 509.10 | TARGET_HIT | 0.50 | 3.90% |
| BUY | retest1 | 2025-01-15 11:05:00 | 510.15 | 2025-01-15 11:30:00 | 508.52 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-16 09:40:00 | 516.55 | 2025-01-16 09:55:00 | 519.33 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-16 09:40:00 | 516.55 | 2025-01-16 15:20:00 | 527.55 | TARGET_HIT | 0.50 | 2.13% |
| SELL | retest1 | 2025-01-17 11:00:00 | 520.20 | 2025-01-17 11:05:00 | 521.81 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-21 10:20:00 | 528.05 | 2025-01-21 11:20:00 | 525.53 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-21 10:20:00 | 528.05 | 2025-01-21 11:45:00 | 528.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 10:50:00 | 487.35 | 2025-01-27 10:55:00 | 489.35 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-02-07 10:40:00 | 552.00 | 2025-02-07 10:45:00 | 549.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-02-20 10:20:00 | 525.70 | 2025-02-20 11:30:00 | 528.70 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-02-20 10:20:00 | 525.70 | 2025-02-20 12:40:00 | 525.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-25 10:50:00 | 521.85 | 2025-02-25 11:00:00 | 523.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-07 09:35:00 | 529.00 | 2025-03-07 09:45:00 | 527.24 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-10 09:45:00 | 532.85 | 2025-03-10 09:55:00 | 531.12 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-11 10:40:00 | 512.35 | 2025-03-11 11:15:00 | 510.25 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-03-12 10:50:00 | 504.55 | 2025-03-12 11:25:00 | 502.04 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-03-12 10:50:00 | 504.55 | 2025-03-12 15:20:00 | 491.60 | TARGET_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2025-03-18 09:30:00 | 506.10 | 2025-03-18 09:35:00 | 504.74 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-03-19 09:40:00 | 520.70 | 2025-03-19 10:00:00 | 523.29 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-03-19 09:40:00 | 520.70 | 2025-03-19 15:20:00 | 525.20 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2025-03-20 10:10:00 | 525.95 | 2025-03-20 10:25:00 | 527.70 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-21 11:10:00 | 537.40 | 2025-03-21 11:15:00 | 536.28 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-04-21 09:30:00 | 586.05 | 2025-04-21 09:35:00 | 583.49 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-22 10:20:00 | 591.50 | 2025-04-22 10:35:00 | 594.78 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-04-22 10:20:00 | 591.50 | 2025-04-22 10:40:00 | 591.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-23 10:00:00 | 580.90 | 2025-04-23 10:05:00 | 577.82 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-04-23 10:00:00 | 580.90 | 2025-04-23 10:25:00 | 580.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-24 10:40:00 | 574.85 | 2025-04-24 11:00:00 | 576.57 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-04-25 09:30:00 | 569.75 | 2025-04-25 09:45:00 | 566.55 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-04-25 09:30:00 | 569.75 | 2025-04-25 11:10:00 | 566.55 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2025-04-28 11:15:00 | 576.40 | 2025-04-28 12:15:00 | 579.37 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-28 11:15:00 | 576.40 | 2025-04-28 15:20:00 | 586.05 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2025-05-02 10:00:00 | 574.40 | 2025-05-02 10:05:00 | 572.49 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-08 10:50:00 | 566.85 | 2025-05-08 11:10:00 | 565.01 | STOP_HIT | 1.00 | -0.33% |
