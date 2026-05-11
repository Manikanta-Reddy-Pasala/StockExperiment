# Gallantt Ispat Ltd. (GALLANTT)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 866.00
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
| ENTRY1 | 49 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 7 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 42
- **Target hits / Stop hits / Partials:** 7 / 42 / 20
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 12.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 16 | 36.4% | 2 | 28 | 14 | 0.18% | 7.7% |
| BUY @ 2nd Alert (retest1) | 44 | 16 | 36.4% | 2 | 28 | 14 | 0.18% | 7.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 11 | 44.0% | 5 | 14 | 6 | 0.19% | 4.6% |
| SELL @ 2nd Alert (retest1) | 25 | 11 | 44.0% | 5 | 14 | 6 | 0.19% | 4.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 69 | 27 | 39.1% | 7 | 42 | 20 | 0.18% | 12.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 11:05:00 | 464.95 | 466.61 | 0.00 | ORB-short ORB[465.15,471.45] vol=3.4x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-05-16 11:15:00 | 466.30 | 466.58 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 11:00:00 | 468.30 | 476.65 | 0.00 | ORB-short ORB[477.00,483.00] vol=2.0x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-05-23 11:05:00 | 469.88 | 476.13 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:50:00 | 457.05 | 452.41 | 0.00 | ORB-long ORB[449.05,455.05] vol=2.3x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:00:00 | 460.03 | 453.35 | 0.00 | T1 1.5R @ 460.03 |
| Stop hit — per-position SL triggered | 2025-05-28 13:55:00 | 457.05 | 456.77 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 09:55:00 | 460.15 | 462.60 | 0.00 | ORB-short ORB[461.60,467.40] vol=2.5x ATR=1.62 |
| Stop hit — per-position SL triggered | 2025-05-30 11:20:00 | 461.77 | 461.55 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:10:00 | 450.75 | 442.97 | 0.00 | ORB-long ORB[440.25,444.05] vol=2.1x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 449.07 | 443.63 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 11:15:00 | 532.10 | 538.38 | 0.00 | ORB-short ORB[536.75,543.25] vol=1.6x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:25:00 | 529.83 | 537.75 | 0.00 | T1 1.5R @ 529.83 |
| Target hit | 2025-06-27 15:20:00 | 525.10 | 533.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 526.45 | 530.60 | 0.00 | ORB-short ORB[528.55,535.50] vol=2.6x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 13:15:00 | 523.33 | 527.77 | 0.00 | T1 1.5R @ 523.33 |
| Target hit | 2025-07-02 15:20:00 | 524.15 | 526.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-07-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:50:00 | 588.50 | 583.96 | 0.00 | ORB-long ORB[579.00,587.40] vol=1.5x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:55:00 | 593.41 | 585.81 | 0.00 | T1 1.5R @ 593.41 |
| Stop hit — per-position SL triggered | 2025-07-14 10:00:00 | 588.50 | 587.36 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:55:00 | 576.95 | 584.29 | 0.00 | ORB-short ORB[586.15,593.00] vol=2.3x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-07-16 10:00:00 | 579.04 | 583.65 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:50:00 | 603.50 | 597.58 | 0.00 | ORB-long ORB[590.45,599.15] vol=8.4x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-07-17 09:55:00 | 600.33 | 597.70 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 630.30 | 625.61 | 0.00 | ORB-long ORB[618.10,627.50] vol=2.1x ATR=2.74 |
| Stop hit — per-position SL triggered | 2025-07-23 09:35:00 | 627.56 | 626.58 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:35:00 | 637.25 | 634.80 | 0.00 | ORB-long ORB[628.10,635.90] vol=5.6x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-07-24 09:40:00 | 634.27 | 634.82 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-08-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:05:00 | 632.15 | 629.27 | 0.00 | ORB-long ORB[626.15,630.45] vol=1.5x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-08-22 10:40:00 | 629.96 | 629.56 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:35:00 | 585.90 | 579.95 | 0.00 | ORB-long ORB[575.25,583.40] vol=3.7x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:40:00 | 590.20 | 582.33 | 0.00 | T1 1.5R @ 590.20 |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 585.90 | 582.62 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 583.05 | 579.91 | 0.00 | ORB-long ORB[574.95,582.35] vol=5.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:50:00 | 586.90 | 581.39 | 0.00 | T1 1.5R @ 586.90 |
| Stop hit — per-position SL triggered | 2025-09-01 10:05:00 | 583.05 | 581.58 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:35:00 | 599.40 | 593.88 | 0.00 | ORB-long ORB[588.15,596.65] vol=2.2x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:40:00 | 603.67 | 596.72 | 0.00 | T1 1.5R @ 603.67 |
| Target hit | 2025-09-02 12:15:00 | 619.45 | 619.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2025-09-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 09:35:00 | 674.15 | 679.61 | 0.00 | ORB-short ORB[675.55,684.00] vol=2.4x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-09-16 09:50:00 | 677.29 | 677.97 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:35:00 | 661.20 | 664.01 | 0.00 | ORB-short ORB[663.75,673.40] vol=2.3x ATR=2.66 |
| Stop hit — per-position SL triggered | 2025-09-17 09:55:00 | 663.86 | 662.54 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:00:00 | 674.60 | 667.71 | 0.00 | ORB-long ORB[661.50,669.35] vol=2.0x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:05:00 | 679.50 | 673.46 | 0.00 | T1 1.5R @ 679.50 |
| Stop hit — per-position SL triggered | 2025-09-24 10:55:00 | 674.60 | 674.71 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:10:00 | 667.80 | 665.88 | 0.00 | ORB-long ORB[660.50,666.40] vol=1.9x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:30:00 | 671.51 | 666.93 | 0.00 | T1 1.5R @ 671.51 |
| Stop hit — per-position SL triggered | 2025-10-03 11:05:00 | 667.80 | 668.02 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 09:35:00 | 662.25 | 665.77 | 0.00 | ORB-short ORB[663.00,669.15] vol=1.8x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:40:00 | 658.23 | 663.68 | 0.00 | T1 1.5R @ 658.23 |
| Target hit | 2025-10-07 10:30:00 | 660.25 | 659.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — BUY (started 2025-10-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:55:00 | 659.45 | 654.89 | 0.00 | ORB-long ORB[648.00,656.65] vol=1.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-10-15 11:00:00 | 657.56 | 655.42 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 660.60 | 657.86 | 0.00 | ORB-long ORB[651.45,659.80] vol=1.7x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-10-17 09:40:00 | 659.00 | 658.13 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:30:00 | 537.50 | 534.39 | 0.00 | ORB-long ORB[531.25,534.80] vol=2.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 09:35:00 | 541.35 | 537.59 | 0.00 | T1 1.5R @ 541.35 |
| Stop hit — per-position SL triggered | 2025-10-29 09:55:00 | 537.50 | 538.46 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 11:05:00 | 530.90 | 534.27 | 0.00 | ORB-short ORB[531.85,539.45] vol=1.7x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:55:00 | 528.91 | 533.64 | 0.00 | T1 1.5R @ 528.91 |
| Target hit | 2025-10-30 15:20:00 | 521.85 | 527.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-10-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:50:00 | 527.80 | 527.45 | 0.00 | ORB-long ORB[521.80,527.10] vol=3.3x ATR=2.05 |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 525.75 | 527.39 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:30:00 | 536.25 | 531.67 | 0.00 | ORB-long ORB[525.00,530.90] vol=5.6x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:35:00 | 540.01 | 533.12 | 0.00 | T1 1.5R @ 540.01 |
| Stop hit — per-position SL triggered | 2025-11-03 09:40:00 | 536.25 | 534.09 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 597.10 | 589.83 | 0.00 | ORB-long ORB[585.80,592.50] vol=1.9x ATR=3.55 |
| Stop hit — per-position SL triggered | 2025-11-14 09:35:00 | 593.55 | 590.67 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-11-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 09:40:00 | 587.10 | 591.47 | 0.00 | ORB-short ORB[591.30,598.50] vol=2.9x ATR=2.95 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 590.05 | 589.71 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:50:00 | 600.20 | 595.01 | 0.00 | ORB-long ORB[591.00,599.00] vol=3.0x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:55:00 | 604.55 | 597.67 | 0.00 | T1 1.5R @ 604.55 |
| Stop hit — per-position SL triggered | 2025-11-27 10:00:00 | 600.20 | 597.87 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 598.15 | 594.71 | 0.00 | ORB-long ORB[590.40,595.85] vol=2.3x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-11-28 10:00:00 | 595.76 | 595.87 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:45:00 | 589.00 | 590.54 | 0.00 | ORB-short ORB[589.30,593.45] vol=5.9x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-12-01 11:00:00 | 590.31 | 590.20 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-12-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:00:00 | 597.95 | 595.54 | 0.00 | ORB-long ORB[587.75,594.00] vol=5.5x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 10:05:00 | 600.83 | 596.59 | 0.00 | T1 1.5R @ 600.83 |
| Stop hit — per-position SL triggered | 2025-12-02 10:10:00 | 597.95 | 596.60 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-12-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:45:00 | 586.00 | 589.01 | 0.00 | ORB-short ORB[588.00,591.65] vol=4.3x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 587.89 | 588.18 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-12-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:10:00 | 588.30 | 591.36 | 0.00 | ORB-short ORB[590.25,594.85] vol=14.0x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-12-15 11:20:00 | 589.33 | 590.01 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-12-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 10:05:00 | 524.45 | 521.91 | 0.00 | ORB-long ORB[518.30,521.95] vol=3.1x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:40:00 | 527.22 | 523.30 | 0.00 | T1 1.5R @ 527.22 |
| Target hit | 2025-12-29 13:15:00 | 535.00 | 535.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2025-12-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:05:00 | 521.60 | 524.60 | 0.00 | ORB-short ORB[522.30,527.95] vol=1.6x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-12-30 11:20:00 | 523.17 | 524.61 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2026-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:50:00 | 544.45 | 542.64 | 0.00 | ORB-long ORB[537.15,542.00] vol=2.0x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:50:00 | 546.96 | 543.93 | 0.00 | T1 1.5R @ 546.96 |
| Stop hit — per-position SL triggered | 2026-01-02 11:05:00 | 544.45 | 543.99 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-01-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:10:00 | 563.20 | 557.13 | 0.00 | ORB-long ORB[551.00,559.00] vol=2.5x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-01-05 10:35:00 | 560.43 | 558.71 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:40:00 | 563.30 | 560.85 | 0.00 | ORB-long ORB[556.70,561.95] vol=2.5x ATR=2.07 |
| Stop hit — per-position SL triggered | 2026-01-06 10:05:00 | 561.23 | 561.23 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-01-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 10:45:00 | 569.70 | 562.92 | 0.00 | ORB-long ORB[564.10,569.40] vol=2.7x ATR=3.12 |
| Stop hit — per-position SL triggered | 2026-01-08 11:00:00 | 566.58 | 563.29 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 11:15:00 | 557.95 | 555.67 | 0.00 | ORB-long ORB[551.25,554.35] vol=1.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2026-01-16 11:20:00 | 556.31 | 555.73 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:30:00 | 536.10 | 539.54 | 0.00 | ORB-short ORB[538.75,545.55] vol=2.2x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:45:00 | 533.42 | 537.55 | 0.00 | T1 1.5R @ 533.42 |
| Stop hit — per-position SL triggered | 2026-01-23 14:00:00 | 536.10 | 534.50 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-01-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:30:00 | 519.25 | 522.01 | 0.00 | ORB-short ORB[521.00,526.50] vol=2.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 520.55 | 521.34 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:35:00 | 516.85 | 513.00 | 0.00 | ORB-long ORB[509.25,514.90] vol=2.2x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 12:10:00 | 519.74 | 515.69 | 0.00 | T1 1.5R @ 519.74 |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 516.85 | 516.19 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 562.45 | 555.95 | 0.00 | ORB-long ORB[550.00,556.00] vol=1.5x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-02-25 10:05:00 | 559.67 | 556.83 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 537.70 | 540.64 | 0.00 | ORB-short ORB[538.00,545.30] vol=2.7x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 535.81 | 539.94 | 0.00 | T1 1.5R @ 535.81 |
| Target hit | 2026-03-11 15:20:00 | 524.65 | 528.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2026-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:45:00 | 542.15 | 538.29 | 0.00 | ORB-long ORB[535.65,541.75] vol=1.5x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-03-17 11:20:00 | 539.82 | 539.87 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:00:00 | 543.60 | 547.89 | 0.00 | ORB-short ORB[546.20,554.10] vol=2.1x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-03-18 10:25:00 | 545.84 | 547.07 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-16 11:05:00 | 464.95 | 2025-05-16 11:15:00 | 466.30 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-23 11:00:00 | 468.30 | 2025-05-23 11:05:00 | 469.88 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-05-28 09:50:00 | 457.05 | 2025-05-28 10:00:00 | 460.03 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-05-28 09:50:00 | 457.05 | 2025-05-28 13:55:00 | 457.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 09:55:00 | 460.15 | 2025-05-30 11:20:00 | 461.77 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-10 10:10:00 | 450.75 | 2025-06-10 10:15:00 | 449.07 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-06-27 11:15:00 | 532.10 | 2025-06-27 11:25:00 | 529.83 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-27 11:15:00 | 532.10 | 2025-06-27 15:20:00 | 525.10 | TARGET_HIT | 0.50 | 1.32% |
| SELL | retest1 | 2025-07-02 09:40:00 | 526.45 | 2025-07-02 13:15:00 | 523.33 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-07-02 09:40:00 | 526.45 | 2025-07-02 15:20:00 | 524.15 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2025-07-14 09:50:00 | 588.50 | 2025-07-14 09:55:00 | 593.41 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2025-07-14 09:50:00 | 588.50 | 2025-07-14 10:00:00 | 588.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-16 09:55:00 | 576.95 | 2025-07-16 10:00:00 | 579.04 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-07-17 09:50:00 | 603.50 | 2025-07-17 09:55:00 | 600.33 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-07-23 09:30:00 | 630.30 | 2025-07-23 09:35:00 | 627.56 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-07-24 09:35:00 | 637.25 | 2025-07-24 09:40:00 | 634.27 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-08-22 10:05:00 | 632.15 | 2025-08-22 10:40:00 | 629.96 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-29 10:35:00 | 585.90 | 2025-08-29 11:40:00 | 590.20 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-08-29 10:35:00 | 585.90 | 2025-08-29 12:15:00 | 585.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 09:45:00 | 583.05 | 2025-09-01 09:50:00 | 586.90 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-09-01 09:45:00 | 583.05 | 2025-09-01 10:05:00 | 583.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 09:35:00 | 599.40 | 2025-09-02 09:40:00 | 603.67 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-09-02 09:35:00 | 599.40 | 2025-09-02 12:15:00 | 619.45 | TARGET_HIT | 0.50 | 3.35% |
| SELL | retest1 | 2025-09-16 09:35:00 | 674.15 | 2025-09-16 09:50:00 | 677.29 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-09-17 09:35:00 | 661.20 | 2025-09-17 09:55:00 | 663.86 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-09-24 10:00:00 | 674.60 | 2025-09-24 10:05:00 | 679.50 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-09-24 10:00:00 | 674.60 | 2025-09-24 10:55:00 | 674.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-03 10:10:00 | 667.80 | 2025-10-03 10:30:00 | 671.51 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-10-03 10:10:00 | 667.80 | 2025-10-03 11:05:00 | 667.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-07 09:35:00 | 662.25 | 2025-10-07 09:40:00 | 658.23 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-10-07 09:35:00 | 662.25 | 2025-10-07 10:30:00 | 660.25 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-10-15 10:55:00 | 659.45 | 2025-10-15 11:00:00 | 657.56 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-17 09:30:00 | 660.60 | 2025-10-17 09:40:00 | 659.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-29 09:30:00 | 537.50 | 2025-10-29 09:35:00 | 541.35 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-10-29 09:30:00 | 537.50 | 2025-10-29 09:55:00 | 537.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 11:05:00 | 530.90 | 2025-10-30 11:55:00 | 528.91 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-30 11:05:00 | 530.90 | 2025-10-30 15:20:00 | 521.85 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2025-10-31 09:50:00 | 527.80 | 2025-10-31 10:15:00 | 525.75 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-11-03 09:30:00 | 536.25 | 2025-11-03 09:35:00 | 540.01 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-11-03 09:30:00 | 536.25 | 2025-11-03 09:40:00 | 536.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 09:30:00 | 597.10 | 2025-11-14 09:35:00 | 593.55 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2025-11-24 09:40:00 | 587.10 | 2025-11-24 10:15:00 | 590.05 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-11-27 09:50:00 | 600.20 | 2025-11-27 09:55:00 | 604.55 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-11-27 09:50:00 | 600.20 | 2025-11-27 10:00:00 | 600.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-28 09:30:00 | 598.15 | 2025-11-28 10:00:00 | 595.76 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-12-01 10:45:00 | 589.00 | 2025-12-01 11:00:00 | 590.31 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-02 10:00:00 | 597.95 | 2025-12-02 10:05:00 | 600.83 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-12-02 10:00:00 | 597.95 | 2025-12-02 10:10:00 | 597.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 09:45:00 | 586.00 | 2025-12-05 10:00:00 | 587.89 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-15 11:10:00 | 588.30 | 2025-12-15 11:20:00 | 589.33 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-29 10:05:00 | 524.45 | 2025-12-29 10:40:00 | 527.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-12-29 10:05:00 | 524.45 | 2025-12-29 13:15:00 | 535.00 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2025-12-30 11:05:00 | 521.60 | 2025-12-30 11:20:00 | 523.17 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-02 09:50:00 | 544.45 | 2026-01-02 10:50:00 | 546.96 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-02 09:50:00 | 544.45 | 2026-01-02 11:05:00 | 544.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-05 10:10:00 | 563.20 | 2026-01-05 10:35:00 | 560.43 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-01-06 09:40:00 | 563.30 | 2026-01-06 10:05:00 | 561.23 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-01-08 10:45:00 | 569.70 | 2026-01-08 11:00:00 | 566.58 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-01-16 11:15:00 | 557.95 | 2026-01-16 11:20:00 | 556.31 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-23 10:30:00 | 536.10 | 2026-01-23 10:45:00 | 533.42 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-01-23 10:30:00 | 536.10 | 2026-01-23 14:00:00 | 536.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 10:30:00 | 519.25 | 2026-01-29 11:15:00 | 520.55 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-30 09:35:00 | 516.85 | 2026-01-30 12:10:00 | 519.74 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-01-30 09:35:00 | 516.85 | 2026-01-30 12:15:00 | 516.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:00:00 | 562.45 | 2026-02-25 10:05:00 | 559.67 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-11 10:35:00 | 537.70 | 2026-03-11 11:20:00 | 535.81 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-03-11 10:35:00 | 537.70 | 2026-03-11 15:20:00 | 524.65 | TARGET_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2026-03-17 09:45:00 | 542.15 | 2026-03-17 11:20:00 | 539.82 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-18 10:00:00 | 543.60 | 2026-03-18 10:25:00 | 545.84 | STOP_HIT | 1.00 | -0.41% |
