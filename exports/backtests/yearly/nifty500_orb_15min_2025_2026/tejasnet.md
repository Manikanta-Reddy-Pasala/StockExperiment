# Tejas Networks Ltd. (TEJASNET)

## Backtest Summary

- **Window:** 2025-08-11 09:15:00 → 2026-05-08 15:25:00 (13588 bars)
- **Last close:** 515.50
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
| ENTRY1 | 35 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 30
- **Target hits / Stop hits / Partials:** 5 / 30 / 10
- **Avg / median % per leg:** 0.18% / -0.21%
- **Sum % (uncompounded):** 8.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 8 | 27.6% | 3 | 21 | 5 | 0.18% | 5.2% |
| BUY @ 2nd Alert (retest1) | 29 | 8 | 27.6% | 3 | 21 | 5 | 0.18% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.19% | 3.1% |
| SELL @ 2nd Alert (retest1) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.19% | 3.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 45 | 15 | 33.3% | 5 | 30 | 10 | 0.18% | 8.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:30:00 | 580.50 | 575.83 | 0.00 | ORB-long ORB[571.85,578.80] vol=2.0x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-08-19 10:35:00 | 578.37 | 575.90 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:30:00 | 591.75 | 585.69 | 0.00 | ORB-long ORB[580.55,583.45] vol=4.9x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 09:35:00 | 594.86 | 590.50 | 0.00 | T1 1.5R @ 594.86 |
| Target hit | 2025-08-21 10:45:00 | 626.80 | 628.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 591.40 | 586.76 | 0.00 | ORB-long ORB[582.55,587.90] vol=2.5x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-09-01 09:55:00 | 589.47 | 587.19 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 11:00:00 | 591.20 | 594.37 | 0.00 | ORB-short ORB[592.20,600.30] vol=1.6x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 11:15:00 | 588.57 | 593.80 | 0.00 | T1 1.5R @ 588.57 |
| Stop hit — per-position SL triggered | 2025-09-05 13:45:00 | 591.20 | 590.63 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-09-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:20:00 | 586.50 | 589.31 | 0.00 | ORB-short ORB[587.15,595.00] vol=1.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-09-09 10:35:00 | 588.11 | 588.75 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 595.25 | 592.66 | 0.00 | ORB-long ORB[589.05,593.95] vol=2.2x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-09-10 09:45:00 | 593.46 | 592.73 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:15:00 | 598.95 | 593.60 | 0.00 | ORB-long ORB[588.30,593.90] vol=2.8x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-09-12 10:20:00 | 597.29 | 597.47 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-09-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:55:00 | 605.90 | 601.32 | 0.00 | ORB-long ORB[596.30,603.00] vol=2.0x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 11:00:00 | 608.57 | 606.55 | 0.00 | T1 1.5R @ 608.57 |
| Target hit | 2025-09-16 12:55:00 | 616.30 | 616.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:50:00 | 622.70 | 619.19 | 0.00 | ORB-long ORB[613.10,620.00] vol=2.0x ATR=2.64 |
| Stop hit — per-position SL triggered | 2025-09-17 10:05:00 | 620.06 | 619.55 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:35:00 | 622.25 | 619.94 | 0.00 | ORB-long ORB[616.05,622.20] vol=2.7x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-09-18 09:40:00 | 620.13 | 620.30 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 592.20 | 597.72 | 0.00 | ORB-short ORB[595.50,601.00] vol=1.8x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 14:25:00 | 590.01 | 596.20 | 0.00 | T1 1.5R @ 590.01 |
| Target hit | 2025-09-24 15:20:00 | 588.90 | 594.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-10-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:10:00 | 607.60 | 599.98 | 0.00 | ORB-long ORB[592.40,599.00] vol=9.1x ATR=3.07 |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 604.53 | 601.25 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-10-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 09:35:00 | 596.85 | 595.31 | 0.00 | ORB-long ORB[588.75,596.80] vol=2.4x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-10-08 09:50:00 | 595.06 | 595.50 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-10-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:55:00 | 601.00 | 597.25 | 0.00 | ORB-long ORB[594.95,598.85] vol=5.8x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-10-10 11:05:00 | 599.47 | 597.82 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 593.20 | 594.86 | 0.00 | ORB-short ORB[594.25,601.00] vol=5.2x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:10:00 | 590.83 | 594.00 | 0.00 | T1 1.5R @ 590.83 |
| Target hit | 2025-10-14 15:20:00 | 584.10 | 588.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:15:00 | 588.75 | 585.46 | 0.00 | ORB-long ORB[584.05,587.90] vol=3.9x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-10-15 11:20:00 | 587.38 | 585.51 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:45:00 | 596.20 | 592.86 | 0.00 | ORB-long ORB[587.50,595.85] vol=1.8x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-10-16 09:55:00 | 594.25 | 593.06 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:50:00 | 598.10 | 594.57 | 0.00 | ORB-long ORB[592.00,598.00] vol=2.9x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-10-17 10:20:00 | 596.13 | 595.71 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-10-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 11:05:00 | 538.90 | 541.65 | 0.00 | ORB-short ORB[540.40,545.65] vol=2.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-10-30 11:10:00 | 539.87 | 541.57 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-10-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:05:00 | 534.85 | 537.32 | 0.00 | ORB-short ORB[536.60,540.50] vol=2.2x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-10-31 11:25:00 | 535.76 | 537.11 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-11-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:30:00 | 543.00 | 538.55 | 0.00 | ORB-long ORB[532.25,540.00] vol=1.9x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-11-03 10:35:00 | 541.41 | 539.24 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:15:00 | 531.00 | 535.71 | 0.00 | ORB-short ORB[538.55,542.20] vol=2.1x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 532.21 | 534.65 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-11-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:55:00 | 517.70 | 521.48 | 0.00 | ORB-short ORB[522.30,525.70] vol=4.0x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:10:00 | 515.53 | 520.31 | 0.00 | T1 1.5R @ 515.53 |
| Stop hit — per-position SL triggered | 2025-11-10 10:20:00 | 517.70 | 520.00 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:45:00 | 516.95 | 513.14 | 0.00 | ORB-long ORB[507.00,514.65] vol=1.8x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-11-12 09:55:00 | 515.30 | 514.05 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-11-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:05:00 | 509.20 | 511.83 | 0.00 | ORB-short ORB[510.75,515.80] vol=2.2x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-11-19 10:30:00 | 510.50 | 510.85 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-12-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:50:00 | 455.20 | 450.76 | 0.00 | ORB-long ORB[445.20,449.40] vol=5.9x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-12-29 09:55:00 | 453.59 | 451.07 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2026-01-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 11:10:00 | 453.60 | 450.55 | 0.00 | ORB-long ORB[449.00,452.10] vol=3.5x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:20:00 | 455.34 | 451.36 | 0.00 | T1 1.5R @ 455.34 |
| Stop hit — per-position SL triggered | 2026-01-02 11:40:00 | 453.60 | 451.67 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:15:00 | 445.55 | 447.22 | 0.00 | ORB-short ORB[446.00,451.10] vol=1.6x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-01-06 11:30:00 | 446.48 | 447.02 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2026-01-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 11:05:00 | 448.15 | 444.92 | 0.00 | ORB-long ORB[443.00,446.55] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-01-07 11:10:00 | 446.94 | 445.01 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2026-01-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:45:00 | 372.30 | 368.93 | 0.00 | ORB-long ORB[367.30,371.85] vol=1.8x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-01-16 09:55:00 | 370.60 | 369.21 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2026-01-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:05:00 | 323.50 | 327.91 | 0.00 | ORB-short ORB[328.35,333.00] vol=1.7x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:40:00 | 321.36 | 326.45 | 0.00 | T1 1.5R @ 321.36 |
| Stop hit — per-position SL triggered | 2026-01-22 14:45:00 | 323.50 | 323.68 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2026-02-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:45:00 | 340.70 | 337.29 | 0.00 | ORB-long ORB[333.25,337.75] vol=2.6x ATR=1.85 |
| Stop hit — per-position SL triggered | 2026-02-04 09:55:00 | 338.85 | 337.46 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 450.05 | 444.47 | 0.00 | ORB-long ORB[440.50,447.00] vol=2.9x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:20:00 | 453.57 | 447.68 | 0.00 | T1 1.5R @ 453.57 |
| Stop hit — per-position SL triggered | 2026-04-10 10:45:00 | 450.05 | 451.56 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 423.00 | 414.87 | 0.00 | ORB-long ORB[412.05,415.95] vol=8.3x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 421.12 | 416.08 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 420.15 | 418.00 | 0.00 | ORB-long ORB[414.50,419.10] vol=3.9x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:10:00 | 423.76 | 419.41 | 0.00 | T1 1.5R @ 423.76 |
| Target hit | 2026-05-04 11:30:00 | 425.15 | 425.63 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-19 10:30:00 | 580.50 | 2025-08-19 10:35:00 | 578.37 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-08-21 09:30:00 | 591.75 | 2025-08-21 09:35:00 | 594.86 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-08-21 09:30:00 | 591.75 | 2025-08-21 10:45:00 | 626.80 | TARGET_HIT | 0.50 | 5.92% |
| BUY | retest1 | 2025-09-01 09:45:00 | 591.40 | 2025-09-01 09:55:00 | 589.47 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-05 11:00:00 | 591.20 | 2025-09-05 11:15:00 | 588.57 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-09-05 11:00:00 | 591.20 | 2025-09-05 13:45:00 | 591.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-09 10:20:00 | 586.50 | 2025-09-09 10:35:00 | 588.11 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-10 09:40:00 | 595.25 | 2025-09-10 09:45:00 | 593.46 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-12 10:15:00 | 598.95 | 2025-09-12 10:20:00 | 597.29 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-16 10:55:00 | 605.90 | 2025-09-16 11:00:00 | 608.57 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-09-16 10:55:00 | 605.90 | 2025-09-16 12:55:00 | 616.30 | TARGET_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2025-09-17 09:50:00 | 622.70 | 2025-09-17 10:05:00 | 620.06 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-09-18 09:35:00 | 622.25 | 2025-09-18 09:40:00 | 620.13 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-09-24 11:15:00 | 592.20 | 2025-09-24 14:25:00 | 590.01 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-24 11:15:00 | 592.20 | 2025-09-24 15:20:00 | 588.90 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2025-10-06 10:10:00 | 607.60 | 2025-10-06 10:15:00 | 604.53 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-10-08 09:35:00 | 596.85 | 2025-10-08 09:50:00 | 595.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-10 10:55:00 | 601.00 | 2025-10-10 11:05:00 | 599.47 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-14 09:40:00 | 593.20 | 2025-10-14 10:10:00 | 590.83 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-14 09:40:00 | 593.20 | 2025-10-14 15:20:00 | 584.10 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2025-10-15 11:15:00 | 588.75 | 2025-10-15 11:20:00 | 587.38 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-16 09:45:00 | 596.20 | 2025-10-16 09:55:00 | 594.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-17 09:50:00 | 598.10 | 2025-10-17 10:20:00 | 596.13 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-30 11:05:00 | 538.90 | 2025-10-30 11:10:00 | 539.87 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-31 11:05:00 | 534.85 | 2025-10-31 11:25:00 | 535.76 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-03 10:30:00 | 543.00 | 2025-11-03 10:35:00 | 541.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-06 11:15:00 | 531.00 | 2025-11-06 12:15:00 | 532.21 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-10 09:55:00 | 517.70 | 2025-11-10 10:10:00 | 515.53 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-11-10 09:55:00 | 517.70 | 2025-11-10 10:20:00 | 517.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 09:45:00 | 516.95 | 2025-11-12 09:55:00 | 515.30 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-19 10:05:00 | 509.20 | 2025-11-19 10:30:00 | 510.50 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-29 09:50:00 | 455.20 | 2025-12-29 09:55:00 | 453.59 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-01-02 11:10:00 | 453.60 | 2026-01-02 11:20:00 | 455.34 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-02 11:10:00 | 453.60 | 2026-01-02 11:40:00 | 453.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-06 11:15:00 | 445.55 | 2026-01-06 11:30:00 | 446.48 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-07 11:05:00 | 448.15 | 2026-01-07 11:10:00 | 446.94 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-16 09:45:00 | 372.30 | 2026-01-16 09:55:00 | 370.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-01-22 11:05:00 | 323.50 | 2026-01-22 11:40:00 | 321.36 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-01-22 11:05:00 | 323.50 | 2026-01-22 14:45:00 | 323.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-04 09:45:00 | 340.70 | 2026-02-04 09:55:00 | 338.85 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-04-10 10:15:00 | 450.05 | 2026-04-10 10:20:00 | 453.57 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-04-10 10:15:00 | 450.05 | 2026-04-10 10:45:00 | 450.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:55:00 | 423.00 | 2026-04-29 11:00:00 | 421.12 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-05-04 09:35:00 | 420.15 | 2026-05-04 10:10:00 | 423.76 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2026-05-04 09:35:00 | 420.15 | 2026-05-04 11:30:00 | 425.15 | TARGET_HIT | 0.50 | 1.19% |
