# Vardhman Textiles Ltd. (VTL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (33784 bars)
- **Last close:** 583.10
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
| PARTIAL | 21 |
| TARGET_HIT | 9 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 47
- **Target hits / Stop hits / Partials:** 9 / 47 / 21
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 19.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 19 | 42.2% | 6 | 26 | 13 | 0.36% | 16.1% |
| BUY @ 2nd Alert (retest1) | 45 | 19 | 42.2% | 6 | 26 | 13 | 0.36% | 16.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 11 | 34.4% | 3 | 21 | 8 | 0.10% | 3.3% |
| SELL @ 2nd Alert (retest1) | 32 | 11 | 34.4% | 3 | 21 | 8 | 0.10% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 77 | 30 | 39.0% | 9 | 47 | 21 | 0.25% | 19.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:35:00 | 450.45 | 452.12 | 0.00 | ORB-short ORB[453.75,458.50] vol=3.8x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-05-17 09:45:00 | 452.70 | 451.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:15:00 | 438.00 | 438.94 | 0.00 | ORB-short ORB[439.05,445.10] vol=2.7x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-05-27 11:05:00 | 439.80 | 439.18 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:45:00 | 475.70 | 473.34 | 0.00 | ORB-long ORB[470.00,474.95] vol=1.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-06-07 11:50:00 | 473.62 | 473.83 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 475.00 | 476.95 | 0.00 | ORB-short ORB[475.05,479.45] vol=3.4x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-06-10 09:40:00 | 477.06 | 476.88 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:35:00 | 470.25 | 467.45 | 0.00 | ORB-long ORB[463.20,468.25] vol=2.7x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:40:00 | 472.85 | 468.36 | 0.00 | T1 1.5R @ 472.85 |
| Stop hit — per-position SL triggered | 2024-06-12 10:45:00 | 470.25 | 468.51 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 11:00:00 | 490.80 | 481.28 | 0.00 | ORB-long ORB[475.85,478.85] vol=11.6x ATR=2.23 |
| Stop hit — per-position SL triggered | 2024-06-25 11:05:00 | 488.57 | 482.90 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:50:00 | 488.50 | 484.06 | 0.00 | ORB-long ORB[480.95,487.95] vol=2.2x ATR=2.81 |
| Stop hit — per-position SL triggered | 2024-06-26 10:00:00 | 485.69 | 485.99 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 487.65 | 485.44 | 0.00 | ORB-long ORB[481.80,485.55] vol=3.0x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:45:00 | 490.42 | 487.78 | 0.00 | T1 1.5R @ 490.42 |
| Target hit | 2024-06-27 10:05:00 | 488.45 | 488.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2024-06-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 11:10:00 | 486.00 | 480.99 | 0.00 | ORB-long ORB[478.70,483.30] vol=6.1x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-06-28 11:20:00 | 484.15 | 481.94 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:50:00 | 486.85 | 483.52 | 0.00 | ORB-long ORB[480.40,484.60] vol=1.9x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 10:45:00 | 490.04 | 486.59 | 0.00 | T1 1.5R @ 490.04 |
| Target hit | 2024-07-01 11:45:00 | 491.45 | 491.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2024-07-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:50:00 | 483.55 | 486.25 | 0.00 | ORB-short ORB[485.00,488.55] vol=1.7x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-07-03 10:55:00 | 484.93 | 485.94 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:35:00 | 479.15 | 482.90 | 0.00 | ORB-short ORB[482.30,487.00] vol=1.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-07-04 11:10:00 | 480.37 | 481.80 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:15:00 | 493.00 | 487.35 | 0.00 | ORB-long ORB[484.20,491.20] vol=6.9x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 11:40:00 | 495.52 | 493.24 | 0.00 | T1 1.5R @ 495.52 |
| Target hit | 2024-07-05 15:15:00 | 527.15 | 532.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-07-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:35:00 | 522.35 | 517.21 | 0.00 | ORB-long ORB[513.10,518.70] vol=3.1x ATR=2.81 |
| Stop hit — per-position SL triggered | 2024-07-09 09:45:00 | 519.54 | 519.14 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:35:00 | 533.65 | 530.30 | 0.00 | ORB-long ORB[525.00,530.85] vol=3.2x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 09:40:00 | 538.56 | 536.71 | 0.00 | T1 1.5R @ 538.56 |
| Target hit | 2024-07-15 10:00:00 | 562.20 | 566.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2024-07-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:05:00 | 553.25 | 547.08 | 0.00 | ORB-long ORB[542.20,550.00] vol=4.3x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-07-16 11:15:00 | 550.42 | 547.38 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 552.70 | 557.14 | 0.00 | ORB-short ORB[555.30,561.85] vol=2.5x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:35:00 | 548.13 | 555.93 | 0.00 | T1 1.5R @ 548.13 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 552.70 | 555.54 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:30:00 | 538.45 | 534.11 | 0.00 | ORB-long ORB[531.10,537.80] vol=1.8x ATR=2.24 |
| Stop hit — per-position SL triggered | 2024-07-23 10:45:00 | 536.21 | 534.48 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 09:30:00 | 511.50 | 513.91 | 0.00 | ORB-short ORB[511.55,516.95] vol=1.7x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:50:00 | 508.05 | 512.40 | 0.00 | T1 1.5R @ 508.05 |
| Stop hit — per-position SL triggered | 2024-08-09 11:15:00 | 511.50 | 509.30 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 487.30 | 490.59 | 0.00 | ORB-short ORB[489.30,495.75] vol=1.6x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 489.74 | 489.95 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:35:00 | 489.45 | 492.46 | 0.00 | ORB-short ORB[491.80,497.45] vol=2.2x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-08-16 09:45:00 | 491.31 | 491.45 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:45:00 | 504.55 | 507.66 | 0.00 | ORB-short ORB[508.80,513.70] vol=2.7x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:50:00 | 502.69 | 507.51 | 0.00 | T1 1.5R @ 502.69 |
| Stop hit — per-position SL triggered | 2024-08-29 12:20:00 | 504.55 | 505.24 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:10:00 | 498.95 | 496.86 | 0.00 | ORB-long ORB[493.60,497.20] vol=9.1x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-09-10 10:15:00 | 497.24 | 497.22 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:45:00 | 505.00 | 499.95 | 0.00 | ORB-long ORB[494.85,498.45] vol=5.4x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-09-11 10:00:00 | 503.16 | 502.58 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:30:00 | 497.50 | 499.80 | 0.00 | ORB-short ORB[497.85,503.00] vol=1.7x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:10:00 | 494.56 | 498.88 | 0.00 | T1 1.5R @ 494.56 |
| Stop hit — per-position SL triggered | 2024-09-12 10:25:00 | 497.50 | 498.70 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:00:00 | 486.90 | 490.27 | 0.00 | ORB-short ORB[488.70,493.15] vol=2.3x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-09-19 10:25:00 | 488.53 | 488.40 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:45:00 | 473.10 | 474.73 | 0.00 | ORB-short ORB[474.15,478.20] vol=1.6x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 13:10:00 | 471.05 | 473.22 | 0.00 | T1 1.5R @ 471.05 |
| Target hit | 2024-09-25 15:20:00 | 468.90 | 471.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:35:00 | 471.15 | 468.67 | 0.00 | ORB-long ORB[462.50,468.85] vol=4.1x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 469.09 | 468.72 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:50:00 | 460.70 | 463.19 | 0.00 | ORB-short ORB[465.00,470.50] vol=1.9x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:55:00 | 458.29 | 462.95 | 0.00 | T1 1.5R @ 458.29 |
| Target hit | 2024-10-07 15:20:00 | 443.45 | 450.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2024-11-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:20:00 | 477.85 | 479.83 | 0.00 | ORB-short ORB[478.90,484.95] vol=1.9x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 10:40:00 | 475.33 | 479.05 | 0.00 | T1 1.5R @ 475.33 |
| Stop hit — per-position SL triggered | 2024-11-12 11:00:00 | 477.85 | 478.80 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-11-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:45:00 | 454.90 | 453.83 | 0.00 | ORB-long ORB[448.00,453.95] vol=2.5x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-11-19 11:55:00 | 453.39 | 453.89 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:35:00 | 439.35 | 442.55 | 0.00 | ORB-short ORB[443.55,447.60] vol=1.7x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-11-21 09:40:00 | 441.85 | 442.39 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 09:35:00 | 435.35 | 438.11 | 0.00 | ORB-short ORB[436.25,441.70] vol=2.9x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-11-22 09:40:00 | 437.31 | 437.99 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:30:00 | 461.10 | 459.70 | 0.00 | ORB-long ORB[453.50,457.75] vol=8.9x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 10:40:00 | 463.96 | 459.96 | 0.00 | T1 1.5R @ 463.96 |
| Target hit | 2024-11-26 15:20:00 | 471.05 | 465.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-11-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:00:00 | 477.35 | 475.43 | 0.00 | ORB-long ORB[470.75,476.85] vol=3.6x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-11-28 10:05:00 | 475.78 | 475.49 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 484.80 | 479.85 | 0.00 | ORB-long ORB[474.00,478.00] vol=4.8x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-12-02 09:40:00 | 482.30 | 480.75 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 505.55 | 508.04 | 0.00 | ORB-short ORB[506.15,510.35] vol=1.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-12-06 09:35:00 | 507.45 | 507.80 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:05:00 | 518.90 | 523.18 | 0.00 | ORB-short ORB[526.35,532.70] vol=1.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 521.14 | 521.91 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:50:00 | 511.55 | 512.64 | 0.00 | ORB-short ORB[511.80,516.60] vol=1.8x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:05:00 | 508.32 | 511.95 | 0.00 | T1 1.5R @ 508.32 |
| Target hit | 2024-12-27 13:10:00 | 510.05 | 510.03 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2024-12-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:10:00 | 518.45 | 513.02 | 0.00 | ORB-long ORB[509.45,515.30] vol=1.9x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 516.20 | 513.30 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-01-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:45:00 | 506.00 | 508.43 | 0.00 | ORB-short ORB[506.45,511.05] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-01-01 11:45:00 | 507.54 | 506.60 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:15:00 | 496.00 | 493.11 | 0.00 | ORB-long ORB[490.45,494.00] vol=2.1x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-01-09 10:20:00 | 494.53 | 493.21 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 11:10:00 | 487.05 | 482.89 | 0.00 | ORB-long ORB[475.15,482.40] vol=2.4x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:35:00 | 490.48 | 483.78 | 0.00 | T1 1.5R @ 490.48 |
| Stop hit — per-position SL triggered | 2025-01-16 12:05:00 | 487.05 | 484.23 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:45:00 | 489.15 | 486.40 | 0.00 | ORB-long ORB[481.80,488.70] vol=1.7x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:55:00 | 492.07 | 488.06 | 0.00 | T1 1.5R @ 492.07 |
| Stop hit — per-position SL triggered | 2025-01-17 10:50:00 | 489.15 | 489.59 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:40:00 | 501.50 | 498.01 | 0.00 | ORB-long ORB[493.10,498.95] vol=3.1x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-01-20 09:45:00 | 499.36 | 498.52 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 11:10:00 | 472.45 | 465.77 | 0.00 | ORB-long ORB[459.35,464.00] vol=11.4x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 470.42 | 466.24 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-02-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:35:00 | 442.70 | 444.30 | 0.00 | ORB-short ORB[444.40,448.70] vol=3.6x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-02-04 10:40:00 | 444.06 | 444.28 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-02-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:55:00 | 446.60 | 444.18 | 0.00 | ORB-long ORB[440.95,446.25] vol=1.9x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 10:00:00 | 449.47 | 445.05 | 0.00 | T1 1.5R @ 449.47 |
| Target hit | 2025-02-06 14:00:00 | 448.85 | 448.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2025-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:40:00 | 429.80 | 425.51 | 0.00 | ORB-long ORB[422.40,427.65] vol=3.6x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-02-20 10:45:00 | 428.46 | 425.60 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 425.10 | 428.32 | 0.00 | ORB-short ORB[426.10,430.00] vol=2.3x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-02-21 09:45:00 | 426.36 | 427.92 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 402.70 | 399.73 | 0.00 | ORB-long ORB[395.20,400.50] vol=2.1x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-03-07 09:35:00 | 400.79 | 399.89 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:00:00 | 373.40 | 370.02 | 0.00 | ORB-long ORB[364.45,369.25] vol=1.6x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:10:00 | 375.40 | 370.71 | 0.00 | T1 1.5R @ 375.40 |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 373.40 | 370.72 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:35:00 | 392.50 | 390.71 | 0.00 | ORB-long ORB[387.50,391.05] vol=9.5x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 09:40:00 | 394.62 | 390.91 | 0.00 | T1 1.5R @ 394.62 |
| Stop hit — per-position SL triggered | 2025-03-20 10:05:00 | 392.50 | 391.23 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 11:15:00 | 404.00 | 401.48 | 0.00 | ORB-long ORB[399.00,403.00] vol=1.7x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 11:50:00 | 405.62 | 402.58 | 0.00 | T1 1.5R @ 405.62 |
| Stop hit — per-position SL triggered | 2025-03-26 13:00:00 | 404.00 | 403.44 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-03-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 09:30:00 | 392.25 | 394.78 | 0.00 | ORB-short ORB[393.60,397.95] vol=1.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-03-28 09:40:00 | 393.89 | 393.89 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-05-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:20:00 | 486.75 | 482.48 | 0.00 | ORB-long ORB[477.10,482.00] vol=2.3x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:30:00 | 490.74 | 484.01 | 0.00 | T1 1.5R @ 490.74 |
| Stop hit — per-position SL triggered | 2025-05-08 10:35:00 | 486.75 | 484.15 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-17 09:35:00 | 450.45 | 2024-05-17 09:45:00 | 452.70 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-05-27 10:15:00 | 438.00 | 2024-05-27 11:05:00 | 439.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-06-07 10:45:00 | 475.70 | 2024-06-07 11:50:00 | 473.62 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-06-10 09:30:00 | 475.00 | 2024-06-10 09:40:00 | 477.06 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-06-12 10:35:00 | 470.25 | 2024-06-12 10:40:00 | 472.85 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-12 10:35:00 | 470.25 | 2024-06-12 10:45:00 | 470.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-25 11:00:00 | 490.80 | 2024-06-25 11:05:00 | 488.57 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-06-26 09:50:00 | 488.50 | 2024-06-26 10:00:00 | 485.69 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-06-27 09:30:00 | 487.65 | 2024-06-27 09:45:00 | 490.42 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-27 09:30:00 | 487.65 | 2024-06-27 10:05:00 | 488.45 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2024-06-28 11:10:00 | 486.00 | 2024-06-28 11:20:00 | 484.15 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-01 09:50:00 | 486.85 | 2024-07-01 10:45:00 | 490.04 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-07-01 09:50:00 | 486.85 | 2024-07-01 11:45:00 | 491.45 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-07-03 10:50:00 | 483.55 | 2024-07-03 10:55:00 | 484.93 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-04 10:35:00 | 479.15 | 2024-07-04 11:10:00 | 480.37 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-05 11:15:00 | 493.00 | 2024-07-05 11:40:00 | 495.52 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-07-05 11:15:00 | 493.00 | 2024-07-05 15:15:00 | 527.15 | TARGET_HIT | 0.50 | 6.93% |
| BUY | retest1 | 2024-07-09 09:35:00 | 522.35 | 2024-07-09 09:45:00 | 519.54 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-07-15 09:35:00 | 533.65 | 2024-07-15 09:40:00 | 538.56 | PARTIAL | 0.50 | 0.92% |
| BUY | retest1 | 2024-07-15 09:35:00 | 533.65 | 2024-07-15 10:00:00 | 562.20 | TARGET_HIT | 0.50 | 5.35% |
| BUY | retest1 | 2024-07-16 11:05:00 | 553.25 | 2024-07-16 11:15:00 | 550.42 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-07-18 09:30:00 | 552.70 | 2024-07-18 09:35:00 | 548.13 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2024-07-18 09:30:00 | 552.70 | 2024-07-18 09:40:00 | 552.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 10:30:00 | 538.45 | 2024-07-23 10:45:00 | 536.21 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-09 09:30:00 | 511.50 | 2024-08-09 09:50:00 | 508.05 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-08-09 09:30:00 | 511.50 | 2024-08-09 11:15:00 | 511.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 09:30:00 | 487.30 | 2024-08-14 09:45:00 | 489.74 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-08-16 09:35:00 | 489.45 | 2024-08-16 09:45:00 | 491.31 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-29 10:45:00 | 504.55 | 2024-08-29 10:50:00 | 502.69 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-29 10:45:00 | 504.55 | 2024-08-29 12:20:00 | 504.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 10:10:00 | 498.95 | 2024-09-10 10:15:00 | 497.24 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-11 09:45:00 | 505.00 | 2024-09-11 10:00:00 | 503.16 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-12 09:30:00 | 497.50 | 2024-09-12 10:10:00 | 494.56 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-12 09:30:00 | 497.50 | 2024-09-12 10:25:00 | 497.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:00:00 | 486.90 | 2024-09-19 10:25:00 | 488.53 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-25 09:45:00 | 473.10 | 2024-09-25 13:10:00 | 471.05 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-25 09:45:00 | 473.10 | 2024-09-25 15:20:00 | 468.90 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2024-10-03 09:35:00 | 471.15 | 2024-10-03 10:15:00 | 469.09 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-07 09:50:00 | 460.70 | 2024-10-07 09:55:00 | 458.29 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-07 09:50:00 | 460.70 | 2024-10-07 15:20:00 | 443.45 | TARGET_HIT | 0.50 | 3.74% |
| SELL | retest1 | 2024-11-12 10:20:00 | 477.85 | 2024-11-12 10:40:00 | 475.33 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-11-12 10:20:00 | 477.85 | 2024-11-12 11:00:00 | 477.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:45:00 | 454.90 | 2024-11-19 11:55:00 | 453.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-11-21 09:35:00 | 439.35 | 2024-11-21 09:40:00 | 441.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-11-22 09:35:00 | 435.35 | 2024-11-22 09:40:00 | 437.31 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-11-26 10:30:00 | 461.10 | 2024-11-26 10:40:00 | 463.96 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-11-26 10:30:00 | 461.10 | 2024-11-26 15:20:00 | 471.05 | TARGET_HIT | 0.50 | 2.16% |
| BUY | retest1 | 2024-11-28 10:00:00 | 477.35 | 2024-11-28 10:05:00 | 475.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-02 09:30:00 | 484.80 | 2024-12-02 09:40:00 | 482.30 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-12-06 09:30:00 | 505.55 | 2024-12-06 09:35:00 | 507.45 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-12-26 10:05:00 | 518.90 | 2024-12-26 11:00:00 | 521.14 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-27 09:50:00 | 511.55 | 2024-12-27 10:05:00 | 508.32 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-12-27 09:50:00 | 511.55 | 2024-12-27 13:10:00 | 510.05 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-12-30 10:10:00 | 518.45 | 2024-12-30 10:15:00 | 516.20 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-01-01 10:45:00 | 506.00 | 2025-01-01 11:45:00 | 507.54 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-09 10:15:00 | 496.00 | 2025-01-09 10:20:00 | 494.53 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-16 11:10:00 | 487.05 | 2025-01-16 11:35:00 | 490.48 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-01-16 11:10:00 | 487.05 | 2025-01-16 12:05:00 | 487.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 09:45:00 | 489.15 | 2025-01-17 09:55:00 | 492.07 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-17 09:45:00 | 489.15 | 2025-01-17 10:50:00 | 489.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-20 09:40:00 | 501.50 | 2025-01-20 09:45:00 | 499.36 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-02-01 11:10:00 | 472.45 | 2025-02-01 11:15:00 | 470.42 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-02-04 10:35:00 | 442.70 | 2025-02-04 10:40:00 | 444.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-06 09:55:00 | 446.60 | 2025-02-06 10:00:00 | 449.47 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-02-06 09:55:00 | 446.60 | 2025-02-06 14:00:00 | 448.85 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-02-20 10:40:00 | 429.80 | 2025-02-20 10:45:00 | 428.46 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-02-21 09:40:00 | 425.10 | 2025-02-21 09:45:00 | 426.36 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-07 09:30:00 | 402.70 | 2025-03-07 09:35:00 | 400.79 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-03-18 11:00:00 | 373.40 | 2025-03-18 11:10:00 | 375.40 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-03-18 11:00:00 | 373.40 | 2025-03-18 11:15:00 | 373.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-20 09:35:00 | 392.50 | 2025-03-20 09:40:00 | 394.62 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-03-20 09:35:00 | 392.50 | 2025-03-20 10:05:00 | 392.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-26 11:15:00 | 404.00 | 2025-03-26 11:50:00 | 405.62 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-26 11:15:00 | 404.00 | 2025-03-26 13:00:00 | 404.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-28 09:30:00 | 392.25 | 2025-03-28 09:40:00 | 393.89 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-05-08 10:20:00 | 486.75 | 2025-05-08 10:30:00 | 490.74 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2025-05-08 10:20:00 | 486.75 | 2025-05-08 10:35:00 | 486.75 | STOP_HIT | 0.50 | 0.00% |
