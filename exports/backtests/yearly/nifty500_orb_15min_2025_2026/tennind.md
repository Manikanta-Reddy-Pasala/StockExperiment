# Tenneco Clean Air India Ltd. (TENNIND)

## Backtest Summary

- **Window:** 2025-11-19 10:00:00 → 2026-05-08 15:25:00 (8616 bars)
- **Last close:** 640.20
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
| ENTRY1 | 27 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 26
- **Target hits / Stop hits / Partials:** 1 / 26 / 6
- **Avg / median % per leg:** -0.14% / -0.35%
- **Sum % (uncompounded):** -4.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 0 | 12 | 2 | -0.17% | -2.4% |
| BUY @ 2nd Alert (retest1) | 14 | 2 | 14.3% | 0 | 12 | 2 | -0.17% | -2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 5 | 26.3% | 1 | 14 | 4 | -0.13% | -2.4% |
| SELL @ 2nd Alert (retest1) | 19 | 5 | 26.3% | 1 | 14 | 4 | -0.13% | -2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 7 | 21.2% | 1 | 26 | 6 | -0.14% | -4.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:30:00 | 502.20 | 496.78 | 0.00 | ORB-long ORB[489.85,495.70] vol=2.3x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-11-27 10:40:00 | 499.20 | 497.23 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 495.20 | 492.93 | 0.00 | ORB-long ORB[490.05,493.95] vol=2.7x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-11-28 09:40:00 | 493.69 | 493.63 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-12-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:55:00 | 470.10 | 473.60 | 0.00 | ORB-short ORB[473.10,478.15] vol=2.0x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:05:00 | 468.29 | 473.21 | 0.00 | T1 1.5R @ 468.29 |
| Stop hit — per-position SL triggered | 2025-12-02 13:35:00 | 470.10 | 470.41 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 09:30:00 | 496.60 | 493.23 | 0.00 | ORB-long ORB[488.60,494.85] vol=3.9x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-12-05 09:35:00 | 494.84 | 493.30 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-12-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:05:00 | 487.80 | 492.16 | 0.00 | ORB-short ORB[491.70,498.75] vol=1.5x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-12-10 10:45:00 | 490.13 | 490.93 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:30:00 | 471.75 | 474.85 | 0.00 | ORB-short ORB[473.80,480.00] vol=1.7x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-12-11 09:50:00 | 474.07 | 474.09 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-12-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:30:00 | 483.00 | 477.74 | 0.00 | ORB-long ORB[474.15,477.95] vol=3.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-12-23 10:35:00 | 481.26 | 478.03 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-12-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:50:00 | 509.95 | 505.76 | 0.00 | ORB-long ORB[501.00,507.95] vol=2.8x ATR=2.49 |
| Stop hit — per-position SL triggered | 2025-12-30 11:55:00 | 507.46 | 507.82 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-01-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:20:00 | 525.50 | 535.65 | 0.00 | ORB-short ORB[540.25,547.80] vol=2.2x ATR=2.89 |
| Stop hit — per-position SL triggered | 2026-01-06 11:40:00 | 528.39 | 530.63 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 11:15:00 | 520.60 | 525.49 | 0.00 | ORB-short ORB[524.10,531.85] vol=1.5x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-01-09 11:30:00 | 522.66 | 525.21 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-01-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 09:55:00 | 537.35 | 535.42 | 0.00 | ORB-long ORB[530.45,535.85] vol=2.4x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-01-14 10:05:00 | 535.22 | 534.99 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 560.00 | 563.67 | 0.00 | ORB-short ORB[563.00,568.75] vol=2.7x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-02-10 10:00:00 | 562.09 | 563.25 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 568.10 | 571.30 | 0.00 | ORB-short ORB[569.25,576.45] vol=1.7x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 570.65 | 571.04 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 547.95 | 549.92 | 0.00 | ORB-short ORB[549.20,555.40] vol=3.9x ATR=2.40 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 550.35 | 549.93 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-02-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:25:00 | 548.40 | 552.10 | 0.00 | ORB-short ORB[551.20,556.85] vol=2.1x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:35:00 | 545.69 | 551.06 | 0.00 | T1 1.5R @ 545.69 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 548.40 | 550.58 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 549.00 | 545.12 | 0.00 | ORB-long ORB[540.00,547.35] vol=2.1x ATR=1.91 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 547.09 | 545.46 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-02-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:05:00 | 569.00 | 573.96 | 0.00 | ORB-short ORB[571.00,578.75] vol=1.8x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:10:00 | 566.24 | 573.52 | 0.00 | T1 1.5R @ 566.24 |
| Stop hit — per-position SL triggered | 2026-02-24 11:30:00 | 569.00 | 572.90 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 599.00 | 595.07 | 0.00 | ORB-long ORB[592.45,598.60] vol=3.3x ATR=2.08 |
| Stop hit — per-position SL triggered | 2026-02-27 10:55:00 | 596.92 | 595.24 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 545.75 | 550.81 | 0.00 | ORB-short ORB[549.25,556.85] vol=3.9x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:15:00 | 543.07 | 548.55 | 0.00 | T1 1.5R @ 543.07 |
| Target hit | 2026-03-05 14:50:00 | 541.35 | 538.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — SELL (started 2026-03-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:20:00 | 524.40 | 525.04 | 0.00 | ORB-short ORB[525.75,532.80] vol=3.1x ATR=3.29 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 527.69 | 525.18 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 525.00 | 531.77 | 0.00 | ORB-short ORB[529.85,537.00] vol=1.6x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-03-17 10:45:00 | 527.35 | 531.30 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-04-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:40:00 | 520.35 | 523.16 | 0.00 | ORB-short ORB[521.60,528.80] vol=2.7x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-04-07 10:00:00 | 522.85 | 522.95 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-04-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:40:00 | 570.15 | 568.30 | 0.00 | ORB-long ORB[557.00,565.25] vol=1.9x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 11:45:00 | 575.48 | 569.13 | 0.00 | T1 1.5R @ 575.48 |
| Stop hit — per-position SL triggered | 2026-04-08 13:00:00 | 570.15 | 570.28 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 564.80 | 561.79 | 0.00 | ORB-long ORB[556.20,564.45] vol=1.6x ATR=2.38 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 562.42 | 562.36 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 600.45 | 597.91 | 0.00 | ORB-long ORB[595.00,599.65] vol=2.5x ATR=2.12 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 598.33 | 598.05 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2026-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:45:00 | 557.90 | 565.42 | 0.00 | ORB-short ORB[566.80,573.05] vol=2.7x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-04-24 10:55:00 | 559.62 | 564.47 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2026-04-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:35:00 | 565.30 | 561.94 | 0.00 | ORB-long ORB[555.95,563.65] vol=1.6x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 568.86 | 563.41 | 0.00 | T1 1.5R @ 568.86 |
| Stop hit — per-position SL triggered | 2026-04-27 09:50:00 | 565.30 | 563.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-27 10:30:00 | 502.20 | 2025-11-27 10:40:00 | 499.20 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2025-11-28 09:30:00 | 495.20 | 2025-11-28 09:40:00 | 493.69 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-02 10:55:00 | 470.10 | 2025-12-02 11:05:00 | 468.29 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-12-02 10:55:00 | 470.10 | 2025-12-02 13:35:00 | 470.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 09:30:00 | 496.60 | 2025-12-05 09:35:00 | 494.84 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-10 10:05:00 | 487.80 | 2025-12-10 10:45:00 | 490.13 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-12-11 09:30:00 | 471.75 | 2025-12-11 09:50:00 | 474.07 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-12-23 10:30:00 | 483.00 | 2025-12-23 10:35:00 | 481.26 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-12-30 09:50:00 | 509.95 | 2025-12-30 11:55:00 | 507.46 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-01-06 10:20:00 | 525.50 | 2026-01-06 11:40:00 | 528.39 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2026-01-09 11:15:00 | 520.60 | 2026-01-09 11:30:00 | 522.66 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-14 09:55:00 | 537.35 | 2026-01-14 10:05:00 | 535.22 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-10 09:45:00 | 560.00 | 2026-02-10 10:00:00 | 562.09 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-13 09:30:00 | 568.10 | 2026-02-13 09:40:00 | 570.65 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-02-18 10:50:00 | 547.95 | 2026-02-18 10:55:00 | 550.35 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-19 10:25:00 | 548.40 | 2026-02-19 10:35:00 | 545.69 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-19 10:25:00 | 548.40 | 2026-02-19 10:40:00 | 548.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:40:00 | 549.00 | 2026-02-20 11:05:00 | 547.09 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-24 11:05:00 | 569.00 | 2026-02-24 11:10:00 | 566.24 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-24 11:05:00 | 569.00 | 2026-02-24 11:30:00 | 569.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 10:45:00 | 599.00 | 2026-02-27 10:55:00 | 596.92 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-05 11:00:00 | 545.75 | 2026-03-05 11:15:00 | 543.07 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-05 11:00:00 | 545.75 | 2026-03-05 14:50:00 | 541.35 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2026-03-13 10:20:00 | 524.40 | 2026-03-13 10:30:00 | 527.69 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2026-03-17 10:35:00 | 525.00 | 2026-03-17 10:45:00 | 527.35 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-07 09:40:00 | 520.35 | 2026-04-07 10:00:00 | 522.85 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-08 10:40:00 | 570.15 | 2026-04-08 11:45:00 | 575.48 | PARTIAL | 0.50 | 0.93% |
| BUY | retest1 | 2026-04-08 10:40:00 | 570.15 | 2026-04-08 13:00:00 | 570.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:45:00 | 564.80 | 2026-04-10 10:05:00 | 562.42 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-22 09:40:00 | 600.45 | 2026-04-22 09:45:00 | 598.33 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-24 10:45:00 | 557.90 | 2026-04-24 10:55:00 | 559.62 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-27 09:35:00 | 565.30 | 2026-04-27 09:45:00 | 568.86 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-27 09:35:00 | 565.30 | 2026-04-27 09:50:00 | 565.30 | STOP_HIT | 0.50 | 0.00% |
