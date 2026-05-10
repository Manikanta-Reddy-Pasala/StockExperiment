# Jubilant Foodworks Ltd. (JUBLFOOD)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 473.00
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 17
- **Target hits / Stop hits / Partials:** 6 / 17 / 10
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 4.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | -0.03% | -0.2% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | -0.03% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 13 | 52.0% | 5 | 12 | 8 | 0.20% | 5.1% |
| SELL @ 2nd Alert (retest1) | 25 | 13 | 52.0% | 5 | 12 | 8 | 0.20% | 5.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 16 | 48.5% | 6 | 17 | 10 | 0.15% | 4.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:10:00 | 552.45 | 550.50 | 0.00 | ORB-long ORB[547.35,550.85] vol=1.8x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:20:00 | 554.49 | 551.20 | 0.00 | T1 1.5R @ 554.49 |
| Target hit | 2026-02-10 13:15:00 | 554.00 | 554.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:10:00 | 538.15 | 538.40 | 0.00 | ORB-short ORB[538.20,545.35] vol=1.6x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 12:00:00 | 535.58 | 538.31 | 0.00 | T1 1.5R @ 535.58 |
| Target hit | 2026-02-12 14:10:00 | 537.80 | 537.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 535.70 | 533.03 | 0.00 | ORB-long ORB[530.70,533.65] vol=4.5x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 534.36 | 533.28 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 529.00 | 529.75 | 0.00 | ORB-short ORB[529.10,533.40] vol=5.3x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:15:00 | 527.36 | 529.64 | 0.00 | T1 1.5R @ 527.36 |
| Target hit | 2026-02-18 15:20:00 | 523.65 | 526.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:20:00 | 527.50 | 531.11 | 0.00 | ORB-short ORB[529.55,535.05] vol=3.8x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:15:00 | 524.46 | 529.57 | 0.00 | T1 1.5R @ 524.46 |
| Target hit | 2026-02-24 15:15:00 | 519.80 | 519.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:55:00 | 514.10 | 515.03 | 0.00 | ORB-short ORB[518.20,525.55] vol=1.5x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-02-25 12:20:00 | 515.87 | 514.68 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:15:00 | 510.15 | 511.55 | 0.00 | ORB-short ORB[511.70,517.20] vol=2.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:30:00 | 508.66 | 511.36 | 0.00 | T1 1.5R @ 508.66 |
| Stop hit — per-position SL triggered | 2026-02-27 11:40:00 | 510.15 | 511.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:25:00 | 509.60 | 501.53 | 0.00 | ORB-long ORB[495.00,501.30] vol=1.8x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-03-04 10:35:00 | 507.36 | 502.69 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 497.25 | 499.49 | 0.00 | ORB-short ORB[500.00,506.90] vol=1.6x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 495.23 | 499.09 | 0.00 | T1 1.5R @ 495.23 |
| Stop hit — per-position SL triggered | 2026-03-05 12:40:00 | 497.25 | 497.73 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 494.00 | 497.73 | 0.00 | ORB-short ORB[498.05,503.00] vol=1.9x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-03-06 11:20:00 | 495.25 | 496.72 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:35:00 | 477.10 | 479.08 | 0.00 | ORB-short ORB[478.15,483.15] vol=1.5x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-03-11 09:45:00 | 478.68 | 478.69 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:10:00 | 457.75 | 460.33 | 0.00 | ORB-short ORB[460.50,467.05] vol=2.6x ATR=1.66 |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 459.41 | 460.26 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:40:00 | 462.10 | 465.25 | 0.00 | ORB-short ORB[464.00,468.75] vol=1.6x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 463.96 | 464.94 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:50:00 | 461.95 | 464.15 | 0.00 | ORB-short ORB[463.20,469.80] vol=1.8x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:25:00 | 459.02 | 462.27 | 0.00 | T1 1.5R @ 459.02 |
| Target hit | 2026-03-19 15:20:00 | 455.95 | 458.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2026-03-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:55:00 | 453.50 | 458.84 | 0.00 | ORB-short ORB[457.95,464.50] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-03-27 11:00:00 | 454.87 | 458.30 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:30:00 | 434.95 | 432.04 | 0.00 | ORB-long ORB[430.00,433.95] vol=2.7x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-04-10 10:45:00 | 433.54 | 432.31 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 463.40 | 459.17 | 0.00 | ORB-long ORB[453.60,459.75] vol=1.9x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 465.82 | 460.20 | 0.00 | T1 1.5R @ 465.82 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 463.40 | 463.10 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 481.30 | 477.07 | 0.00 | ORB-long ORB[473.30,479.35] vol=2.1x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-04-22 09:55:00 | 479.34 | 478.27 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 489.60 | 491.62 | 0.00 | ORB-short ORB[490.00,496.40] vol=3.1x ATR=2.12 |
| Stop hit — per-position SL triggered | 2026-04-24 09:40:00 | 491.72 | 491.69 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:45:00 | 478.20 | 479.43 | 0.00 | ORB-short ORB[479.70,485.25] vol=8.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 479.61 | 479.31 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-04-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:05:00 | 479.80 | 481.58 | 0.00 | ORB-short ORB[480.20,486.25] vol=2.7x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-04-29 12:00:00 | 482.25 | 480.67 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:55:00 | 473.80 | 478.23 | 0.00 | ORB-short ORB[477.85,483.75] vol=2.0x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:00:00 | 471.63 | 477.75 | 0.00 | T1 1.5R @ 471.63 |
| Stop hit — per-position SL triggered | 2026-04-30 11:20:00 | 473.80 | 476.67 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 469.50 | 473.76 | 0.00 | ORB-short ORB[474.00,476.40] vol=1.5x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:35:00 | 467.87 | 473.05 | 0.00 | T1 1.5R @ 467.87 |
| Target hit | 2026-05-05 15:20:00 | 465.00 | 468.32 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:10:00 | 552.45 | 2026-02-10 10:20:00 | 554.49 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-10 10:10:00 | 552.45 | 2026-02-10 13:15:00 | 554.00 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-12 11:10:00 | 538.15 | 2026-02-12 12:00:00 | 535.58 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-12 11:10:00 | 538.15 | 2026-02-12 14:10:00 | 537.80 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2026-02-17 10:35:00 | 535.70 | 2026-02-17 10:45:00 | 534.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-18 11:05:00 | 529.00 | 2026-02-18 11:15:00 | 527.36 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 11:05:00 | 529.00 | 2026-02-18 15:20:00 | 523.65 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-02-24 10:20:00 | 527.50 | 2026-02-24 11:15:00 | 524.46 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-24 10:20:00 | 527.50 | 2026-02-24 15:15:00 | 519.80 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2026-02-25 10:55:00 | 514.10 | 2026-02-25 12:20:00 | 515.87 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-27 11:15:00 | 510.15 | 2026-02-27 11:30:00 | 508.66 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-02-27 11:15:00 | 510.15 | 2026-02-27 11:40:00 | 510.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-04 10:25:00 | 509.60 | 2026-03-04 10:35:00 | 507.36 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-05 11:05:00 | 497.25 | 2026-03-05 11:25:00 | 495.23 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-05 11:05:00 | 497.25 | 2026-03-05 12:40:00 | 497.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 494.00 | 2026-03-06 11:20:00 | 495.25 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-11 09:35:00 | 477.10 | 2026-03-11 09:45:00 | 478.68 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-13 11:10:00 | 457.75 | 2026-03-13 11:15:00 | 459.41 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-03-17 10:40:00 | 462.10 | 2026-03-17 11:15:00 | 463.96 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-19 09:50:00 | 461.95 | 2026-03-19 11:25:00 | 459.02 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-19 09:50:00 | 461.95 | 2026-03-19 15:20:00 | 455.95 | TARGET_HIT | 0.50 | 1.30% |
| SELL | retest1 | 2026-03-27 10:55:00 | 453.50 | 2026-03-27 11:00:00 | 454.87 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-10 10:30:00 | 434.95 | 2026-04-10 10:45:00 | 433.54 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-21 10:00:00 | 463.40 | 2026-04-21 10:05:00 | 465.82 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-21 10:00:00 | 463.40 | 2026-04-21 11:00:00 | 463.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:40:00 | 481.30 | 2026-04-22 09:55:00 | 479.34 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-24 09:35:00 | 489.60 | 2026-04-24 09:40:00 | 491.72 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-28 10:45:00 | 478.20 | 2026-04-28 11:15:00 | 479.61 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-29 10:05:00 | 479.80 | 2026-04-29 12:00:00 | 482.25 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-04-30 10:55:00 | 473.80 | 2026-04-30 11:00:00 | 471.63 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-30 10:55:00 | 473.80 | 2026-04-30 11:20:00 | 473.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:55:00 | 469.50 | 2026-05-05 11:35:00 | 467.87 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-05-05 10:55:00 | 469.50 | 2026-05-05 15:20:00 | 465.00 | TARGET_HIT | 0.50 | 0.96% |
