# Blue Jet Healthcare Ltd. (BLUEJET)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-04-04 15:25:00 (16833 bars)
- **Last close:** 726.20
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
| ENTRY1 | 44 |
| ENTRY2 | 0 |
| PARTIAL | 15 |
| TARGET_HIT | 7 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 37
- **Target hits / Stop hits / Partials:** 7 / 37 / 15
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 14.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 7 | 26.9% | 2 | 19 | 5 | -0.09% | -2.3% |
| BUY @ 2nd Alert (retest1) | 26 | 7 | 26.9% | 2 | 19 | 5 | -0.09% | -2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 15 | 45.5% | 5 | 18 | 10 | 0.52% | 17.0% |
| SELL @ 2nd Alert (retest1) | 33 | 15 | 45.5% | 5 | 18 | 10 | 0.52% | 17.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 59 | 22 | 37.3% | 7 | 37 | 15 | 0.25% | 14.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:30:00 | 368.90 | 371.78 | 0.00 | ORB-short ORB[371.20,376.00] vol=3.4x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-05-15 10:55:00 | 369.91 | 371.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:55:00 | 373.55 | 374.87 | 0.00 | ORB-short ORB[376.00,380.60] vol=6.2x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 10:30:00 | 371.16 | 374.12 | 0.00 | T1 1.5R @ 371.16 |
| Stop hit — per-position SL triggered | 2024-05-24 11:20:00 | 373.55 | 373.75 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 11:05:00 | 366.70 | 368.95 | 0.00 | ORB-short ORB[367.40,372.65] vol=1.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-05-29 12:20:00 | 367.51 | 368.30 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:40:00 | 369.95 | 371.12 | 0.00 | ORB-short ORB[370.00,374.45] vol=2.9x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-05-30 09:45:00 | 371.22 | 371.67 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 376.00 | 372.81 | 0.00 | ORB-long ORB[370.00,373.95] vol=2.2x ATR=2.11 |
| Stop hit — per-position SL triggered | 2024-06-07 10:10:00 | 373.89 | 373.66 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:05:00 | 389.65 | 387.94 | 0.00 | ORB-long ORB[384.00,387.90] vol=3.6x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:10:00 | 392.32 | 388.46 | 0.00 | T1 1.5R @ 392.32 |
| Stop hit — per-position SL triggered | 2024-06-11 10:30:00 | 389.65 | 388.69 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:45:00 | 386.35 | 383.49 | 0.00 | ORB-long ORB[377.15,382.00] vol=2.0x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:15:00 | 388.66 | 384.25 | 0.00 | T1 1.5R @ 388.66 |
| Stop hit — per-position SL triggered | 2024-06-12 11:20:00 | 386.35 | 384.74 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 10:20:00 | 396.55 | 393.45 | 0.00 | ORB-long ORB[390.10,394.00] vol=3.0x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-06-19 10:35:00 | 394.37 | 393.64 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:00:00 | 416.00 | 409.58 | 0.00 | ORB-long ORB[405.25,409.70] vol=4.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-06-26 11:05:00 | 414.56 | 410.21 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 417.20 | 419.46 | 0.00 | ORB-short ORB[418.30,422.30] vol=2.0x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-07-02 09:45:00 | 419.82 | 419.41 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:20:00 | 411.75 | 413.15 | 0.00 | ORB-short ORB[412.00,416.40] vol=2.1x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-07-04 10:30:00 | 413.31 | 413.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:35:00 | 444.60 | 439.47 | 0.00 | ORB-long ORB[433.10,438.70] vol=8.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-07-09 10:40:00 | 441.87 | 439.74 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 432.25 | 434.12 | 0.00 | ORB-short ORB[433.00,438.00] vol=1.8x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 427.98 | 432.40 | 0.00 | T1 1.5R @ 427.98 |
| Target hit | 2024-07-10 10:45:00 | 420.25 | 420.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 409.80 | 413.31 | 0.00 | ORB-short ORB[411.80,416.90] vol=1.8x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 411.30 | 413.16 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:10:00 | 451.65 | 449.30 | 0.00 | ORB-long ORB[443.35,448.00] vol=4.6x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 10:50:00 | 454.84 | 451.16 | 0.00 | T1 1.5R @ 454.84 |
| Target hit | 2024-07-30 12:40:00 | 457.75 | 460.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2024-07-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:50:00 | 462.05 | 459.45 | 0.00 | ORB-long ORB[454.10,459.20] vol=3.1x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 10:10:00 | 466.93 | 464.42 | 0.00 | T1 1.5R @ 466.93 |
| Target hit | 2024-07-31 10:30:00 | 463.90 | 467.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2024-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:50:00 | 461.25 | 462.42 | 0.00 | ORB-short ORB[462.15,467.00] vol=3.8x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 12:05:00 | 457.41 | 461.90 | 0.00 | T1 1.5R @ 457.41 |
| Stop hit — per-position SL triggered | 2024-08-14 12:20:00 | 461.25 | 461.83 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:40:00 | 459.30 | 462.38 | 0.00 | ORB-short ORB[461.75,465.70] vol=1.6x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-08-16 09:45:00 | 461.46 | 462.28 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:55:00 | 481.85 | 479.20 | 0.00 | ORB-long ORB[474.35,480.65] vol=3.8x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-08-20 10:00:00 | 479.29 | 479.27 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:15:00 | 473.40 | 474.01 | 0.00 | ORB-short ORB[473.95,478.35] vol=1.9x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-08-23 10:40:00 | 475.67 | 474.21 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 480.80 | 478.93 | 0.00 | ORB-long ORB[473.85,480.45] vol=2.9x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-08-26 09:35:00 | 478.06 | 478.93 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:45:00 | 487.25 | 484.22 | 0.00 | ORB-long ORB[480.00,482.05] vol=6.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-08-27 09:50:00 | 485.69 | 484.49 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 10:00:00 | 484.85 | 487.51 | 0.00 | ORB-short ORB[487.55,493.00] vol=1.6x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:35:00 | 481.85 | 485.98 | 0.00 | T1 1.5R @ 481.85 |
| Target hit | 2024-09-02 15:20:00 | 466.20 | 468.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-09-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:25:00 | 483.65 | 480.19 | 0.00 | ORB-long ORB[474.75,481.15] vol=6.3x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 10:30:00 | 487.46 | 481.20 | 0.00 | T1 1.5R @ 487.46 |
| Stop hit — per-position SL triggered | 2024-09-05 10:40:00 | 483.65 | 481.88 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 11:00:00 | 495.60 | 486.57 | 0.00 | ORB-long ORB[482.65,489.90] vol=2.4x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-09-06 11:05:00 | 492.51 | 488.19 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 09:35:00 | 518.30 | 521.37 | 0.00 | ORB-short ORB[518.55,525.65] vol=2.1x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:10:00 | 513.27 | 518.76 | 0.00 | T1 1.5R @ 513.27 |
| Target hit | 2024-09-10 15:20:00 | 499.30 | 507.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-09-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:50:00 | 521.55 | 518.73 | 0.00 | ORB-long ORB[515.10,520.50] vol=2.2x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-09-20 10:55:00 | 519.39 | 518.75 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:45:00 | 528.00 | 524.02 | 0.00 | ORB-long ORB[520.75,524.90] vol=4.0x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-09-23 11:00:00 | 526.06 | 524.08 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:35:00 | 539.00 | 531.63 | 0.00 | ORB-long ORB[524.10,529.45] vol=8.5x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-09-24 10:40:00 | 535.80 | 532.17 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:00:00 | 516.35 | 518.70 | 0.00 | ORB-short ORB[519.35,523.70] vol=1.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-09-27 11:40:00 | 518.28 | 518.30 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:30:00 | 489.40 | 492.21 | 0.00 | ORB-short ORB[489.90,496.95] vol=3.3x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:35:00 | 486.86 | 491.19 | 0.00 | T1 1.5R @ 486.86 |
| Stop hit — per-position SL triggered | 2024-10-07 10:40:00 | 489.40 | 491.06 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:40:00 | 506.35 | 504.15 | 0.00 | ORB-long ORB[499.80,503.10] vol=3.4x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-10-11 09:45:00 | 504.07 | 504.55 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:00:00 | 508.45 | 514.81 | 0.00 | ORB-short ORB[515.25,522.20] vol=2.0x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-10-14 11:15:00 | 510.52 | 514.24 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 10:25:00 | 539.10 | 534.79 | 0.00 | ORB-long ORB[526.55,534.45] vol=5.3x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-11-12 10:30:00 | 536.01 | 534.83 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 528.50 | 533.56 | 0.00 | ORB-short ORB[534.30,540.45] vol=1.8x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-11-13 09:35:00 | 531.86 | 533.14 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:45:00 | 530.35 | 528.16 | 0.00 | ORB-long ORB[523.60,528.90] vol=3.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-11-28 10:05:00 | 528.07 | 528.63 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 09:50:00 | 522.25 | 524.27 | 0.00 | ORB-short ORB[525.75,531.10] vol=5.8x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:05:00 | 519.44 | 521.32 | 0.00 | T1 1.5R @ 519.44 |
| Target hit | 2024-12-04 15:20:00 | 507.50 | 514.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2024-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 11:00:00 | 499.85 | 502.18 | 0.00 | ORB-short ORB[500.45,505.70] vol=2.0x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-12-10 11:30:00 | 501.84 | 501.98 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:35:00 | 496.65 | 498.88 | 0.00 | ORB-short ORB[498.30,503.35] vol=2.8x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 10:50:00 | 492.98 | 497.09 | 0.00 | T1 1.5R @ 492.98 |
| Stop hit — per-position SL triggered | 2024-12-16 11:25:00 | 496.65 | 496.31 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 11:00:00 | 565.30 | 557.19 | 0.00 | ORB-long ORB[554.00,561.60] vol=3.5x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 562.18 | 557.99 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:30:00 | 584.60 | 581.93 | 0.00 | ORB-long ORB[576.25,583.90] vol=5.0x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-01-02 11:40:00 | 582.50 | 582.72 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:50:00 | 591.75 | 595.45 | 0.00 | ORB-short ORB[593.55,599.30] vol=1.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-01-09 11:05:00 | 593.72 | 595.30 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:00:00 | 592.50 | 595.48 | 0.00 | ORB-short ORB[594.05,600.00] vol=2.4x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:30:00 | 588.77 | 593.41 | 0.00 | T1 1.5R @ 588.77 |
| Stop hit — per-position SL triggered | 2025-01-21 12:00:00 | 592.50 | 591.96 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:30:00 | 874.45 | 881.62 | 0.00 | ORB-short ORB[881.60,892.05] vol=2.6x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:40:00 | 866.90 | 875.07 | 0.00 | T1 1.5R @ 866.90 |
| Target hit | 2025-03-26 15:20:00 | 857.85 | 862.34 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:30:00 | 368.90 | 2024-05-15 10:55:00 | 369.91 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-24 09:55:00 | 373.55 | 2024-05-24 10:30:00 | 371.16 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-05-24 09:55:00 | 373.55 | 2024-05-24 11:20:00 | 373.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-29 11:05:00 | 366.70 | 2024-05-29 12:20:00 | 367.51 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-30 09:40:00 | 369.95 | 2024-05-30 09:45:00 | 371.22 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-07 09:30:00 | 376.00 | 2024-06-07 10:10:00 | 373.89 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-06-11 10:05:00 | 389.65 | 2024-06-11 10:10:00 | 392.32 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-06-11 10:05:00 | 389.65 | 2024-06-11 10:30:00 | 389.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 10:45:00 | 386.35 | 2024-06-12 11:15:00 | 388.66 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-06-12 10:45:00 | 386.35 | 2024-06-12 11:20:00 | 386.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-19 10:20:00 | 396.55 | 2024-06-19 10:35:00 | 394.37 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-06-26 11:00:00 | 416.00 | 2024-06-26 11:05:00 | 414.56 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-02 09:30:00 | 417.20 | 2024-07-02 09:45:00 | 419.82 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2024-07-04 10:20:00 | 411.75 | 2024-07-04 10:30:00 | 413.31 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-09 10:35:00 | 444.60 | 2024-07-09 10:40:00 | 441.87 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2024-07-10 10:10:00 | 432.25 | 2024-07-10 10:20:00 | 427.98 | PARTIAL | 0.50 | 0.99% |
| SELL | retest1 | 2024-07-10 10:10:00 | 432.25 | 2024-07-10 10:45:00 | 420.25 | TARGET_HIT | 0.50 | 2.78% |
| SELL | retest1 | 2024-07-18 09:35:00 | 409.80 | 2024-07-18 09:40:00 | 411.30 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-30 10:10:00 | 451.65 | 2024-07-30 10:50:00 | 454.84 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-07-30 10:10:00 | 451.65 | 2024-07-30 12:40:00 | 457.75 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2024-07-31 09:50:00 | 462.05 | 2024-07-31 10:10:00 | 466.93 | PARTIAL | 0.50 | 1.06% |
| BUY | retest1 | 2024-07-31 09:50:00 | 462.05 | 2024-07-31 10:30:00 | 463.90 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-08-14 10:50:00 | 461.25 | 2024-08-14 12:05:00 | 457.41 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2024-08-14 10:50:00 | 461.25 | 2024-08-14 12:20:00 | 461.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-16 09:40:00 | 459.30 | 2024-08-16 09:45:00 | 461.46 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-20 09:55:00 | 481.85 | 2024-08-20 10:00:00 | 479.29 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-08-23 10:15:00 | 473.40 | 2024-08-23 10:40:00 | 475.67 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-08-26 09:30:00 | 480.80 | 2024-08-26 09:35:00 | 478.06 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-08-27 09:45:00 | 487.25 | 2024-08-27 09:50:00 | 485.69 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-02 10:00:00 | 484.85 | 2024-09-02 10:35:00 | 481.85 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-09-02 10:00:00 | 484.85 | 2024-09-02 15:20:00 | 466.20 | TARGET_HIT | 0.50 | 3.85% |
| BUY | retest1 | 2024-09-05 10:25:00 | 483.65 | 2024-09-05 10:30:00 | 487.46 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-09-05 10:25:00 | 483.65 | 2024-09-05 10:40:00 | 483.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-06 11:00:00 | 495.60 | 2024-09-06 11:05:00 | 492.51 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2024-09-10 09:35:00 | 518.30 | 2024-09-10 10:10:00 | 513.27 | PARTIAL | 0.50 | 0.97% |
| SELL | retest1 | 2024-09-10 09:35:00 | 518.30 | 2024-09-10 15:20:00 | 499.30 | TARGET_HIT | 0.50 | 3.67% |
| BUY | retest1 | 2024-09-20 10:50:00 | 521.55 | 2024-09-20 10:55:00 | 519.39 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-09-23 10:45:00 | 528.00 | 2024-09-23 11:00:00 | 526.06 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-24 10:35:00 | 539.00 | 2024-09-24 10:40:00 | 535.80 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-09-27 11:00:00 | 516.35 | 2024-09-27 11:40:00 | 518.28 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-07 10:30:00 | 489.40 | 2024-10-07 10:35:00 | 486.86 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-07 10:30:00 | 489.40 | 2024-10-07 10:40:00 | 489.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:40:00 | 506.35 | 2024-10-11 09:45:00 | 504.07 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-14 11:00:00 | 508.45 | 2024-10-14 11:15:00 | 510.52 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-11-12 10:25:00 | 539.10 | 2024-11-12 10:30:00 | 536.01 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-11-13 09:30:00 | 528.50 | 2024-11-13 09:35:00 | 531.86 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2024-11-28 09:45:00 | 530.35 | 2024-11-28 10:05:00 | 528.07 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-04 09:50:00 | 522.25 | 2024-12-04 10:05:00 | 519.44 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-12-04 09:50:00 | 522.25 | 2024-12-04 15:20:00 | 507.50 | TARGET_HIT | 0.50 | 2.82% |
| SELL | retest1 | 2024-12-10 11:00:00 | 499.85 | 2024-12-10 11:30:00 | 501.84 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-16 09:35:00 | 496.65 | 2024-12-16 10:50:00 | 492.98 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-12-16 09:35:00 | 496.65 | 2024-12-16 11:25:00 | 496.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-31 11:00:00 | 565.30 | 2024-12-31 11:15:00 | 562.18 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-01-02 10:30:00 | 584.60 | 2025-01-02 11:40:00 | 582.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-09 10:50:00 | 591.75 | 2025-01-09 11:05:00 | 593.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-21 10:00:00 | 592.50 | 2025-01-21 10:30:00 | 588.77 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-01-21 10:00:00 | 592.50 | 2025-01-21 12:00:00 | 592.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-26 09:30:00 | 874.45 | 2025-03-26 09:40:00 | 866.90 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2025-03-26 09:30:00 | 874.45 | 2025-03-26 15:20:00 | 857.85 | TARGET_HIT | 0.50 | 1.90% |
