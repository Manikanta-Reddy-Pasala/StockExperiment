# Dabur India Ltd. (DABUR)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15238 bars)
- **Last close:** 487.00
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
| ENTRY1 | 82 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 16 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 111 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 66
- **Target hits / Stop hits / Partials:** 16 / 66 / 29
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 11.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 26 | 41.9% | 10 | 36 | 16 | 0.10% | 6.2% |
| BUY @ 2nd Alert (retest1) | 62 | 26 | 41.9% | 10 | 36 | 16 | 0.10% | 6.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 19 | 38.8% | 6 | 30 | 13 | 0.10% | 5.1% |
| SELL @ 2nd Alert (retest1) | 49 | 19 | 38.8% | 6 | 30 | 13 | 0.10% | 5.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 111 | 45 | 40.5% | 16 | 66 | 29 | 0.10% | 11.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-14 09:45:00 | 470.00 | 470.33 | 0.00 | ORB-short ORB[470.05,474.55] vol=1.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-05-14 10:25:00 | 471.14 | 470.11 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:55:00 | 473.10 | 469.24 | 0.00 | ORB-long ORB[467.30,471.45] vol=1.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 12:30:00 | 474.59 | 470.04 | 0.00 | T1 1.5R @ 474.59 |
| Target hit | 2025-05-16 15:20:00 | 476.20 | 471.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 482.20 | 479.04 | 0.00 | ORB-long ORB[475.10,480.40] vol=1.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-05-19 09:35:00 | 480.98 | 479.18 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 11:05:00 | 482.10 | 480.11 | 0.00 | ORB-long ORB[477.30,482.00] vol=3.4x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-05-22 11:15:00 | 481.07 | 480.30 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:25:00 | 484.00 | 482.63 | 0.00 | ORB-long ORB[480.40,483.30] vol=1.6x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-05-27 10:45:00 | 483.16 | 482.72 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:00:00 | 488.00 | 486.32 | 0.00 | ORB-long ORB[483.20,487.95] vol=1.6x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:05:00 | 489.46 | 486.94 | 0.00 | T1 1.5R @ 489.46 |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 488.00 | 487.10 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:35:00 | 485.20 | 485.81 | 0.00 | ORB-short ORB[486.00,487.65] vol=4.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 485.98 | 485.49 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:55:00 | 484.15 | 483.09 | 0.00 | ORB-long ORB[482.10,483.65] vol=5.4x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-06-09 11:50:00 | 483.48 | 483.27 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:30:00 | 493.45 | 492.06 | 0.00 | ORB-long ORB[489.05,492.05] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-06-11 09:40:00 | 492.59 | 492.35 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:50:00 | 482.40 | 483.34 | 0.00 | ORB-short ORB[483.15,487.80] vol=3.0x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 10:15:00 | 480.61 | 482.74 | 0.00 | T1 1.5R @ 480.61 |
| Target hit | 2025-06-12 15:20:00 | 473.60 | 477.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 11:15:00 | 471.05 | 468.15 | 0.00 | ORB-long ORB[467.30,469.35] vol=3.3x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-06-16 11:25:00 | 470.19 | 468.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:00:00 | 467.35 | 465.13 | 0.00 | ORB-long ORB[463.10,465.80] vol=2.4x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-06-23 11:05:00 | 466.49 | 465.26 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:15:00 | 478.95 | 477.56 | 0.00 | ORB-long ORB[475.35,478.00] vol=3.0x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-06-25 10:25:00 | 478.08 | 477.99 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:45:00 | 485.20 | 483.75 | 0.00 | ORB-long ORB[481.60,484.35] vol=2.7x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:10:00 | 486.24 | 484.23 | 0.00 | T1 1.5R @ 486.24 |
| Target hit | 2025-06-27 14:00:00 | 486.30 | 486.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2025-06-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:05:00 | 480.05 | 482.92 | 0.00 | ORB-short ORB[484.35,487.75] vol=1.6x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-06-30 10:20:00 | 481.12 | 482.30 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 11:00:00 | 491.00 | 489.65 | 0.00 | ORB-long ORB[487.50,490.95] vol=2.4x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 11:05:00 | 491.80 | 489.81 | 0.00 | T1 1.5R @ 491.80 |
| Stop hit — per-position SL triggered | 2025-07-03 11:25:00 | 491.00 | 490.19 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:30:00 | 517.80 | 515.54 | 0.00 | ORB-long ORB[511.00,517.50] vol=2.0x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-07-09 09:40:00 | 516.67 | 515.96 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:05:00 | 506.65 | 508.40 | 0.00 | ORB-short ORB[507.60,510.90] vol=1.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-08-12 11:20:00 | 507.56 | 508.22 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:45:00 | 499.55 | 501.11 | 0.00 | ORB-short ORB[501.60,504.95] vol=2.1x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-08-13 11:05:00 | 500.42 | 500.90 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:45:00 | 502.90 | 503.84 | 0.00 | ORB-short ORB[503.15,505.50] vol=1.6x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-08-14 10:55:00 | 503.60 | 503.73 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 527.70 | 523.57 | 0.00 | ORB-long ORB[521.10,524.75] vol=4.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-08-20 09:35:00 | 526.59 | 523.87 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:35:00 | 522.80 | 521.98 | 0.00 | ORB-long ORB[520.05,522.45] vol=2.2x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-09-01 10:45:00 | 521.71 | 521.99 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:35:00 | 536.85 | 532.84 | 0.00 | ORB-long ORB[523.65,528.80] vol=2.3x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:05:00 | 539.60 | 536.31 | 0.00 | T1 1.5R @ 539.60 |
| Target hit | 2025-09-02 13:55:00 | 541.90 | 541.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2025-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 09:30:00 | 539.85 | 541.78 | 0.00 | ORB-short ORB[540.30,547.40] vol=1.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-09-09 09:35:00 | 541.12 | 541.65 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 09:30:00 | 536.35 | 538.21 | 0.00 | ORB-short ORB[536.50,544.10] vol=1.5x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-09-12 09:40:00 | 537.66 | 537.75 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 11:00:00 | 541.10 | 539.70 | 0.00 | ORB-long ORB[536.50,539.95] vol=5.2x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-09-15 11:55:00 | 540.30 | 540.03 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:40:00 | 537.95 | 536.33 | 0.00 | ORB-long ORB[534.45,537.40] vol=4.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-09-18 11:00:00 | 536.94 | 536.43 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:00:00 | 534.60 | 537.94 | 0.00 | ORB-short ORB[536.00,542.80] vol=1.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-09-19 11:10:00 | 535.48 | 537.84 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:00:00 | 530.15 | 531.72 | 0.00 | ORB-short ORB[530.70,536.75] vol=1.9x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-09-22 12:20:00 | 531.05 | 531.00 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 11:00:00 | 515.80 | 519.10 | 0.00 | ORB-short ORB[517.25,520.95] vol=1.6x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 13:15:00 | 514.11 | 517.66 | 0.00 | T1 1.5R @ 514.11 |
| Target hit | 2025-09-25 15:20:00 | 506.35 | 513.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-10-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:50:00 | 485.00 | 487.69 | 0.00 | ORB-short ORB[489.35,492.30] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-10-01 11:10:00 | 486.10 | 487.33 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:15:00 | 493.40 | 494.55 | 0.00 | ORB-short ORB[494.00,497.75] vol=2.1x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 13:45:00 | 491.86 | 493.87 | 0.00 | T1 1.5R @ 491.86 |
| Stop hit — per-position SL triggered | 2025-10-06 13:55:00 | 493.40 | 493.83 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:15:00 | 490.50 | 492.21 | 0.00 | ORB-short ORB[492.25,495.20] vol=2.3x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-10-07 11:45:00 | 491.25 | 491.32 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:30:00 | 483.15 | 482.26 | 0.00 | ORB-long ORB[479.15,482.05] vol=2.9x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 13:25:00 | 484.63 | 483.19 | 0.00 | T1 1.5R @ 484.63 |
| Target hit | 2025-10-09 15:20:00 | 486.00 | 483.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2025-10-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:05:00 | 483.80 | 485.83 | 0.00 | ORB-short ORB[486.20,489.40] vol=2.8x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-10-13 11:10:00 | 484.72 | 485.62 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:00:00 | 484.80 | 486.84 | 0.00 | ORB-short ORB[487.75,489.55] vol=2.0x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 485.58 | 486.67 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:35:00 | 491.60 | 490.48 | 0.00 | ORB-long ORB[486.05,489.25] vol=3.2x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:50:00 | 492.86 | 490.93 | 0.00 | T1 1.5R @ 492.86 |
| Stop hit — per-position SL triggered | 2025-10-15 11:20:00 | 491.60 | 491.15 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:55:00 | 496.85 | 495.59 | 0.00 | ORB-long ORB[493.90,496.35] vol=2.3x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 11:50:00 | 498.22 | 496.03 | 0.00 | T1 1.5R @ 498.22 |
| Target hit | 2025-10-16 15:20:00 | 499.85 | 498.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2025-10-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:10:00 | 505.60 | 502.97 | 0.00 | ORB-long ORB[499.60,503.45] vol=4.0x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:25:00 | 507.34 | 504.41 | 0.00 | T1 1.5R @ 507.34 |
| Target hit | 2025-10-17 11:50:00 | 506.60 | 507.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 510.30 | 508.00 | 0.00 | ORB-long ORB[505.30,509.55] vol=1.8x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-10-23 09:40:00 | 508.87 | 508.28 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 507.25 | 505.17 | 0.00 | ORB-long ORB[501.80,505.00] vol=3.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-10-29 11:10:00 | 506.35 | 505.46 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:35:00 | 492.65 | 490.14 | 0.00 | ORB-long ORB[485.60,489.05] vol=3.3x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-11-03 09:40:00 | 491.13 | 490.21 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:25:00 | 509.50 | 505.66 | 0.00 | ORB-long ORB[501.95,508.75] vol=2.1x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-11-04 10:45:00 | 507.53 | 506.30 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 10:55:00 | 522.35 | 517.67 | 0.00 | ORB-long ORB[513.55,519.05] vol=2.3x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-11-06 11:35:00 | 520.74 | 518.96 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:55:00 | 505.25 | 507.34 | 0.00 | ORB-short ORB[507.80,510.90] vol=2.8x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:40:00 | 503.63 | 506.30 | 0.00 | T1 1.5R @ 503.63 |
| Target hit | 2025-12-08 14:50:00 | 503.70 | 503.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 502.30 | 504.60 | 0.00 | ORB-short ORB[503.30,505.65] vol=1.7x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:55:00 | 500.86 | 503.54 | 0.00 | T1 1.5R @ 500.86 |
| Target hit | 2025-12-10 13:15:00 | 502.05 | 501.68 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — SELL (started 2025-12-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:25:00 | 496.90 | 499.89 | 0.00 | ORB-short ORB[501.10,503.70] vol=1.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-12-12 10:30:00 | 498.07 | 498.94 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:40:00 | 503.75 | 501.04 | 0.00 | ORB-long ORB[494.60,499.90] vol=3.2x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-12-16 09:50:00 | 502.56 | 501.40 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 11:00:00 | 490.90 | 491.59 | 0.00 | ORB-short ORB[491.60,494.05] vol=2.1x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 11:15:00 | 489.63 | 491.24 | 0.00 | T1 1.5R @ 489.63 |
| Stop hit — per-position SL triggered | 2025-12-18 11:20:00 | 490.90 | 491.21 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:00:00 | 486.70 | 487.80 | 0.00 | ORB-short ORB[487.50,490.60] vol=2.2x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-12-29 11:30:00 | 487.31 | 487.70 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:05:00 | 490.40 | 489.04 | 0.00 | ORB-long ORB[487.05,489.90] vol=3.1x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-12-30 10:45:00 | 489.54 | 489.42 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 499.65 | 501.66 | 0.00 | ORB-short ORB[502.00,505.90] vol=1.6x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:45:00 | 498.39 | 501.00 | 0.00 | T1 1.5R @ 498.39 |
| Stop hit — per-position SL triggered | 2026-01-01 12:25:00 | 499.65 | 500.68 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:00:00 | 510.05 | 506.34 | 0.00 | ORB-long ORB[499.00,503.00] vol=3.7x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:10:00 | 512.04 | 508.66 | 0.00 | T1 1.5R @ 512.04 |
| Target hit | 2026-01-02 13:40:00 | 517.00 | 517.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — BUY (started 2026-01-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:40:00 | 532.10 | 530.19 | 0.00 | ORB-long ORB[523.85,529.40] vol=2.3x ATR=1.92 |
| Stop hit — per-position SL triggered | 2026-01-05 10:10:00 | 530.18 | 531.10 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:50:00 | 525.40 | 521.52 | 0.00 | ORB-long ORB[519.50,524.00] vol=1.8x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-01-07 10:55:00 | 524.24 | 521.89 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-01-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 10:05:00 | 525.25 | 522.67 | 0.00 | ORB-long ORB[520.40,525.00] vol=1.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-01-12 10:30:00 | 523.64 | 523.33 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:20:00 | 518.20 | 522.21 | 0.00 | ORB-short ORB[524.70,529.15] vol=4.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-01-13 10:25:00 | 519.99 | 522.11 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:55:00 | 517.75 | 519.61 | 0.00 | ORB-short ORB[518.10,523.70] vol=6.1x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:10:00 | 516.07 | 519.22 | 0.00 | T1 1.5R @ 516.07 |
| Stop hit — per-position SL triggered | 2026-01-14 11:35:00 | 517.75 | 518.67 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-01-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:20:00 | 516.80 | 513.57 | 0.00 | ORB-long ORB[510.20,514.65] vol=1.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-01-19 10:25:00 | 515.33 | 513.76 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:50:00 | 511.15 | 512.92 | 0.00 | ORB-short ORB[512.05,516.50] vol=5.2x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:05:00 | 509.04 | 511.97 | 0.00 | T1 1.5R @ 509.04 |
| Target hit | 2026-01-20 15:20:00 | 504.40 | 509.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2026-01-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 09:40:00 | 531.95 | 530.31 | 0.00 | ORB-long ORB[522.25,525.90] vol=9.7x ATR=1.68 |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 530.27 | 530.54 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-01-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:45:00 | 503.60 | 510.40 | 0.00 | ORB-short ORB[510.60,518.00] vol=2.2x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-01-28 11:10:00 | 505.27 | 509.59 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-01-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:20:00 | 503.10 | 508.23 | 0.00 | ORB-short ORB[511.20,516.75] vol=3.3x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:45:00 | 500.37 | 506.72 | 0.00 | T1 1.5R @ 500.37 |
| Stop hit — per-position SL triggered | 2026-01-29 12:10:00 | 503.10 | 505.71 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:00:00 | 513.05 | 508.57 | 0.00 | ORB-long ORB[504.60,509.25] vol=3.9x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:05:00 | 515.07 | 512.16 | 0.00 | T1 1.5R @ 515.07 |
| Stop hit — per-position SL triggered | 2026-02-01 11:10:00 | 513.05 | 511.44 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 511.70 | 509.70 | 0.00 | ORB-long ORB[508.50,510.85] vol=2.9x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-02-09 11:05:00 | 510.78 | 509.89 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:15:00 | 519.30 | 520.63 | 0.00 | ORB-short ORB[520.00,522.85] vol=3.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-02-12 10:20:00 | 520.32 | 520.60 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:45:00 | 514.80 | 516.69 | 0.00 | ORB-short ORB[515.90,518.95] vol=2.0x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-02-13 09:50:00 | 515.94 | 516.53 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 513.80 | 510.74 | 0.00 | ORB-long ORB[508.50,512.95] vol=3.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:10:00 | 515.35 | 511.23 | 0.00 | T1 1.5R @ 515.35 |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 513.80 | 513.69 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 518.25 | 516.74 | 0.00 | ORB-long ORB[511.10,514.55] vol=1.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 517.26 | 516.96 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-02-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:05:00 | 513.25 | 512.34 | 0.00 | ORB-long ORB[509.40,512.85] vol=1.8x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 512.00 | 512.33 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 508.40 | 504.82 | 0.00 | ORB-long ORB[500.40,502.50] vol=1.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 507.16 | 505.54 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 519.00 | 520.89 | 0.00 | ORB-short ORB[521.20,523.75] vol=1.8x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-02-25 10:40:00 | 520.15 | 520.22 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 516.00 | 516.54 | 0.00 | ORB-short ORB[519.15,525.50] vol=1.9x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-02-27 11:55:00 | 517.29 | 516.44 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 486.40 | 488.18 | 0.00 | ORB-short ORB[488.05,494.00] vol=1.5x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 484.70 | 487.62 | 0.00 | T1 1.5R @ 484.70 |
| Stop hit — per-position SL triggered | 2026-03-05 12:55:00 | 486.40 | 486.16 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-03-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:00:00 | 477.70 | 475.12 | 0.00 | ORB-long ORB[471.20,476.55] vol=3.0x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 12:55:00 | 479.61 | 476.31 | 0.00 | T1 1.5R @ 479.61 |
| Target hit | 2026-03-10 15:20:00 | 482.50 | 478.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 460.65 | 457.65 | 0.00 | ORB-long ORB[453.85,458.50] vol=1.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-03-13 11:05:00 | 459.20 | 457.74 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 455.10 | 456.01 | 0.00 | ORB-short ORB[456.55,460.70] vol=1.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:35:00 | 453.27 | 455.77 | 0.00 | T1 1.5R @ 453.27 |
| Stop hit — per-position SL triggered | 2026-03-17 11:55:00 | 455.10 | 454.75 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 433.35 | 431.03 | 0.00 | ORB-long ORB[428.45,431.95] vol=8.1x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:45:00 | 435.02 | 431.25 | 0.00 | T1 1.5R @ 435.02 |
| Stop hit — per-position SL triggered | 2026-04-15 11:50:00 | 433.35 | 431.78 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 433.00 | 436.21 | 0.00 | ORB-short ORB[435.15,441.40] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-04-16 10:25:00 | 434.28 | 433.86 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 436.20 | 430.43 | 0.00 | ORB-long ORB[424.20,429.80] vol=1.7x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:15:00 | 438.59 | 432.03 | 0.00 | T1 1.5R @ 438.59 |
| Target hit | 2026-04-17 15:20:00 | 443.50 | 440.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 447.00 | 443.74 | 0.00 | ORB-long ORB[439.05,444.80] vol=1.8x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 449.11 | 446.57 | 0.00 | T1 1.5R @ 449.11 |
| Target hit | 2026-04-21 12:15:00 | 448.55 | 449.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 82 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 452.50 | 457.23 | 0.00 | ORB-short ORB[457.45,462.50] vol=1.9x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 450.80 | 456.13 | 0.00 | T1 1.5R @ 450.80 |
| Target hit | 2026-04-24 15:10:00 | 452.15 | 452.09 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-14 09:45:00 | 470.00 | 2025-05-14 10:25:00 | 471.14 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-16 10:55:00 | 473.10 | 2025-05-16 12:30:00 | 474.59 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-05-16 10:55:00 | 473.10 | 2025-05-16 15:20:00 | 476.20 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-05-19 09:30:00 | 482.20 | 2025-05-19 09:35:00 | 480.98 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-22 11:05:00 | 482.10 | 2025-05-22 11:15:00 | 481.07 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-05-27 10:25:00 | 484.00 | 2025-05-27 10:45:00 | 483.16 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-06-02 11:00:00 | 488.00 | 2025-06-02 11:05:00 | 489.46 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-06-02 11:00:00 | 488.00 | 2025-06-02 11:15:00 | 488.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 10:35:00 | 485.20 | 2025-06-06 11:15:00 | 485.98 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-06-09 10:55:00 | 484.15 | 2025-06-09 11:50:00 | 483.48 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-06-11 09:30:00 | 493.45 | 2025-06-11 09:40:00 | 492.59 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-06-12 09:50:00 | 482.40 | 2025-06-12 10:15:00 | 480.61 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-06-12 09:50:00 | 482.40 | 2025-06-12 15:20:00 | 473.60 | TARGET_HIT | 0.50 | 1.82% |
| BUY | retest1 | 2025-06-16 11:15:00 | 471.05 | 2025-06-16 11:25:00 | 470.19 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-23 11:00:00 | 467.35 | 2025-06-23 11:05:00 | 466.49 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-25 10:15:00 | 478.95 | 2025-06-25 10:25:00 | 478.08 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-27 10:45:00 | 485.20 | 2025-06-27 11:10:00 | 486.24 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-06-27 10:45:00 | 485.20 | 2025-06-27 14:00:00 | 486.30 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2025-06-30 10:05:00 | 480.05 | 2025-06-30 10:20:00 | 481.12 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-03 11:00:00 | 491.00 | 2025-07-03 11:05:00 | 491.80 | PARTIAL | 0.50 | 0.16% |
| BUY | retest1 | 2025-07-03 11:00:00 | 491.00 | 2025-07-03 11:25:00 | 491.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 09:30:00 | 517.80 | 2025-07-09 09:40:00 | 516.67 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-12 11:05:00 | 506.65 | 2025-08-12 11:20:00 | 507.56 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-13 10:45:00 | 499.55 | 2025-08-13 11:05:00 | 500.42 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-08-14 10:45:00 | 502.90 | 2025-08-14 10:55:00 | 503.60 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-08-20 09:30:00 | 527.70 | 2025-08-20 09:35:00 | 526.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-01 10:35:00 | 522.80 | 2025-09-01 10:45:00 | 521.71 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-02 09:35:00 | 536.85 | 2025-09-02 10:05:00 | 539.60 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-09-02 09:35:00 | 536.85 | 2025-09-02 13:55:00 | 541.90 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2025-09-09 09:30:00 | 539.85 | 2025-09-09 09:35:00 | 541.12 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-12 09:30:00 | 536.35 | 2025-09-12 09:40:00 | 537.66 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-15 11:00:00 | 541.10 | 2025-09-15 11:55:00 | 540.30 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-09-18 10:40:00 | 537.95 | 2025-09-18 11:00:00 | 536.94 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-19 11:00:00 | 534.60 | 2025-09-19 11:10:00 | 535.48 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-22 11:00:00 | 530.15 | 2025-09-22 12:20:00 | 531.05 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-25 11:00:00 | 515.80 | 2025-09-25 13:15:00 | 514.11 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-25 11:00:00 | 515.80 | 2025-09-25 15:20:00 | 506.35 | TARGET_HIT | 0.50 | 1.83% |
| SELL | retest1 | 2025-10-01 10:50:00 | 485.00 | 2025-10-01 11:10:00 | 486.10 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-06 11:15:00 | 493.40 | 2025-10-06 13:45:00 | 491.86 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-06 11:15:00 | 493.40 | 2025-10-06 13:55:00 | 493.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-07 11:15:00 | 490.50 | 2025-10-07 11:45:00 | 491.25 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-10-09 10:30:00 | 483.15 | 2025-10-09 13:25:00 | 484.63 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-10-09 10:30:00 | 483.15 | 2025-10-09 15:20:00 | 486.00 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-10-13 11:05:00 | 483.80 | 2025-10-13 11:10:00 | 484.72 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-14 11:00:00 | 484.80 | 2025-10-14 11:15:00 | 485.58 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-15 10:35:00 | 491.60 | 2025-10-15 10:50:00 | 492.86 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-10-15 10:35:00 | 491.60 | 2025-10-15 11:20:00 | 491.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-16 10:55:00 | 496.85 | 2025-10-16 11:50:00 | 498.22 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-16 10:55:00 | 496.85 | 2025-10-16 15:20:00 | 499.85 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-10-17 10:10:00 | 505.60 | 2025-10-17 10:25:00 | 507.34 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-10-17 10:10:00 | 505.60 | 2025-10-17 11:50:00 | 506.60 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-10-23 09:30:00 | 510.30 | 2025-10-23 09:40:00 | 508.87 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-29 10:25:00 | 507.25 | 2025-10-29 11:10:00 | 506.35 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-03 09:35:00 | 492.65 | 2025-11-03 09:40:00 | 491.13 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-04 10:25:00 | 509.50 | 2025-11-04 10:45:00 | 507.53 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-11-06 10:55:00 | 522.35 | 2025-11-06 11:35:00 | 520.74 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-08 10:55:00 | 505.25 | 2025-12-08 11:40:00 | 503.63 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-08 10:55:00 | 505.25 | 2025-12-08 14:50:00 | 503.70 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-10 10:45:00 | 502.30 | 2025-12-10 11:55:00 | 500.86 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-10 10:45:00 | 502.30 | 2025-12-10 13:15:00 | 502.05 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2025-12-12 10:25:00 | 496.90 | 2025-12-12 10:30:00 | 498.07 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-16 09:40:00 | 503.75 | 2025-12-16 09:50:00 | 502.56 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-18 11:00:00 | 490.90 | 2025-12-18 11:15:00 | 489.63 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-18 11:00:00 | 490.90 | 2025-12-18 11:20:00 | 490.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 11:00:00 | 486.70 | 2025-12-29 11:30:00 | 487.31 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-12-30 10:05:00 | 490.40 | 2025-12-30 10:45:00 | 489.54 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-01-01 11:00:00 | 499.65 | 2026-01-01 11:45:00 | 498.39 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-01-01 11:00:00 | 499.65 | 2026-01-01 12:25:00 | 499.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 10:00:00 | 510.05 | 2026-01-02 10:10:00 | 512.04 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-02 10:00:00 | 510.05 | 2026-01-02 13:40:00 | 517.00 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2026-01-05 09:40:00 | 532.10 | 2026-01-05 10:10:00 | 530.18 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-07 10:50:00 | 525.40 | 2026-01-07 10:55:00 | 524.24 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-12 10:05:00 | 525.25 | 2026-01-12 10:30:00 | 523.64 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-13 10:20:00 | 518.20 | 2026-01-13 10:25:00 | 519.99 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-14 10:55:00 | 517.75 | 2026-01-14 11:10:00 | 516.07 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-14 10:55:00 | 517.75 | 2026-01-14 11:35:00 | 517.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-19 10:20:00 | 516.80 | 2026-01-19 10:25:00 | 515.33 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-20 09:50:00 | 511.15 | 2026-01-20 12:05:00 | 509.04 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-20 09:50:00 | 511.15 | 2026-01-20 15:20:00 | 504.40 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2026-01-23 09:40:00 | 531.95 | 2026-01-23 10:15:00 | 530.27 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-28 10:45:00 | 503.60 | 2026-01-28 11:10:00 | 505.27 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-29 10:20:00 | 503.10 | 2026-01-29 10:45:00 | 500.37 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-01-29 10:20:00 | 503.10 | 2026-01-29 12:10:00 | 503.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:00:00 | 513.05 | 2026-02-01 11:05:00 | 515.07 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-01 11:00:00 | 513.05 | 2026-02-01 11:10:00 | 513.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-09 10:35:00 | 511.70 | 2026-02-09 11:05:00 | 510.78 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-12 10:15:00 | 519.30 | 2026-02-12 10:20:00 | 520.32 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-13 09:45:00 | 514.80 | 2026-02-13 09:50:00 | 515.94 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-16 11:00:00 | 513.80 | 2026-02-16 11:10:00 | 515.35 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-16 11:00:00 | 513.80 | 2026-02-16 15:15:00 | 513.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:20:00 | 518.25 | 2026-02-17 10:45:00 | 517.26 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-19 10:05:00 | 513.25 | 2026-02-19 10:15:00 | 512.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-20 10:30:00 | 508.40 | 2026-02-20 11:05:00 | 507.16 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-25 10:25:00 | 519.00 | 2026-02-25 10:40:00 | 520.15 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-27 11:05:00 | 516.00 | 2026-02-27 11:55:00 | 517.29 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 10:55:00 | 486.40 | 2026-03-05 11:25:00 | 484.70 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-03-05 10:55:00 | 486.40 | 2026-03-05 12:55:00 | 486.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 11:00:00 | 477.70 | 2026-03-10 12:55:00 | 479.61 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-03-10 11:00:00 | 477.70 | 2026-03-10 15:20:00 | 482.50 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2026-03-13 10:50:00 | 460.65 | 2026-03-13 11:05:00 | 459.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-17 10:25:00 | 455.10 | 2026-03-17 10:35:00 | 453.27 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-03-17 10:25:00 | 455.10 | 2026-03-17 11:55:00 | 455.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 11:15:00 | 433.35 | 2026-04-15 11:45:00 | 435.02 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-15 11:15:00 | 433.35 | 2026-04-15 11:50:00 | 433.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:55:00 | 433.00 | 2026-04-16 10:25:00 | 434.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 10:05:00 | 436.20 | 2026-04-17 10:15:00 | 438.59 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-17 10:05:00 | 436.20 | 2026-04-17 15:20:00 | 443.50 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2026-04-21 09:45:00 | 447.00 | 2026-04-21 10:05:00 | 449.11 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-21 09:45:00 | 447.00 | 2026-04-21 12:15:00 | 448.55 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-24 11:00:00 | 452.50 | 2026-04-24 11:15:00 | 450.80 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-24 11:00:00 | 452.50 | 2026-04-24 15:10:00 | 452.15 | TARGET_HIT | 0.50 | 0.08% |
