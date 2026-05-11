# Berger Paints India Ltd. (BERGEPAINT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-12-06 15:25:00 (10683 bars)
- **Last close:** 480.75
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
| ENTRY1 | 60 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 8 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 52
- **Target hits / Stop hits / Partials:** 8 / 52 / 23
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 7.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 16 | 38.1% | 3 | 26 | 13 | 0.08% | 3.4% |
| BUY @ 2nd Alert (retest1) | 42 | 16 | 38.1% | 3 | 26 | 13 | 0.08% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 15 | 36.6% | 5 | 26 | 10 | 0.09% | 3.6% |
| SELL @ 2nd Alert (retest1) | 41 | 15 | 36.6% | 5 | 26 | 10 | 0.09% | 3.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 83 | 31 | 37.3% | 8 | 52 | 23 | 0.08% | 7.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 487.70 | 489.60 | 0.00 | ORB-short ORB[488.45,494.00] vol=1.7x ATR=1.24 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 488.94 | 488.76 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:00:00 | 491.65 | 489.97 | 0.00 | ORB-long ORB[487.45,491.20] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-05-23 10:35:00 | 490.54 | 490.36 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 485.40 | 487.53 | 0.00 | ORB-short ORB[486.35,489.90] vol=2.9x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-05-24 09:40:00 | 486.48 | 487.34 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 11:00:00 | 491.85 | 489.91 | 0.00 | ORB-long ORB[487.10,489.95] vol=3.4x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:25:00 | 493.33 | 490.55 | 0.00 | T1 1.5R @ 493.33 |
| Stop hit — per-position SL triggered | 2024-05-28 11:30:00 | 491.85 | 490.67 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:10:00 | 484.40 | 487.44 | 0.00 | ORB-short ORB[488.05,491.90] vol=1.9x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-05-30 12:05:00 | 485.33 | 486.75 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:25:00 | 504.00 | 502.33 | 0.00 | ORB-long ORB[497.00,503.00] vol=2.9x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-06-13 10:35:00 | 502.43 | 502.47 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:40:00 | 502.20 | 504.10 | 0.00 | ORB-short ORB[503.45,506.70] vol=1.9x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:10:00 | 500.30 | 502.83 | 0.00 | T1 1.5R @ 500.30 |
| Target hit | 2024-06-19 11:15:00 | 501.60 | 501.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2024-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:35:00 | 503.70 | 501.55 | 0.00 | ORB-long ORB[497.50,502.00] vol=1.9x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:05:00 | 505.70 | 503.37 | 0.00 | T1 1.5R @ 505.70 |
| Target hit | 2024-06-20 15:15:00 | 510.15 | 510.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2024-06-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 11:00:00 | 505.85 | 509.07 | 0.00 | ORB-short ORB[509.55,515.00] vol=1.8x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-06-21 11:45:00 | 507.14 | 508.77 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:00:00 | 500.40 | 497.79 | 0.00 | ORB-long ORB[495.00,498.55] vol=5.0x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-06-27 10:45:00 | 499.13 | 498.82 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:45:00 | 513.60 | 511.45 | 0.00 | ORB-long ORB[509.05,513.00] vol=3.0x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:50:00 | 515.59 | 512.90 | 0.00 | T1 1.5R @ 515.59 |
| Stop hit — per-position SL triggered | 2024-07-03 09:55:00 | 513.60 | 512.98 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 11:05:00 | 514.00 | 511.89 | 0.00 | ORB-long ORB[507.60,513.90] vol=2.1x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 11:25:00 | 515.57 | 512.20 | 0.00 | T1 1.5R @ 515.57 |
| Stop hit — per-position SL triggered | 2024-07-04 12:35:00 | 514.00 | 513.01 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 510.50 | 512.70 | 0.00 | ORB-short ORB[511.90,516.10] vol=4.0x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-07-08 09:45:00 | 511.84 | 512.78 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:30:00 | 510.45 | 511.97 | 0.00 | ORB-short ORB[510.80,515.90] vol=1.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-07-10 09:35:00 | 511.51 | 511.40 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 519.40 | 521.02 | 0.00 | ORB-short ORB[520.00,523.30] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-07-16 10:50:00 | 520.48 | 520.78 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 09:50:00 | 526.25 | 524.63 | 0.00 | ORB-long ORB[520.45,525.90] vol=2.2x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-07-19 10:05:00 | 524.65 | 525.04 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 11:10:00 | 529.75 | 526.68 | 0.00 | ORB-long ORB[524.15,527.95] vol=4.1x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-07-24 11:30:00 | 528.55 | 526.94 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 537.80 | 535.37 | 0.00 | ORB-long ORB[530.00,535.00] vol=5.0x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:35:00 | 540.20 | 537.73 | 0.00 | T1 1.5R @ 540.20 |
| Target hit | 2024-07-26 15:20:00 | 541.75 | 540.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-07-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:55:00 | 546.50 | 544.51 | 0.00 | ORB-long ORB[539.95,545.55] vol=2.1x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-07-30 10:05:00 | 545.00 | 544.70 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:50:00 | 556.65 | 554.03 | 0.00 | ORB-long ORB[551.00,555.00] vol=2.1x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-07-31 13:10:00 | 554.97 | 555.76 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 09:45:00 | 535.40 | 538.04 | 0.00 | ORB-short ORB[536.50,544.20] vol=2.0x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-08-07 10:25:00 | 537.25 | 537.28 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 11:00:00 | 526.85 | 529.80 | 0.00 | ORB-short ORB[528.80,532.80] vol=3.9x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:40:00 | 524.95 | 529.15 | 0.00 | T1 1.5R @ 524.95 |
| Target hit | 2024-08-08 15:20:00 | 518.50 | 524.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-08-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:50:00 | 523.55 | 521.24 | 0.00 | ORB-long ORB[518.85,522.85] vol=1.7x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-08-09 10:00:00 | 521.80 | 521.46 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:30:00 | 528.00 | 525.42 | 0.00 | ORB-long ORB[521.00,526.00] vol=2.1x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-08-13 09:35:00 | 526.24 | 525.70 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:55:00 | 530.25 | 533.99 | 0.00 | ORB-short ORB[533.20,540.75] vol=1.7x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-08-14 11:00:00 | 531.76 | 533.93 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:55:00 | 580.55 | 576.34 | 0.00 | ORB-long ORB[569.50,575.75] vol=5.8x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 12:05:00 | 582.77 | 578.90 | 0.00 | T1 1.5R @ 582.77 |
| Target hit | 2024-08-27 15:20:00 | 583.20 | 581.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2024-08-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 11:00:00 | 578.20 | 578.32 | 0.00 | ORB-short ORB[578.25,583.70] vol=1.8x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 12:05:00 | 576.12 | 578.23 | 0.00 | T1 1.5R @ 576.12 |
| Target hit | 2024-08-28 15:20:00 | 573.10 | 576.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-08-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:55:00 | 575.45 | 572.44 | 0.00 | ORB-long ORB[569.15,574.35] vol=3.7x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-08-29 10:10:00 | 573.82 | 572.80 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:45:00 | 579.25 | 577.02 | 0.00 | ORB-long ORB[575.05,578.90] vol=1.8x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 577.80 | 578.42 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:25:00 | 600.75 | 597.29 | 0.00 | ORB-long ORB[593.05,600.00] vol=2.0x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:00:00 | 603.73 | 599.07 | 0.00 | T1 1.5R @ 603.73 |
| Stop hit — per-position SL triggered | 2024-09-05 11:25:00 | 600.75 | 599.52 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 589.00 | 592.89 | 0.00 | ORB-short ORB[592.00,597.45] vol=3.0x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-09-06 09:50:00 | 590.90 | 592.34 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 09:45:00 | 602.40 | 598.04 | 0.00 | ORB-long ORB[593.65,601.00] vol=1.7x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-09-09 09:50:00 | 600.42 | 598.42 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:35:00 | 605.40 | 602.99 | 0.00 | ORB-long ORB[601.00,603.65] vol=2.0x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-09-10 09:40:00 | 604.26 | 603.19 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:00:00 | 623.65 | 619.62 | 0.00 | ORB-long ORB[615.25,622.40] vol=1.9x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:05:00 | 626.89 | 621.24 | 0.00 | T1 1.5R @ 626.89 |
| Stop hit — per-position SL triggered | 2024-09-12 10:10:00 | 623.65 | 621.52 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:50:00 | 623.85 | 621.82 | 0.00 | ORB-long ORB[618.45,623.45] vol=3.7x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 11:50:00 | 626.15 | 622.86 | 0.00 | T1 1.5R @ 626.15 |
| Stop hit — per-position SL triggered | 2024-09-16 12:35:00 | 623.85 | 623.16 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 10:50:00 | 626.30 | 623.42 | 0.00 | ORB-long ORB[619.00,625.80] vol=2.9x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-09-19 10:55:00 | 624.59 | 623.46 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 09:30:00 | 615.95 | 618.35 | 0.00 | ORB-short ORB[617.00,622.00] vol=1.9x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-09-20 09:35:00 | 617.53 | 618.11 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:50:00 | 616.10 | 620.03 | 0.00 | ORB-short ORB[618.55,624.00] vol=1.8x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:05:00 | 613.81 | 619.25 | 0.00 | T1 1.5R @ 613.81 |
| Stop hit — per-position SL triggered | 2024-09-23 11:40:00 | 616.10 | 617.55 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:05:00 | 603.50 | 606.64 | 0.00 | ORB-short ORB[605.25,610.00] vol=2.0x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-09-25 10:20:00 | 605.11 | 606.25 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-09-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 11:00:00 | 609.85 | 610.15 | 0.00 | ORB-short ORB[611.50,615.90] vol=4.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:25:00 | 607.61 | 609.93 | 0.00 | T1 1.5R @ 607.61 |
| Stop hit — per-position SL triggered | 2024-09-26 11:35:00 | 609.85 | 609.79 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:35:00 | 624.05 | 620.63 | 0.00 | ORB-long ORB[614.25,619.55] vol=3.3x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:45:00 | 627.03 | 621.73 | 0.00 | T1 1.5R @ 627.03 |
| Stop hit — per-position SL triggered | 2024-09-27 10:50:00 | 624.05 | 622.29 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 09:35:00 | 616.75 | 618.70 | 0.00 | ORB-short ORB[617.05,624.00] vol=1.6x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:55:00 | 613.89 | 617.81 | 0.00 | T1 1.5R @ 613.89 |
| Target hit | 2024-10-01 12:15:00 | 614.30 | 614.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — SELL (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 564.65 | 572.26 | 0.00 | ORB-short ORB[574.20,581.65] vol=1.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 567.00 | 570.73 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:35:00 | 569.00 | 566.25 | 0.00 | ORB-long ORB[561.20,567.10] vol=5.1x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:45:00 | 572.13 | 567.65 | 0.00 | T1 1.5R @ 572.13 |
| Stop hit — per-position SL triggered | 2024-10-08 10:10:00 | 569.00 | 568.30 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 575.60 | 578.22 | 0.00 | ORB-short ORB[576.45,582.00] vol=1.6x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-10-10 11:20:00 | 576.89 | 577.74 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 09:40:00 | 567.90 | 570.81 | 0.00 | ORB-short ORB[568.20,574.75] vol=2.2x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-10-11 11:55:00 | 569.58 | 569.08 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:40:00 | 566.45 | 569.43 | 0.00 | ORB-short ORB[568.25,576.10] vol=4.9x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-10-14 10:40:00 | 567.97 | 567.95 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:50:00 | 550.70 | 551.58 | 0.00 | ORB-short ORB[551.60,559.60] vol=3.0x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:35:00 | 547.27 | 551.39 | 0.00 | T1 1.5R @ 547.27 |
| Target hit | 2024-10-22 15:20:00 | 539.35 | 546.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2024-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 09:30:00 | 545.60 | 541.87 | 0.00 | ORB-long ORB[535.65,542.70] vol=3.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-10-24 09:35:00 | 543.52 | 542.15 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 11:10:00 | 545.15 | 543.45 | 0.00 | ORB-long ORB[539.55,543.75] vol=4.8x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 11:20:00 | 546.92 | 543.94 | 0.00 | T1 1.5R @ 546.92 |
| Stop hit — per-position SL triggered | 2024-10-30 13:15:00 | 545.15 | 546.32 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 543.00 | 545.07 | 0.00 | ORB-short ORB[543.05,547.45] vol=2.2x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 11:35:00 | 541.26 | 544.56 | 0.00 | T1 1.5R @ 541.26 |
| Stop hit — per-position SL triggered | 2024-10-31 12:15:00 | 543.00 | 544.26 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-11-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:35:00 | 517.00 | 519.47 | 0.00 | ORB-short ORB[520.30,527.30] vol=7.7x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:50:00 | 513.88 | 517.82 | 0.00 | T1 1.5R @ 513.88 |
| Stop hit — per-position SL triggered | 2024-11-05 09:55:00 | 517.00 | 517.78 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-11-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:00:00 | 517.95 | 520.17 | 0.00 | ORB-short ORB[519.50,527.00] vol=1.8x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 520.12 | 520.07 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-11-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 11:00:00 | 510.50 | 514.54 | 0.00 | ORB-short ORB[513.05,518.00] vol=2.1x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-11-08 11:10:00 | 511.76 | 514.28 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-11-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:35:00 | 484.60 | 481.63 | 0.00 | ORB-long ORB[478.45,482.50] vol=1.8x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:00:00 | 487.06 | 482.50 | 0.00 | T1 1.5R @ 487.06 |
| Stop hit — per-position SL triggered | 2024-11-26 12:30:00 | 484.60 | 484.50 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:50:00 | 485.70 | 487.16 | 0.00 | ORB-short ORB[486.10,492.00] vol=1.9x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-11-27 11:00:00 | 486.91 | 487.11 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 492.55 | 491.29 | 0.00 | ORB-long ORB[489.15,492.10] vol=2.5x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-11-28 10:00:00 | 491.41 | 491.54 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:55:00 | 485.45 | 488.00 | 0.00 | ORB-short ORB[487.55,493.75] vol=3.6x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-12-03 11:00:00 | 486.51 | 487.71 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:40:00 | 478.15 | 479.98 | 0.00 | ORB-short ORB[479.45,484.00] vol=1.9x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:45:00 | 476.38 | 479.58 | 0.00 | T1 1.5R @ 476.38 |
| Stop hit — per-position SL triggered | 2024-12-05 10:10:00 | 478.15 | 479.14 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:10:00 | 480.10 | 482.94 | 0.00 | ORB-short ORB[482.30,486.95] vol=1.6x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-12-06 11:40:00 | 481.18 | 482.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-22 09:40:00 | 487.70 | 2024-05-22 09:55:00 | 488.94 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-23 10:00:00 | 491.65 | 2024-05-23 10:35:00 | 490.54 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-05-24 09:35:00 | 485.40 | 2024-05-24 09:40:00 | 486.48 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-05-28 11:00:00 | 491.85 | 2024-05-28 11:25:00 | 493.33 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-05-28 11:00:00 | 491.85 | 2024-05-28 11:30:00 | 491.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 11:10:00 | 484.40 | 2024-05-30 12:05:00 | 485.33 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-06-13 10:25:00 | 504.00 | 2024-06-13 10:35:00 | 502.43 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-19 09:40:00 | 502.20 | 2024-06-19 10:10:00 | 500.30 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-06-19 09:40:00 | 502.20 | 2024-06-19 11:15:00 | 501.60 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-06-20 09:35:00 | 503.70 | 2024-06-20 10:05:00 | 505.70 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-20 09:35:00 | 503.70 | 2024-06-20 15:15:00 | 510.15 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2024-06-21 11:00:00 | 505.85 | 2024-06-21 11:45:00 | 507.14 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-27 10:00:00 | 500.40 | 2024-06-27 10:45:00 | 499.13 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-03 09:45:00 | 513.60 | 2024-07-03 09:50:00 | 515.59 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-03 09:45:00 | 513.60 | 2024-07-03 09:55:00 | 513.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 11:05:00 | 514.00 | 2024-07-04 11:25:00 | 515.57 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-07-04 11:05:00 | 514.00 | 2024-07-04 12:35:00 | 514.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 09:40:00 | 510.50 | 2024-07-08 09:45:00 | 511.84 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-10 09:30:00 | 510.45 | 2024-07-10 09:35:00 | 511.51 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-16 10:30:00 | 519.40 | 2024-07-16 10:50:00 | 520.48 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-19 09:50:00 | 526.25 | 2024-07-19 10:05:00 | 524.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-24 11:10:00 | 529.75 | 2024-07-24 11:30:00 | 528.55 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-26 09:30:00 | 537.80 | 2024-07-26 10:35:00 | 540.20 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-26 09:30:00 | 537.80 | 2024-07-26 15:20:00 | 541.75 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2024-07-30 09:55:00 | 546.50 | 2024-07-30 10:05:00 | 545.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-31 09:50:00 | 556.65 | 2024-07-31 13:10:00 | 554.97 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-07 09:45:00 | 535.40 | 2024-08-07 10:25:00 | 537.25 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-08 11:00:00 | 526.85 | 2024-08-08 11:40:00 | 524.95 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-08 11:00:00 | 526.85 | 2024-08-08 15:20:00 | 518.50 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2024-08-09 09:50:00 | 523.55 | 2024-08-09 10:00:00 | 521.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-13 09:30:00 | 528.00 | 2024-08-13 09:35:00 | 526.24 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-14 10:55:00 | 530.25 | 2024-08-14 11:00:00 | 531.76 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-27 10:55:00 | 580.55 | 2024-08-27 12:05:00 | 582.77 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-08-27 10:55:00 | 580.55 | 2024-08-27 15:20:00 | 583.20 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-28 11:00:00 | 578.20 | 2024-08-28 12:05:00 | 576.12 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-28 11:00:00 | 578.20 | 2024-08-28 15:20:00 | 573.10 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2024-08-29 09:55:00 | 575.45 | 2024-08-29 10:10:00 | 573.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-03 09:45:00 | 579.25 | 2024-09-03 11:15:00 | 577.80 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-05 10:25:00 | 600.75 | 2024-09-05 11:00:00 | 603.73 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-09-05 10:25:00 | 600.75 | 2024-09-05 11:25:00 | 600.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 09:45:00 | 589.00 | 2024-09-06 09:50:00 | 590.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-09 09:45:00 | 602.40 | 2024-09-09 09:50:00 | 600.42 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-10 09:35:00 | 605.40 | 2024-09-10 09:40:00 | 604.26 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-12 10:00:00 | 623.65 | 2024-09-12 10:05:00 | 626.89 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-12 10:00:00 | 623.65 | 2024-09-12 10:10:00 | 623.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-16 10:50:00 | 623.85 | 2024-09-16 11:50:00 | 626.15 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-16 10:50:00 | 623.85 | 2024-09-16 12:35:00 | 623.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 10:50:00 | 626.30 | 2024-09-19 10:55:00 | 624.59 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-20 09:30:00 | 615.95 | 2024-09-20 09:35:00 | 617.53 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-23 10:50:00 | 616.10 | 2024-09-23 11:05:00 | 613.81 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-23 10:50:00 | 616.10 | 2024-09-23 11:40:00 | 616.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 10:05:00 | 603.50 | 2024-09-25 10:20:00 | 605.11 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-26 11:00:00 | 609.85 | 2024-09-26 11:25:00 | 607.61 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-26 11:00:00 | 609.85 | 2024-09-26 11:35:00 | 609.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-27 10:35:00 | 624.05 | 2024-09-27 10:45:00 | 627.03 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-27 10:35:00 | 624.05 | 2024-09-27 10:50:00 | 624.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 09:35:00 | 616.75 | 2024-10-01 09:55:00 | 613.89 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-01 09:35:00 | 616.75 | 2024-10-01 12:15:00 | 614.30 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-07 10:45:00 | 564.65 | 2024-10-07 11:05:00 | 567.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-08 09:35:00 | 569.00 | 2024-10-08 09:45:00 | 572.13 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-10-08 09:35:00 | 569.00 | 2024-10-08 10:10:00 | 569.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-10 11:00:00 | 575.60 | 2024-10-10 11:20:00 | 576.89 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-11 09:40:00 | 567.90 | 2024-10-11 11:55:00 | 569.58 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-14 09:40:00 | 566.45 | 2024-10-14 10:40:00 | 567.97 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-22 10:50:00 | 550.70 | 2024-10-22 11:35:00 | 547.27 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-10-22 10:50:00 | 550.70 | 2024-10-22 15:20:00 | 539.35 | TARGET_HIT | 0.50 | 2.06% |
| BUY | retest1 | 2024-10-24 09:30:00 | 545.60 | 2024-10-24 09:35:00 | 543.52 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-30 11:10:00 | 545.15 | 2024-10-30 11:20:00 | 546.92 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-10-30 11:10:00 | 545.15 | 2024-10-30 13:15:00 | 545.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-31 11:15:00 | 543.00 | 2024-10-31 11:35:00 | 541.26 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-10-31 11:15:00 | 543.00 | 2024-10-31 12:15:00 | 543.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-05 09:35:00 | 517.00 | 2024-11-05 09:50:00 | 513.88 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-11-05 09:35:00 | 517.00 | 2024-11-05 09:55:00 | 517.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 10:00:00 | 517.95 | 2024-11-07 10:15:00 | 520.12 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-11-08 11:00:00 | 510.50 | 2024-11-08 11:10:00 | 511.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-26 10:35:00 | 484.60 | 2024-11-26 11:00:00 | 487.06 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-11-26 10:35:00 | 484.60 | 2024-11-26 12:30:00 | 484.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-27 10:50:00 | 485.70 | 2024-11-27 11:00:00 | 486.91 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-28 09:30:00 | 492.55 | 2024-11-28 10:00:00 | 491.41 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-03 10:55:00 | 485.45 | 2024-12-03 11:00:00 | 486.51 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-05 09:40:00 | 478.15 | 2024-12-05 09:45:00 | 476.38 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-05 09:40:00 | 478.15 | 2024-12-05 10:10:00 | 478.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-06 11:10:00 | 480.10 | 2024-12-06 11:40:00 | 481.18 | STOP_HIT | 1.00 | -0.23% |
