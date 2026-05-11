# Kalyan Jewellers India Ltd. (KALYANKJIL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 425.00
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 14 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 108 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 65
- **Target hits / Stop hits / Partials:** 14 / 65 / 29
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 16.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 14 | 30.4% | 5 | 32 | 9 | 0.03% | 1.2% |
| BUY @ 2nd Alert (retest1) | 46 | 14 | 30.4% | 5 | 32 | 9 | 0.03% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 62 | 29 | 46.8% | 9 | 33 | 20 | 0.25% | 15.3% |
| SELL @ 2nd Alert (retest1) | 62 | 29 | 46.8% | 9 | 33 | 20 | 0.25% | 15.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 108 | 43 | 39.8% | 14 | 65 | 29 | 0.15% | 16.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:05:00 | 558.00 | 554.34 | 0.00 | ORB-long ORB[551.00,557.25] vol=2.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-05-16 10:20:00 | 555.79 | 554.71 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:35:00 | 554.15 | 558.52 | 0.00 | ORB-short ORB[557.35,563.40] vol=2.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-05-19 09:40:00 | 556.05 | 558.18 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:40:00 | 554.55 | 551.54 | 0.00 | ORB-long ORB[547.00,552.15] vol=2.1x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-05-21 09:55:00 | 552.35 | 552.07 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 09:40:00 | 574.35 | 570.00 | 0.00 | ORB-long ORB[564.00,570.85] vol=1.6x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-05-27 09:55:00 | 572.29 | 571.51 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:00:00 | 567.25 | 565.34 | 0.00 | ORB-long ORB[562.35,566.90] vol=1.6x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 565.47 | 566.22 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:40:00 | 562.10 | 563.19 | 0.00 | ORB-short ORB[563.45,567.90] vol=2.9x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-05-30 11:30:00 | 563.90 | 562.95 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 09:35:00 | 551.80 | 553.64 | 0.00 | ORB-short ORB[551.85,559.60] vol=3.2x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-06-05 09:55:00 | 553.14 | 553.24 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:00:00 | 544.80 | 546.97 | 0.00 | ORB-short ORB[545.55,549.70] vol=2.8x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 10:15:00 | 542.60 | 546.15 | 0.00 | T1 1.5R @ 542.60 |
| Target hit | 2025-06-11 15:20:00 | 534.75 | 540.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-06-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 10:35:00 | 525.65 | 522.31 | 0.00 | ORB-long ORB[519.30,523.95] vol=1.8x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-06-17 10:40:00 | 524.33 | 522.46 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:35:00 | 517.50 | 514.93 | 0.00 | ORB-long ORB[509.70,517.35] vol=2.4x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:50:00 | 520.00 | 516.16 | 0.00 | T1 1.5R @ 520.00 |
| Stop hit — per-position SL triggered | 2025-06-18 10:05:00 | 517.50 | 516.42 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 09:30:00 | 507.50 | 509.11 | 0.00 | ORB-short ORB[508.00,513.80] vol=1.9x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-06-20 09:35:00 | 509.18 | 509.07 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-23 09:40:00 | 504.05 | 509.15 | 0.00 | ORB-short ORB[507.40,515.00] vol=1.6x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-06-23 09:50:00 | 506.39 | 508.23 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:35:00 | 522.90 | 521.71 | 0.00 | ORB-long ORB[517.45,522.75] vol=1.9x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-06-24 09:45:00 | 521.02 | 521.66 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 10:30:00 | 562.85 | 558.72 | 0.00 | ORB-long ORB[553.10,558.45] vol=1.6x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-07-01 10:40:00 | 560.67 | 558.90 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:40:00 | 592.80 | 589.38 | 0.00 | ORB-long ORB[586.10,590.35] vol=1.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-07-16 10:50:00 | 591.34 | 589.72 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:40:00 | 595.95 | 594.35 | 0.00 | ORB-long ORB[589.70,595.75] vol=1.7x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 09:50:00 | 598.26 | 596.65 | 0.00 | T1 1.5R @ 598.26 |
| Target hit | 2025-07-17 12:15:00 | 601.85 | 602.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2025-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:50:00 | 595.00 | 600.12 | 0.00 | ORB-short ORB[602.30,607.50] vol=1.5x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-07-25 11:05:00 | 596.85 | 599.81 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:00:00 | 604.50 | 602.50 | 0.00 | ORB-long ORB[595.10,603.50] vol=2.2x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 601.61 | 602.45 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 09:30:00 | 583.50 | 581.61 | 0.00 | ORB-long ORB[577.40,583.45] vol=1.5x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-08-04 09:35:00 | 581.46 | 581.59 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:00:00 | 582.05 | 586.20 | 0.00 | ORB-short ORB[583.00,589.90] vol=1.8x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:50:00 | 579.73 | 585.04 | 0.00 | T1 1.5R @ 579.73 |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 582.05 | 584.61 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 512.55 | 515.66 | 0.00 | ORB-short ORB[515.10,517.95] vol=1.8x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-08-14 09:35:00 | 514.20 | 515.55 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 10:55:00 | 526.00 | 531.52 | 0.00 | ORB-short ORB[532.25,539.90] vol=1.5x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 12:05:00 | 523.18 | 530.23 | 0.00 | T1 1.5R @ 523.18 |
| Target hit | 2025-08-18 15:20:00 | 519.60 | 526.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 504.40 | 508.51 | 0.00 | ORB-short ORB[509.00,512.50] vol=2.6x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:45:00 | 502.12 | 507.51 | 0.00 | T1 1.5R @ 502.12 |
| Stop hit — per-position SL triggered | 2025-08-26 10:00:00 | 504.40 | 506.91 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 09:45:00 | 502.75 | 497.37 | 0.00 | ORB-long ORB[492.95,498.55] vol=1.7x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 10:00:00 | 505.90 | 498.72 | 0.00 | T1 1.5R @ 505.90 |
| Target hit | 2025-08-28 15:20:00 | 511.00 | 506.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-09-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:40:00 | 514.50 | 509.69 | 0.00 | ORB-long ORB[506.20,512.55] vol=2.1x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:05:00 | 517.04 | 511.18 | 0.00 | T1 1.5R @ 517.04 |
| Stop hit — per-position SL triggered | 2025-09-02 11:10:00 | 514.50 | 511.29 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 11:05:00 | 508.85 | 505.56 | 0.00 | ORB-long ORB[500.00,504.85] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-09-08 11:25:00 | 507.75 | 505.73 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 09:30:00 | 501.10 | 503.16 | 0.00 | ORB-short ORB[502.25,507.45] vol=2.3x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-09-09 09:40:00 | 502.30 | 502.71 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 11:15:00 | 500.40 | 501.75 | 0.00 | ORB-short ORB[500.55,505.70] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-09-10 11:50:00 | 501.26 | 501.54 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 11:05:00 | 516.70 | 512.09 | 0.00 | ORB-long ORB[505.75,513.25] vol=8.5x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 515.07 | 512.92 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:00:00 | 521.10 | 523.17 | 0.00 | ORB-short ORB[521.25,525.75] vol=1.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-09-18 11:15:00 | 522.28 | 523.12 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:50:00 | 501.45 | 503.95 | 0.00 | ORB-short ORB[503.90,506.90] vol=2.3x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 10:55:00 | 499.62 | 503.26 | 0.00 | T1 1.5R @ 499.62 |
| Target hit | 2025-09-23 15:20:00 | 495.35 | 498.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2025-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:40:00 | 488.15 | 492.35 | 0.00 | ORB-short ORB[491.65,496.95] vol=1.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-09-24 09:45:00 | 489.59 | 491.96 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:45:00 | 454.30 | 458.21 | 0.00 | ORB-short ORB[460.70,466.20] vol=1.6x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:10:00 | 450.02 | 456.72 | 0.00 | T1 1.5R @ 450.02 |
| Stop hit — per-position SL triggered | 2025-09-26 10:20:00 | 454.30 | 456.12 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 11:15:00 | 462.50 | 460.07 | 0.00 | ORB-long ORB[455.45,461.65] vol=2.9x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-10-01 11:25:00 | 460.86 | 460.12 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:00:00 | 480.00 | 488.70 | 0.00 | ORB-short ORB[489.15,495.65] vol=1.5x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-10-06 11:20:00 | 481.93 | 487.87 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 476.25 | 478.51 | 0.00 | ORB-short ORB[476.50,483.55] vol=2.2x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 477.77 | 478.50 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:50:00 | 476.15 | 471.97 | 0.00 | ORB-long ORB[467.05,472.60] vol=2.3x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-10-15 11:00:00 | 474.85 | 472.49 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:45:00 | 480.65 | 478.93 | 0.00 | ORB-long ORB[476.05,479.70] vol=7.3x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:55:00 | 482.56 | 479.46 | 0.00 | T1 1.5R @ 482.56 |
| Target hit | 2025-10-16 15:20:00 | 487.00 | 485.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 501.65 | 499.21 | 0.00 | ORB-long ORB[494.95,499.65] vol=4.0x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-10-23 09:50:00 | 499.73 | 500.55 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:50:00 | 499.70 | 496.94 | 0.00 | ORB-long ORB[492.60,495.95] vol=2.0x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 498.28 | 497.25 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 11:05:00 | 502.70 | 498.56 | 0.00 | ORB-long ORB[494.20,498.65] vol=3.3x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-10-27 11:10:00 | 501.35 | 498.69 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 09:35:00 | 515.75 | 511.89 | 0.00 | ORB-long ORB[508.55,513.00] vol=2.2x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:30:00 | 518.68 | 515.00 | 0.00 | T1 1.5R @ 518.68 |
| Stop hit — per-position SL triggered | 2025-10-30 12:10:00 | 515.75 | 515.52 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:30:00 | 513.80 | 511.18 | 0.00 | ORB-long ORB[508.40,512.35] vol=1.8x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-11-03 09:35:00 | 512.31 | 511.56 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 518.85 | 521.93 | 0.00 | ORB-short ORB[521.10,525.00] vol=1.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-11-06 09:35:00 | 520.38 | 521.76 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:30:00 | 488.95 | 490.81 | 0.00 | ORB-short ORB[489.70,495.40] vol=1.8x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 09:50:00 | 487.21 | 489.69 | 0.00 | T1 1.5R @ 487.21 |
| Stop hit — per-position SL triggered | 2025-11-18 10:05:00 | 488.95 | 489.44 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:40:00 | 506.90 | 503.27 | 0.00 | ORB-long ORB[499.80,505.60] vol=2.7x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:50:00 | 509.30 | 504.31 | 0.00 | T1 1.5R @ 509.30 |
| Target hit | 2025-11-20 13:30:00 | 508.35 | 508.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — SELL (started 2025-11-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:45:00 | 499.30 | 502.00 | 0.00 | ORB-short ORB[500.35,505.65] vol=2.0x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-11-21 11:05:00 | 500.48 | 501.78 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:10:00 | 499.00 | 496.39 | 0.00 | ORB-long ORB[492.35,498.00] vol=1.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 497.47 | 496.44 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:15:00 | 493.65 | 496.79 | 0.00 | ORB-short ORB[496.55,500.00] vol=2.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 12:00:00 | 492.43 | 496.09 | 0.00 | T1 1.5R @ 492.43 |
| Stop hit — per-position SL triggered | 2025-11-27 14:30:00 | 493.65 | 494.73 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:30:00 | 499.40 | 495.85 | 0.00 | ORB-long ORB[492.05,496.50] vol=5.2x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-11-28 10:35:00 | 497.96 | 496.31 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:15:00 | 502.45 | 504.21 | 0.00 | ORB-short ORB[502.95,506.95] vol=1.7x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 13:05:00 | 500.94 | 503.47 | 0.00 | T1 1.5R @ 500.94 |
| Stop hit — per-position SL triggered | 2025-12-02 13:55:00 | 502.45 | 503.29 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 494.95 | 496.77 | 0.00 | ORB-short ORB[495.25,500.95] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-12-03 09:40:00 | 496.03 | 496.59 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:15:00 | 492.30 | 490.09 | 0.00 | ORB-long ORB[485.80,492.00] vol=2.2x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-12-04 11:45:00 | 491.09 | 490.32 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:55:00 | 487.70 | 489.25 | 0.00 | ORB-short ORB[488.10,492.45] vol=1.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 488.95 | 489.30 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:35:00 | 486.65 | 490.34 | 0.00 | ORB-short ORB[490.50,494.60] vol=1.7x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:55:00 | 484.22 | 489.67 | 0.00 | T1 1.5R @ 484.22 |
| Target hit | 2025-12-08 15:20:00 | 474.85 | 481.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:30:00 | 462.65 | 467.79 | 0.00 | ORB-short ORB[466.80,473.00] vol=2.2x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:50:00 | 459.55 | 464.71 | 0.00 | T1 1.5R @ 459.55 |
| Stop hit — per-position SL triggered | 2025-12-09 09:55:00 | 462.65 | 464.14 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:15:00 | 484.00 | 478.45 | 0.00 | ORB-long ORB[475.40,478.25] vol=2.3x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-12-15 10:35:00 | 482.40 | 479.39 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 11:00:00 | 468.25 | 469.69 | 0.00 | ORB-short ORB[469.00,471.75] vol=5.5x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 469.63 | 469.63 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 11:00:00 | 487.40 | 485.94 | 0.00 | ORB-long ORB[482.00,487.00] vol=2.8x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 486.49 | 486.02 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 11:10:00 | 489.20 | 486.05 | 0.00 | ORB-long ORB[484.00,488.40] vol=2.3x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-12-23 11:20:00 | 488.00 | 486.52 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 491.70 | 489.96 | 0.00 | ORB-long ORB[486.20,490.35] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-12-24 11:00:00 | 490.52 | 489.97 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 495.85 | 492.75 | 0.00 | ORB-long ORB[488.10,493.80] vol=2.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-12-29 10:10:00 | 494.24 | 493.92 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:15:00 | 482.05 | 484.29 | 0.00 | ORB-short ORB[483.00,487.40] vol=2.0x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:30:00 | 480.39 | 483.99 | 0.00 | T1 1.5R @ 480.39 |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 482.05 | 482.68 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:10:00 | 490.75 | 487.82 | 0.00 | ORB-long ORB[484.60,487.85] vol=3.9x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:15:00 | 492.50 | 489.24 | 0.00 | T1 1.5R @ 492.50 |
| Target hit | 2026-01-02 15:20:00 | 495.80 | 493.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2026-01-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:40:00 | 492.15 | 495.31 | 0.00 | ORB-short ORB[492.70,499.25] vol=2.2x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:45:00 | 490.21 | 494.55 | 0.00 | T1 1.5R @ 490.21 |
| Stop hit — per-position SL triggered | 2026-01-05 09:55:00 | 492.15 | 493.65 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:35:00 | 494.20 | 499.76 | 0.00 | ORB-short ORB[499.45,505.55] vol=2.0x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 496.29 | 498.24 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:55:00 | 460.10 | 464.68 | 0.00 | ORB-short ORB[465.75,471.85] vol=1.8x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:10:00 | 457.62 | 463.63 | 0.00 | T1 1.5R @ 457.62 |
| Target hit | 2026-01-19 15:00:00 | 458.00 | 454.49 | 0.00 | Trail-exit close>VWAP |

### Cycle 68 — SELL (started 2026-01-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:00:00 | 370.75 | 373.94 | 0.00 | ORB-short ORB[370.85,375.60] vol=1.6x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 10:45:00 | 367.65 | 372.42 | 0.00 | T1 1.5R @ 367.65 |
| Target hit | 2026-01-28 15:20:00 | 368.15 | 369.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2026-01-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:20:00 | 360.90 | 363.94 | 0.00 | ORB-short ORB[365.50,369.65] vol=1.6x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:40:00 | 358.33 | 363.27 | 0.00 | T1 1.5R @ 358.33 |
| Stop hit — per-position SL triggered | 2026-01-29 11:00:00 | 360.90 | 362.67 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 429.25 | 430.20 | 0.00 | ORB-short ORB[429.50,435.35] vol=1.8x ATR=1.76 |
| Stop hit — per-position SL triggered | 2026-02-11 11:05:00 | 431.01 | 430.20 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:35:00 | 413.00 | 417.73 | 0.00 | ORB-short ORB[418.30,422.85] vol=1.7x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 411.00 | 416.03 | 0.00 | T1 1.5R @ 411.00 |
| Stop hit — per-position SL triggered | 2026-02-18 11:30:00 | 413.00 | 415.92 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 410.50 | 414.31 | 0.00 | ORB-short ORB[415.65,420.65] vol=1.5x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:10:00 | 408.53 | 412.43 | 0.00 | T1 1.5R @ 408.53 |
| Target hit | 2026-02-19 15:20:00 | 403.70 | 409.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2026-03-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:40:00 | 382.70 | 380.16 | 0.00 | ORB-long ORB[376.75,381.60] vol=1.9x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:50:00 | 385.40 | 381.31 | 0.00 | T1 1.5R @ 385.40 |
| Stop hit — per-position SL triggered | 2026-03-16 10:10:00 | 382.70 | 381.82 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:30:00 | 431.50 | 439.96 | 0.00 | ORB-short ORB[439.70,443.45] vol=3.5x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:35:00 | 428.16 | 435.01 | 0.00 | T1 1.5R @ 428.16 |
| Target hit | 2026-04-17 13:15:00 | 423.85 | 423.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 75 — SELL (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 409.60 | 411.26 | 0.00 | ORB-short ORB[410.10,414.25] vol=1.8x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-04-23 09:50:00 | 410.77 | 410.89 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 416.50 | 414.22 | 0.00 | ORB-long ORB[411.00,414.70] vol=2.1x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 415.58 | 414.31 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 406.00 | 410.01 | 0.00 | ORB-short ORB[411.65,415.00] vol=10.9x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-05-05 11:30:00 | 407.20 | 408.90 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 412.90 | 410.76 | 0.00 | ORB-long ORB[408.55,412.05] vol=3.9x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-05-06 11:25:00 | 411.95 | 410.99 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 412.05 | 414.92 | 0.00 | ORB-short ORB[414.30,419.65] vol=2.5x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:50:00 | 409.98 | 412.86 | 0.00 | T1 1.5R @ 409.98 |
| Target hit | 2026-05-07 14:25:00 | 409.95 | 409.82 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 10:05:00 | 558.00 | 2025-05-16 10:20:00 | 555.79 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-05-19 09:35:00 | 554.15 | 2025-05-19 09:40:00 | 556.05 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-05-21 09:40:00 | 554.55 | 2025-05-21 09:55:00 | 552.35 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-05-27 09:40:00 | 574.35 | 2025-05-27 09:55:00 | 572.29 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-28 10:00:00 | 567.25 | 2025-05-28 11:15:00 | 565.47 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-05-30 10:40:00 | 562.10 | 2025-05-30 11:30:00 | 563.90 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-06-05 09:35:00 | 551.80 | 2025-06-05 09:55:00 | 553.14 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-11 10:00:00 | 544.80 | 2025-06-11 10:15:00 | 542.60 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-06-11 10:00:00 | 544.80 | 2025-06-11 15:20:00 | 534.75 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2025-06-17 10:35:00 | 525.65 | 2025-06-17 10:40:00 | 524.33 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-18 09:35:00 | 517.50 | 2025-06-18 09:50:00 | 520.00 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-06-18 09:35:00 | 517.50 | 2025-06-18 10:05:00 | 517.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-20 09:30:00 | 507.50 | 2025-06-20 09:35:00 | 509.18 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-06-23 09:40:00 | 504.05 | 2025-06-23 09:50:00 | 506.39 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-06-24 09:35:00 | 522.90 | 2025-06-24 09:45:00 | 521.02 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-07-01 10:30:00 | 562.85 | 2025-07-01 10:40:00 | 560.67 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-16 10:40:00 | 592.80 | 2025-07-16 10:50:00 | 591.34 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-17 09:40:00 | 595.95 | 2025-07-17 09:50:00 | 598.26 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-17 09:40:00 | 595.95 | 2025-07-17 12:15:00 | 601.85 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2025-07-25 10:50:00 | 595.00 | 2025-07-25 11:05:00 | 596.85 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-29 10:00:00 | 604.50 | 2025-07-29 10:15:00 | 601.61 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-08-04 09:30:00 | 583.50 | 2025-08-04 09:35:00 | 581.46 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-07 11:00:00 | 582.05 | 2025-08-07 11:50:00 | 579.73 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-07 11:00:00 | 582.05 | 2025-08-07 12:15:00 | 582.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 09:30:00 | 512.55 | 2025-08-14 09:35:00 | 514.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-18 10:55:00 | 526.00 | 2025-08-18 12:05:00 | 523.18 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-08-18 10:55:00 | 526.00 | 2025-08-18 15:20:00 | 519.60 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2025-08-26 09:35:00 | 504.40 | 2025-08-26 09:45:00 | 502.12 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-08-26 09:35:00 | 504.40 | 2025-08-26 10:00:00 | 504.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-28 09:45:00 | 502.75 | 2025-08-28 10:00:00 | 505.90 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-08-28 09:45:00 | 502.75 | 2025-08-28 15:20:00 | 511.00 | TARGET_HIT | 0.50 | 1.64% |
| BUY | retest1 | 2025-09-02 10:40:00 | 514.50 | 2025-09-02 11:05:00 | 517.04 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-09-02 10:40:00 | 514.50 | 2025-09-02 11:10:00 | 514.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 11:05:00 | 508.85 | 2025-09-08 11:25:00 | 507.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-09 09:30:00 | 501.10 | 2025-09-09 09:40:00 | 502.30 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-10 11:15:00 | 500.40 | 2025-09-10 11:50:00 | 501.26 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-17 11:05:00 | 516.70 | 2025-09-17 11:15:00 | 515.07 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-18 11:00:00 | 521.10 | 2025-09-18 11:15:00 | 522.28 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-23 10:50:00 | 501.45 | 2025-09-23 10:55:00 | 499.62 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-23 10:50:00 | 501.45 | 2025-09-23 15:20:00 | 495.35 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2025-09-24 09:40:00 | 488.15 | 2025-09-24 09:45:00 | 489.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-26 09:45:00 | 454.30 | 2025-09-26 10:10:00 | 450.02 | PARTIAL | 0.50 | 0.94% |
| SELL | retest1 | 2025-09-26 09:45:00 | 454.30 | 2025-09-26 10:20:00 | 454.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-01 11:15:00 | 462.50 | 2025-10-01 11:25:00 | 460.86 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-10-06 11:00:00 | 480.00 | 2025-10-06 11:20:00 | 481.93 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-13 09:30:00 | 476.25 | 2025-10-13 09:35:00 | 477.77 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-15 10:50:00 | 476.15 | 2025-10-15 11:00:00 | 474.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-16 10:45:00 | 480.65 | 2025-10-16 10:55:00 | 482.56 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-16 10:45:00 | 480.65 | 2025-10-16 15:20:00 | 487.00 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-10-23 09:30:00 | 501.65 | 2025-10-23 09:50:00 | 499.73 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-24 09:50:00 | 499.70 | 2025-10-24 10:15:00 | 498.28 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-27 11:05:00 | 502.70 | 2025-10-27 11:10:00 | 501.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-30 09:35:00 | 515.75 | 2025-10-30 11:30:00 | 518.68 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-30 09:35:00 | 515.75 | 2025-10-30 12:10:00 | 515.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 09:30:00 | 513.80 | 2025-11-03 09:35:00 | 512.31 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-06 09:30:00 | 518.85 | 2025-11-06 09:35:00 | 520.38 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-18 09:30:00 | 488.95 | 2025-11-18 09:50:00 | 487.21 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-18 09:30:00 | 488.95 | 2025-11-18 10:05:00 | 488.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-20 10:40:00 | 506.90 | 2025-11-20 10:50:00 | 509.30 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-11-20 10:40:00 | 506.90 | 2025-11-20 13:30:00 | 508.35 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-21 10:45:00 | 499.30 | 2025-11-21 11:05:00 | 500.48 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-24 10:10:00 | 499.00 | 2025-11-24 10:15:00 | 497.47 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-27 11:15:00 | 493.65 | 2025-11-27 12:00:00 | 492.43 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-11-27 11:15:00 | 493.65 | 2025-11-27 14:30:00 | 493.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-28 10:30:00 | 499.40 | 2025-11-28 10:35:00 | 497.96 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-02 11:15:00 | 502.45 | 2025-12-02 13:05:00 | 500.94 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-02 11:15:00 | 502.45 | 2025-12-02 13:55:00 | 502.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 09:30:00 | 494.95 | 2025-12-03 09:40:00 | 496.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-04 11:15:00 | 492.30 | 2025-12-04 11:45:00 | 491.09 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-05 09:55:00 | 487.70 | 2025-12-05 10:00:00 | 488.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-08 10:35:00 | 486.65 | 2025-12-08 10:55:00 | 484.22 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-12-08 10:35:00 | 486.65 | 2025-12-08 15:20:00 | 474.85 | TARGET_HIT | 0.50 | 2.42% |
| SELL | retest1 | 2025-12-09 09:30:00 | 462.65 | 2025-12-09 09:50:00 | 459.55 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-12-09 09:30:00 | 462.65 | 2025-12-09 09:55:00 | 462.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-15 10:15:00 | 484.00 | 2025-12-15 10:35:00 | 482.40 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-18 11:00:00 | 468.25 | 2025-12-18 11:15:00 | 469.63 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-22 11:00:00 | 487.40 | 2025-12-22 11:15:00 | 486.49 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-23 11:10:00 | 489.20 | 2025-12-23 11:20:00 | 488.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-24 10:55:00 | 491.70 | 2025-12-24 11:00:00 | 490.52 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-29 09:30:00 | 495.85 | 2025-12-29 10:10:00 | 494.24 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-30 11:15:00 | 482.05 | 2025-12-30 11:30:00 | 480.39 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-30 11:15:00 | 482.05 | 2025-12-30 13:15:00 | 482.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 10:10:00 | 490.75 | 2026-01-02 10:15:00 | 492.50 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-01-02 10:10:00 | 490.75 | 2026-01-02 15:20:00 | 495.80 | TARGET_HIT | 0.50 | 1.03% |
| SELL | retest1 | 2026-01-05 09:40:00 | 492.15 | 2026-01-05 09:45:00 | 490.21 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-01-05 09:40:00 | 492.15 | 2026-01-05 09:55:00 | 492.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-13 09:35:00 | 494.20 | 2026-01-13 09:45:00 | 496.29 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-01-19 10:55:00 | 460.10 | 2026-01-19 11:10:00 | 457.62 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-01-19 10:55:00 | 460.10 | 2026-01-19 15:00:00 | 458.00 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2026-01-28 10:00:00 | 370.75 | 2026-01-28 10:45:00 | 367.65 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2026-01-28 10:00:00 | 370.75 | 2026-01-28 15:20:00 | 368.15 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2026-01-29 10:20:00 | 360.90 | 2026-01-29 10:40:00 | 358.33 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-01-29 10:20:00 | 360.90 | 2026-01-29 11:00:00 | 360.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 10:35:00 | 429.25 | 2026-02-11 11:05:00 | 431.01 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-18 10:35:00 | 413.00 | 2026-02-18 11:25:00 | 411.00 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-18 10:35:00 | 413.00 | 2026-02-18 11:30:00 | 413.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:35:00 | 410.50 | 2026-02-19 12:10:00 | 408.53 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-19 10:35:00 | 410.50 | 2026-02-19 15:20:00 | 403.70 | TARGET_HIT | 0.50 | 1.66% |
| BUY | retest1 | 2026-03-16 09:40:00 | 382.70 | 2026-03-16 09:50:00 | 385.40 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-16 09:40:00 | 382.70 | 2026-03-16 10:10:00 | 382.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-17 10:30:00 | 431.50 | 2026-04-17 10:35:00 | 428.16 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-04-17 10:30:00 | 431.50 | 2026-04-17 13:15:00 | 423.85 | TARGET_HIT | 0.50 | 1.77% |
| SELL | retest1 | 2026-04-23 09:35:00 | 409.60 | 2026-04-23 09:50:00 | 410.77 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-28 10:55:00 | 416.50 | 2026-04-28 11:00:00 | 415.58 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-05 10:55:00 | 406.00 | 2026-05-05 11:30:00 | 407.20 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 11:05:00 | 412.90 | 2026-05-06 11:25:00 | 411.95 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-07 09:30:00 | 412.05 | 2026-05-07 09:50:00 | 409.98 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-05-07 09:30:00 | 412.05 | 2026-05-07 14:25:00 | 409.95 | TARGET_HIT | 0.50 | 0.51% |
