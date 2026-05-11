# Jubilant Foodworks Ltd. (JUBLFOOD)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
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
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 17 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 126 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 74
- **Target hits / Stop hits / Partials:** 17 / 74 / 35
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 19.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 30 | 39.0% | 11 | 47 | 19 | 0.12% | 9.6% |
| BUY @ 2nd Alert (retest1) | 77 | 30 | 39.0% | 11 | 47 | 19 | 0.12% | 9.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 22 | 44.9% | 6 | 27 | 16 | 0.19% | 9.5% |
| SELL @ 2nd Alert (retest1) | 49 | 22 | 44.9% | 6 | 27 | 16 | 0.19% | 9.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 126 | 52 | 41.3% | 17 | 74 | 35 | 0.15% | 19.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 464.15 | 467.31 | 0.00 | ORB-short ORB[468.45,471.75] vol=2.3x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-05-16 11:20:00 | 465.82 | 467.21 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:40:00 | 465.65 | 466.70 | 0.00 | ORB-short ORB[466.00,469.00] vol=2.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:05:00 | 463.77 | 466.04 | 0.00 | T1 1.5R @ 463.77 |
| Stop hit — per-position SL triggered | 2024-05-21 10:10:00 | 465.65 | 465.87 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 09:40:00 | 521.95 | 518.16 | 0.00 | ORB-long ORB[512.80,518.35] vol=1.6x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-05-30 09:55:00 | 519.55 | 519.40 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:55:00 | 542.50 | 537.59 | 0.00 | ORB-long ORB[535.50,542.00] vol=2.5x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-06-21 11:05:00 | 540.79 | 538.04 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:45:00 | 552.00 | 546.86 | 0.00 | ORB-long ORB[544.80,549.95] vol=1.8x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 12:25:00 | 555.96 | 549.85 | 0.00 | T1 1.5R @ 555.96 |
| Target hit | 2024-06-24 15:20:00 | 568.10 | 563.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 583.30 | 576.75 | 0.00 | ORB-long ORB[568.50,576.50] vol=2.4x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-06-25 09:40:00 | 580.32 | 577.85 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:40:00 | 560.30 | 558.20 | 0.00 | ORB-long ORB[553.15,557.35] vol=1.6x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-06-27 10:45:00 | 558.33 | 558.60 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:40:00 | 560.30 | 556.76 | 0.00 | ORB-long ORB[554.20,558.70] vol=1.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:20:00 | 563.55 | 560.48 | 0.00 | T1 1.5R @ 563.55 |
| Target hit | 2024-06-28 15:20:00 | 562.30 | 561.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-07-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:40:00 | 574.20 | 568.47 | 0.00 | ORB-long ORB[559.85,567.20] vol=6.4x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-07-01 09:45:00 | 571.77 | 569.17 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 565.80 | 566.90 | 0.00 | ORB-short ORB[568.00,576.00] vol=1.7x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-07-02 10:40:00 | 567.54 | 567.08 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:00:00 | 575.90 | 572.48 | 0.00 | ORB-long ORB[569.25,574.45] vol=1.8x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-07-08 10:20:00 | 574.10 | 573.36 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:20:00 | 571.85 | 572.61 | 0.00 | ORB-short ORB[572.00,575.95] vol=1.9x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 569.40 | 572.04 | 0.00 | T1 1.5R @ 569.40 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 571.85 | 571.71 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:05:00 | 574.20 | 578.78 | 0.00 | ORB-short ORB[579.05,585.85] vol=1.9x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-07-11 11:40:00 | 575.94 | 577.89 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:35:00 | 576.10 | 578.16 | 0.00 | ORB-short ORB[577.25,584.85] vol=1.6x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-07-12 10:55:00 | 577.92 | 577.86 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:50:00 | 576.20 | 578.50 | 0.00 | ORB-short ORB[576.45,582.05] vol=1.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-07-15 11:45:00 | 577.91 | 578.01 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:00:00 | 579.15 | 578.86 | 0.00 | ORB-long ORB[574.65,579.00] vol=3.0x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:10:00 | 581.54 | 579.18 | 0.00 | T1 1.5R @ 581.54 |
| Target hit | 2024-07-16 11:55:00 | 580.25 | 581.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2024-07-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 10:55:00 | 577.35 | 582.01 | 0.00 | ORB-short ORB[582.55,587.20] vol=1.9x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 11:20:00 | 574.86 | 580.67 | 0.00 | T1 1.5R @ 574.86 |
| Target hit | 2024-07-18 15:20:00 | 568.00 | 572.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-07-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 10:00:00 | 574.45 | 571.57 | 0.00 | ORB-long ORB[564.80,573.00] vol=1.8x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:20:00 | 577.41 | 573.64 | 0.00 | T1 1.5R @ 577.41 |
| Stop hit — per-position SL triggered | 2024-07-19 10:55:00 | 574.45 | 575.41 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:55:00 | 565.80 | 563.61 | 0.00 | ORB-long ORB[557.95,564.75] vol=3.4x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-07-24 11:30:00 | 563.95 | 564.42 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:40:00 | 589.85 | 582.73 | 0.00 | ORB-long ORB[580.00,586.00] vol=5.2x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 10:45:00 | 592.79 | 585.56 | 0.00 | T1 1.5R @ 592.79 |
| Target hit | 2024-07-29 11:20:00 | 590.05 | 591.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2024-08-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:40:00 | 611.50 | 606.42 | 0.00 | ORB-long ORB[601.20,608.85] vol=3.2x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-08-07 11:05:00 | 609.01 | 607.81 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 09:40:00 | 600.90 | 604.04 | 0.00 | ORB-short ORB[602.15,611.10] vol=2.8x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-08-08 09:55:00 | 603.06 | 602.30 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 11:00:00 | 640.00 | 642.14 | 0.00 | ORB-short ORB[640.90,649.00] vol=1.6x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-08-14 11:10:00 | 641.53 | 642.10 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 641.20 | 644.88 | 0.00 | ORB-short ORB[643.15,648.85] vol=1.9x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:00:00 | 638.11 | 642.62 | 0.00 | T1 1.5R @ 638.11 |
| Target hit | 2024-08-19 15:10:00 | 634.90 | 633.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — SELL (started 2024-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:55:00 | 630.95 | 631.77 | 0.00 | ORB-short ORB[631.20,636.90] vol=6.8x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-08-20 11:10:00 | 632.66 | 631.79 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:00:00 | 648.40 | 638.57 | 0.00 | ORB-long ORB[627.05,633.00] vol=4.3x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-08-21 11:05:00 | 646.22 | 639.28 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 658.30 | 653.10 | 0.00 | ORB-long ORB[647.40,654.90] vol=2.0x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 09:40:00 | 661.96 | 656.82 | 0.00 | T1 1.5R @ 661.96 |
| Stop hit — per-position SL triggered | 2024-08-22 09:50:00 | 658.30 | 657.79 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:40:00 | 653.05 | 651.41 | 0.00 | ORB-long ORB[648.55,652.90] vol=2.0x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 10:00:00 | 655.16 | 652.37 | 0.00 | T1 1.5R @ 655.16 |
| Target hit | 2024-08-26 15:20:00 | 656.45 | 656.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-08-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:35:00 | 660.25 | 658.64 | 0.00 | ORB-long ORB[655.35,659.85] vol=2.9x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 12:30:00 | 662.83 | 659.61 | 0.00 | T1 1.5R @ 662.83 |
| Stop hit — per-position SL triggered | 2024-08-27 13:25:00 | 660.25 | 660.63 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 11:05:00 | 661.65 | 662.29 | 0.00 | ORB-short ORB[662.00,667.65] vol=1.7x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 11:55:00 | 659.12 | 662.12 | 0.00 | T1 1.5R @ 659.12 |
| Target hit | 2024-08-28 14:00:00 | 661.00 | 660.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — SELL (started 2024-09-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 11:10:00 | 648.60 | 651.53 | 0.00 | ORB-short ORB[650.10,654.15] vol=1.7x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:35:00 | 646.11 | 650.70 | 0.00 | T1 1.5R @ 646.11 |
| Stop hit — per-position SL triggered | 2024-09-02 14:10:00 | 648.60 | 647.78 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:55:00 | 647.50 | 643.77 | 0.00 | ORB-long ORB[639.35,646.95] vol=2.1x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-09-06 10:05:00 | 645.70 | 644.37 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:55:00 | 669.00 | 667.47 | 0.00 | ORB-long ORB[661.55,668.30] vol=1.7x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-09-11 10:00:00 | 666.32 | 667.46 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 664.20 | 666.30 | 0.00 | ORB-short ORB[664.25,671.35] vol=1.8x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-09-17 09:55:00 | 666.21 | 665.61 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:40:00 | 685.95 | 682.93 | 0.00 | ORB-long ORB[678.90,685.85] vol=1.9x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-09-19 09:50:00 | 683.49 | 683.67 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 09:30:00 | 710.30 | 707.05 | 0.00 | ORB-long ORB[702.50,708.00] vol=1.8x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-09-23 09:35:00 | 707.86 | 707.62 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:45:00 | 711.80 | 708.67 | 0.00 | ORB-long ORB[699.95,709.80] vol=3.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-09-24 11:10:00 | 710.24 | 709.44 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:45:00 | 700.30 | 701.19 | 0.00 | ORB-short ORB[700.50,704.75] vol=2.9x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:05:00 | 697.87 | 700.34 | 0.00 | T1 1.5R @ 697.87 |
| Stop hit — per-position SL triggered | 2024-09-26 10:45:00 | 700.30 | 699.68 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:20:00 | 619.25 | 626.03 | 0.00 | ORB-short ORB[625.70,634.60] vol=1.6x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:35:00 | 615.03 | 623.98 | 0.00 | T1 1.5R @ 615.03 |
| Stop hit — per-position SL triggered | 2024-10-07 10:40:00 | 619.25 | 623.40 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 11:10:00 | 625.15 | 621.47 | 0.00 | ORB-long ORB[617.95,623.00] vol=1.6x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-10-11 11:20:00 | 623.75 | 621.88 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:45:00 | 616.60 | 621.55 | 0.00 | ORB-short ORB[623.55,632.70] vol=5.6x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:25:00 | 614.13 | 620.97 | 0.00 | T1 1.5R @ 614.13 |
| Stop hit — per-position SL triggered | 2024-10-14 12:50:00 | 616.60 | 620.08 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:05:00 | 627.65 | 624.64 | 0.00 | ORB-long ORB[619.90,627.45] vol=2.1x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-10-16 10:35:00 | 625.74 | 624.45 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:25:00 | 614.65 | 618.25 | 0.00 | ORB-short ORB[619.50,624.20] vol=2.8x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-10-17 10:45:00 | 616.45 | 617.81 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:40:00 | 618.75 | 612.25 | 0.00 | ORB-long ORB[604.30,611.80] vol=1.6x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-10-18 10:55:00 | 616.40 | 612.86 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 09:55:00 | 628.85 | 624.84 | 0.00 | ORB-long ORB[621.40,626.00] vol=2.1x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-10-21 10:10:00 | 626.38 | 626.70 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 10:55:00 | 600.25 | 603.59 | 0.00 | ORB-short ORB[601.70,607.75] vol=2.0x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 11:30:00 | 597.46 | 601.90 | 0.00 | T1 1.5R @ 597.46 |
| Stop hit — per-position SL triggered | 2024-10-24 12:25:00 | 600.25 | 600.36 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 11:15:00 | 574.10 | 578.43 | 0.00 | ORB-short ORB[581.05,588.70] vol=1.5x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-10-29 11:40:00 | 575.93 | 577.52 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-10-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:50:00 | 581.00 | 576.79 | 0.00 | ORB-long ORB[570.30,578.25] vol=2.8x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 11:40:00 | 584.58 | 578.78 | 0.00 | T1 1.5R @ 584.58 |
| Stop hit — per-position SL triggered | 2024-10-30 13:15:00 | 581.00 | 580.31 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-11-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:45:00 | 567.05 | 571.24 | 0.00 | ORB-short ORB[570.00,577.55] vol=2.3x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 12:20:00 | 563.76 | 568.14 | 0.00 | T1 1.5R @ 563.76 |
| Target hit | 2024-11-04 14:35:00 | 564.95 | 563.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — BUY (started 2024-11-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 10:45:00 | 583.30 | 579.50 | 0.00 | ORB-long ORB[567.10,575.90] vol=2.6x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 11:55:00 | 587.47 | 581.78 | 0.00 | T1 1.5R @ 587.47 |
| Stop hit — per-position SL triggered | 2024-11-05 12:05:00 | 583.30 | 581.96 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:45:00 | 602.85 | 600.16 | 0.00 | ORB-long ORB[594.45,600.00] vol=2.5x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 600.86 | 600.27 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-11-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:45:00 | 612.50 | 607.20 | 0.00 | ORB-long ORB[601.90,610.00] vol=1.8x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-11-11 10:55:00 | 609.77 | 607.41 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:55:00 | 619.50 | 614.62 | 0.00 | ORB-long ORB[604.55,610.95] vol=1.6x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-11-19 11:10:00 | 617.80 | 615.25 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-11-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 11:05:00 | 615.35 | 612.46 | 0.00 | ORB-long ORB[606.00,614.85] vol=2.9x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-11-21 11:35:00 | 613.47 | 613.13 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-11-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:55:00 | 644.50 | 641.06 | 0.00 | ORB-long ORB[634.40,640.70] vol=1.8x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:35:00 | 647.28 | 642.48 | 0.00 | T1 1.5R @ 647.28 |
| Target hit | 2024-11-26 15:05:00 | 647.35 | 647.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — BUY (started 2024-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 11:00:00 | 652.90 | 650.41 | 0.00 | ORB-long ORB[644.15,651.80] vol=10.6x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-11-27 11:05:00 | 651.09 | 650.45 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:50:00 | 657.15 | 655.16 | 0.00 | ORB-long ORB[651.15,657.00] vol=2.0x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-11-28 10:00:00 | 655.39 | 655.17 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-11-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:45:00 | 647.95 | 643.52 | 0.00 | ORB-long ORB[639.60,646.10] vol=3.3x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-11-29 11:20:00 | 646.08 | 645.85 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-12-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:40:00 | 648.75 | 643.92 | 0.00 | ORB-long ORB[639.65,645.55] vol=1.5x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 11:05:00 | 651.19 | 645.75 | 0.00 | T1 1.5R @ 651.19 |
| Target hit | 2024-12-02 15:20:00 | 651.35 | 649.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2024-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:35:00 | 660.00 | 657.84 | 0.00 | ORB-long ORB[652.85,658.90] vol=2.4x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-12-04 11:00:00 | 658.05 | 658.38 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:10:00 | 665.35 | 662.43 | 0.00 | ORB-long ORB[660.00,663.70] vol=1.6x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:50:00 | 668.66 | 664.05 | 0.00 | T1 1.5R @ 668.66 |
| Target hit | 2024-12-06 15:20:00 | 693.10 | 683.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2024-12-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:45:00 | 688.80 | 687.68 | 0.00 | ORB-long ORB[681.30,688.30] vol=1.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-12-16 11:10:00 | 686.61 | 687.70 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:00:00 | 687.80 | 679.83 | 0.00 | ORB-long ORB[669.45,679.20] vol=2.0x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:55:00 | 691.56 | 684.01 | 0.00 | T1 1.5R @ 691.56 |
| Stop hit — per-position SL triggered | 2024-12-23 13:00:00 | 687.80 | 685.66 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-12-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:35:00 | 719.80 | 714.02 | 0.00 | ORB-long ORB[707.45,717.95] vol=4.5x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-12-27 09:40:00 | 716.11 | 714.86 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:45:00 | 716.30 | 710.35 | 0.00 | ORB-long ORB[705.10,710.95] vol=1.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-12-30 10:50:00 | 714.17 | 710.76 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-12-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 11:05:00 | 719.60 | 713.51 | 0.00 | ORB-long ORB[708.85,714.25] vol=1.7x ATR=2.48 |
| Stop hit — per-position SL triggered | 2024-12-31 11:45:00 | 717.12 | 715.44 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:25:00 | 724.75 | 720.84 | 0.00 | ORB-long ORB[715.00,721.70] vol=3.2x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-01-01 10:45:00 | 722.66 | 721.63 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:55:00 | 745.00 | 737.82 | 0.00 | ORB-long ORB[733.15,741.70] vol=1.7x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:20:00 | 748.91 | 740.15 | 0.00 | T1 1.5R @ 748.91 |
| Target hit | 2025-01-02 15:20:00 | 756.00 | 748.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 763.00 | 759.83 | 0.00 | ORB-long ORB[752.60,760.00] vol=3.4x ATR=2.59 |
| Stop hit — per-position SL triggered | 2025-01-03 09:35:00 | 760.41 | 760.42 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:35:00 | 733.60 | 738.61 | 0.00 | ORB-short ORB[736.80,746.00] vol=1.9x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-01-10 09:40:00 | 735.95 | 738.21 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 700.00 | 703.26 | 0.00 | ORB-short ORB[703.00,712.35] vol=1.6x ATR=2.29 |
| Stop hit — per-position SL triggered | 2025-01-15 10:10:00 | 702.29 | 701.28 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 11:05:00 | 700.15 | 704.84 | 0.00 | ORB-short ORB[703.00,713.45] vol=1.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:45:00 | 697.42 | 703.86 | 0.00 | T1 1.5R @ 697.42 |
| Target hit | 2025-01-16 15:20:00 | 690.60 | 692.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:50:00 | 698.80 | 692.71 | 0.00 | ORB-long ORB[691.40,698.40] vol=2.2x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-01-20 11:05:00 | 696.45 | 693.23 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-01-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:30:00 | 657.85 | 661.27 | 0.00 | ORB-short ORB[659.45,668.05] vol=1.7x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:50:00 | 654.00 | 659.21 | 0.00 | T1 1.5R @ 654.00 |
| Target hit | 2025-01-27 15:20:00 | 637.45 | 644.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2025-01-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:40:00 | 674.35 | 668.66 | 0.00 | ORB-long ORB[660.55,669.70] vol=1.7x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:40:00 | 679.68 | 672.88 | 0.00 | T1 1.5R @ 679.68 |
| Target hit | 2025-01-29 15:20:00 | 689.75 | 680.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — SELL (started 2025-01-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 11:00:00 | 681.25 | 686.52 | 0.00 | ORB-short ORB[682.55,686.95] vol=2.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-01-30 11:10:00 | 683.53 | 686.22 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:00:00 | 713.10 | 718.84 | 0.00 | ORB-short ORB[718.40,726.40] vol=1.7x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-02-06 11:45:00 | 715.33 | 717.15 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-11 09:35:00 | 683.85 | 680.52 | 0.00 | ORB-long ORB[675.55,681.95] vol=2.2x ATR=2.67 |
| Stop hit — per-position SL triggered | 2025-02-11 09:40:00 | 681.18 | 680.60 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 10:40:00 | 684.25 | 677.59 | 0.00 | ORB-long ORB[673.50,683.10] vol=2.0x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-02-18 11:20:00 | 681.57 | 680.74 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:45:00 | 711.30 | 708.45 | 0.00 | ORB-long ORB[703.30,710.35] vol=2.1x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-02-20 11:50:00 | 708.93 | 709.72 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-03-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:10:00 | 635.75 | 630.98 | 0.00 | ORB-long ORB[626.05,633.60] vol=3.0x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:50:00 | 639.50 | 633.68 | 0.00 | T1 1.5R @ 639.50 |
| Stop hit — per-position SL triggered | 2025-03-19 11:25:00 | 635.75 | 634.51 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:00:00 | 629.30 | 629.25 | 0.00 | ORB-long ORB[620.00,629.00] vol=2.3x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-03-20 11:25:00 | 626.62 | 629.19 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-03-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:00:00 | 637.85 | 632.41 | 0.00 | ORB-long ORB[629.50,634.80] vol=1.5x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 10:05:00 | 641.24 | 633.84 | 0.00 | T1 1.5R @ 641.24 |
| Stop hit — per-position SL triggered | 2025-03-24 10:10:00 | 637.85 | 634.09 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:30:00 | 668.45 | 671.24 | 0.00 | ORB-short ORB[669.20,675.00] vol=1.6x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-03-26 09:35:00 | 671.06 | 671.06 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-03-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 10:55:00 | 666.90 | 670.39 | 0.00 | ORB-short ORB[671.10,679.00] vol=3.2x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 11:10:00 | 663.86 | 669.32 | 0.00 | T1 1.5R @ 663.86 |
| Stop hit — per-position SL triggered | 2025-03-28 14:40:00 | 666.90 | 665.70 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 674.40 | 669.72 | 0.00 | ORB-long ORB[662.05,671.50] vol=2.3x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-04-02 09:35:00 | 671.42 | 670.32 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-04-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 09:40:00 | 685.55 | 681.26 | 0.00 | ORB-long ORB[675.15,682.40] vol=1.6x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-04-03 09:45:00 | 682.50 | 681.32 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2025-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:00:00 | 695.55 | 700.15 | 0.00 | ORB-short ORB[700.05,706.45] vol=1.7x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:10:00 | 692.37 | 699.42 | 0.00 | T1 1.5R @ 692.37 |
| Stop hit — per-position SL triggered | 2025-04-23 10:25:00 | 695.55 | 697.28 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 11:15:00 | 707.25 | 711.50 | 0.00 | ORB-short ORB[709.05,718.45] vol=2.5x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 12:15:00 | 703.96 | 710.21 | 0.00 | T1 1.5R @ 703.96 |
| Stop hit — per-position SL triggered | 2025-04-24 13:10:00 | 707.25 | 709.63 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2025-04-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:05:00 | 701.25 | 707.34 | 0.00 | ORB-short ORB[712.00,719.80] vol=3.4x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-04-25 10:10:00 | 703.83 | 707.06 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:30:00 | 716.55 | 712.05 | 0.00 | ORB-long ORB[705.75,715.65] vol=1.8x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:50:00 | 720.24 | 716.05 | 0.00 | T1 1.5R @ 720.24 |
| Target hit | 2025-04-30 11:25:00 | 719.25 | 720.07 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 11:15:00 | 464.15 | 2024-05-16 11:20:00 | 465.82 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-21 09:40:00 | 465.65 | 2024-05-21 10:05:00 | 463.77 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-21 09:40:00 | 465.65 | 2024-05-21 10:10:00 | 465.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-30 09:40:00 | 521.95 | 2024-05-30 09:55:00 | 519.55 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-06-21 10:55:00 | 542.50 | 2024-06-21 11:05:00 | 540.79 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-24 10:45:00 | 552.00 | 2024-06-24 12:25:00 | 555.96 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-06-24 10:45:00 | 552.00 | 2024-06-24 15:20:00 | 568.10 | TARGET_HIT | 0.50 | 2.92% |
| BUY | retest1 | 2024-06-25 09:35:00 | 583.30 | 2024-06-25 09:40:00 | 580.32 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-06-27 10:40:00 | 560.30 | 2024-06-27 10:45:00 | 558.33 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-28 10:40:00 | 560.30 | 2024-06-28 11:20:00 | 563.55 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-06-28 10:40:00 | 560.30 | 2024-06-28 15:20:00 | 562.30 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2024-07-01 09:40:00 | 574.20 | 2024-07-01 09:45:00 | 571.77 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-02 10:35:00 | 565.80 | 2024-07-02 10:40:00 | 567.54 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-08 10:00:00 | 575.90 | 2024-07-08 10:20:00 | 574.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-10 10:20:00 | 571.85 | 2024-07-10 10:35:00 | 569.40 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-10 10:20:00 | 571.85 | 2024-07-10 10:55:00 | 571.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 11:05:00 | 574.20 | 2024-07-11 11:40:00 | 575.94 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-12 10:35:00 | 576.10 | 2024-07-12 10:55:00 | 577.92 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-15 10:50:00 | 576.20 | 2024-07-15 11:45:00 | 577.91 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-16 10:00:00 | 579.15 | 2024-07-16 10:10:00 | 581.54 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-16 10:00:00 | 579.15 | 2024-07-16 11:55:00 | 580.25 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2024-07-18 10:55:00 | 577.35 | 2024-07-18 11:20:00 | 574.86 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-18 10:55:00 | 577.35 | 2024-07-18 15:20:00 | 568.00 | TARGET_HIT | 0.50 | 1.62% |
| BUY | retest1 | 2024-07-19 10:00:00 | 574.45 | 2024-07-19 10:20:00 | 577.41 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-07-19 10:00:00 | 574.45 | 2024-07-19 10:55:00 | 574.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 10:55:00 | 565.80 | 2024-07-24 11:30:00 | 563.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-29 10:40:00 | 589.85 | 2024-07-29 10:45:00 | 592.79 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-29 10:40:00 | 589.85 | 2024-07-29 11:20:00 | 590.05 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2024-08-07 10:40:00 | 611.50 | 2024-08-07 11:05:00 | 609.01 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-08-08 09:40:00 | 600.90 | 2024-08-08 09:55:00 | 603.06 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-14 11:00:00 | 640.00 | 2024-08-14 11:10:00 | 641.53 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-19 09:35:00 | 641.20 | 2024-08-19 10:00:00 | 638.11 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-08-19 09:35:00 | 641.20 | 2024-08-19 15:10:00 | 634.90 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2024-08-20 10:55:00 | 630.95 | 2024-08-20 11:10:00 | 632.66 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-21 11:00:00 | 648.40 | 2024-08-21 11:05:00 | 646.22 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-22 09:30:00 | 658.30 | 2024-08-22 09:40:00 | 661.96 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-22 09:30:00 | 658.30 | 2024-08-22 09:50:00 | 658.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 09:40:00 | 653.05 | 2024-08-26 10:00:00 | 655.16 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-26 09:40:00 | 653.05 | 2024-08-26 15:20:00 | 656.45 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2024-08-27 10:35:00 | 660.25 | 2024-08-27 12:30:00 | 662.83 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-08-27 10:35:00 | 660.25 | 2024-08-27 13:25:00 | 660.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 11:05:00 | 661.65 | 2024-08-28 11:55:00 | 659.12 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-28 11:05:00 | 661.65 | 2024-08-28 14:00:00 | 661.00 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-09-02 11:10:00 | 648.60 | 2024-09-02 11:35:00 | 646.11 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-02 11:10:00 | 648.60 | 2024-09-02 14:10:00 | 648.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-06 09:55:00 | 647.50 | 2024-09-06 10:05:00 | 645.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-11 09:55:00 | 669.00 | 2024-09-11 10:00:00 | 666.32 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-17 09:40:00 | 664.20 | 2024-09-17 09:55:00 | 666.21 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-19 09:40:00 | 685.95 | 2024-09-19 09:50:00 | 683.49 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-23 09:30:00 | 710.30 | 2024-09-23 09:35:00 | 707.86 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-24 10:45:00 | 711.80 | 2024-09-24 11:10:00 | 710.24 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-26 09:45:00 | 700.30 | 2024-09-26 10:05:00 | 697.87 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-09-26 09:45:00 | 700.30 | 2024-09-26 10:45:00 | 700.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:20:00 | 619.25 | 2024-10-07 10:35:00 | 615.03 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-10-07 10:20:00 | 619.25 | 2024-10-07 10:40:00 | 619.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 11:10:00 | 625.15 | 2024-10-11 11:20:00 | 623.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-14 10:45:00 | 616.60 | 2024-10-14 11:25:00 | 614.13 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-14 10:45:00 | 616.60 | 2024-10-14 12:50:00 | 616.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 10:05:00 | 627.65 | 2024-10-16 10:35:00 | 625.74 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-17 10:25:00 | 614.65 | 2024-10-17 10:45:00 | 616.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-18 10:40:00 | 618.75 | 2024-10-18 10:55:00 | 616.40 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-21 09:55:00 | 628.85 | 2024-10-21 10:10:00 | 626.38 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-24 10:55:00 | 600.25 | 2024-10-24 11:30:00 | 597.46 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-24 10:55:00 | 600.25 | 2024-10-24 12:25:00 | 600.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 11:15:00 | 574.10 | 2024-10-29 11:40:00 | 575.93 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-30 09:50:00 | 581.00 | 2024-10-30 11:40:00 | 584.58 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-30 09:50:00 | 581.00 | 2024-10-30 13:15:00 | 581.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-04 10:45:00 | 567.05 | 2024-11-04 12:20:00 | 563.76 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-11-04 10:45:00 | 567.05 | 2024-11-04 14:35:00 | 564.95 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2024-11-05 10:45:00 | 583.30 | 2024-11-05 11:55:00 | 587.47 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-11-05 10:45:00 | 583.30 | 2024-11-05 12:05:00 | 583.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-08 09:45:00 | 602.85 | 2024-11-08 09:50:00 | 600.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-11 10:45:00 | 612.50 | 2024-11-11 10:55:00 | 609.77 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-11-19 10:55:00 | 619.50 | 2024-11-19 11:10:00 | 617.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-21 11:05:00 | 615.35 | 2024-11-21 11:35:00 | 613.47 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-26 10:55:00 | 644.50 | 2024-11-26 11:35:00 | 647.28 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-11-26 10:55:00 | 644.50 | 2024-11-26 15:05:00 | 647.35 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2024-11-27 11:00:00 | 652.90 | 2024-11-27 11:05:00 | 651.09 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-28 09:50:00 | 657.15 | 2024-11-28 10:00:00 | 655.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-29 10:45:00 | 647.95 | 2024-11-29 11:20:00 | 646.08 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-02 10:40:00 | 648.75 | 2024-12-02 11:05:00 | 651.19 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-02 10:40:00 | 648.75 | 2024-12-02 15:20:00 | 651.35 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2024-12-04 10:35:00 | 660.00 | 2024-12-04 11:00:00 | 658.05 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-06 10:10:00 | 665.35 | 2024-12-06 10:50:00 | 668.66 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-12-06 10:10:00 | 665.35 | 2024-12-06 15:20:00 | 693.10 | TARGET_HIT | 0.50 | 4.17% |
| BUY | retest1 | 2024-12-16 10:45:00 | 688.80 | 2024-12-16 11:10:00 | 686.61 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-23 11:00:00 | 687.80 | 2024-12-23 11:55:00 | 691.56 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-12-23 11:00:00 | 687.80 | 2024-12-23 13:00:00 | 687.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 09:35:00 | 719.80 | 2024-12-27 09:40:00 | 716.11 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-12-30 10:45:00 | 716.30 | 2024-12-30 10:50:00 | 714.17 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-31 11:05:00 | 719.60 | 2024-12-31 11:45:00 | 717.12 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-01 10:25:00 | 724.75 | 2025-01-01 10:45:00 | 722.66 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-02 10:55:00 | 745.00 | 2025-01-02 11:20:00 | 748.91 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-01-02 10:55:00 | 745.00 | 2025-01-02 15:20:00 | 756.00 | TARGET_HIT | 0.50 | 1.48% |
| BUY | retest1 | 2025-01-03 09:30:00 | 763.00 | 2025-01-03 09:35:00 | 760.41 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-10 09:35:00 | 733.60 | 2025-01-10 09:40:00 | 735.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-15 09:30:00 | 700.00 | 2025-01-15 10:10:00 | 702.29 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-16 11:05:00 | 700.15 | 2025-01-16 11:45:00 | 697.42 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-16 11:05:00 | 700.15 | 2025-01-16 15:20:00 | 690.60 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2025-01-20 10:50:00 | 698.80 | 2025-01-20 11:05:00 | 696.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-27 09:30:00 | 657.85 | 2025-01-27 09:50:00 | 654.00 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-27 09:30:00 | 657.85 | 2025-01-27 15:20:00 | 637.45 | TARGET_HIT | 0.50 | 3.10% |
| BUY | retest1 | 2025-01-29 09:40:00 | 674.35 | 2025-01-29 10:40:00 | 679.68 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-01-29 09:40:00 | 674.35 | 2025-01-29 15:20:00 | 689.75 | TARGET_HIT | 0.50 | 2.28% |
| SELL | retest1 | 2025-01-30 11:00:00 | 681.25 | 2025-01-30 11:10:00 | 683.53 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-06 11:00:00 | 713.10 | 2025-02-06 11:45:00 | 715.33 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-11 09:35:00 | 683.85 | 2025-02-11 09:40:00 | 681.18 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-18 10:40:00 | 684.25 | 2025-02-18 11:20:00 | 681.57 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-20 10:45:00 | 711.30 | 2025-02-20 11:50:00 | 708.93 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-19 10:10:00 | 635.75 | 2025-03-19 10:50:00 | 639.50 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-19 10:10:00 | 635.75 | 2025-03-19 11:25:00 | 635.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-20 11:00:00 | 629.30 | 2025-03-20 11:25:00 | 626.62 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-03-24 10:00:00 | 637.85 | 2025-03-24 10:05:00 | 641.24 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-03-24 10:00:00 | 637.85 | 2025-03-24 10:10:00 | 637.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-26 09:30:00 | 668.45 | 2025-03-26 09:35:00 | 671.06 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-03-28 10:55:00 | 666.90 | 2025-03-28 11:10:00 | 663.86 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-03-28 10:55:00 | 666.90 | 2025-03-28 14:40:00 | 666.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 09:30:00 | 674.40 | 2025-04-02 09:35:00 | 671.42 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-03 09:40:00 | 685.55 | 2025-04-03 09:45:00 | 682.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-04-23 10:00:00 | 695.55 | 2025-04-23 10:10:00 | 692.37 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-04-23 10:00:00 | 695.55 | 2025-04-23 10:25:00 | 695.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-24 11:15:00 | 707.25 | 2025-04-24 12:15:00 | 703.96 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-04-24 11:15:00 | 707.25 | 2025-04-24 13:10:00 | 707.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 10:05:00 | 701.25 | 2025-04-25 10:10:00 | 703.83 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-30 09:30:00 | 716.55 | 2025-04-30 09:50:00 | 720.24 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-30 09:30:00 | 716.55 | 2025-04-30 11:25:00 | 719.25 | TARGET_HIT | 0.50 | 0.38% |
