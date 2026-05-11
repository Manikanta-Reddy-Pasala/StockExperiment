# Dabur India Ltd. (DABUR)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-10-01 15:25:00 (25855 bars)
- **Last close:** 618.80
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
| ENTRY1 | 106 |
| ENTRY2 | 0 |
| PARTIAL | 46 |
| TARGET_HIT | 20 |
| STOP_HIT | 86 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 152 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 86
- **Target hits / Stop hits / Partials:** 20 / 86 / 46
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 15.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 26 | 44.8% | 9 | 32 | 17 | 0.14% | 8.4% |
| BUY @ 2nd Alert (retest1) | 58 | 26 | 44.8% | 9 | 32 | 17 | 0.14% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 94 | 40 | 42.6% | 11 | 54 | 29 | 0.08% | 7.6% |
| SELL @ 2nd Alert (retest1) | 94 | 40 | 42.6% | 11 | 54 | 29 | 0.08% | 7.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 152 | 66 | 43.4% | 20 | 86 | 46 | 0.11% | 16.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 11:05:00 | 523.35 | 519.16 | 0.00 | ORB-long ORB[513.65,519.00] vol=1.6x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 11:15:00 | 525.26 | 519.90 | 0.00 | T1 1.5R @ 525.26 |
| Target hit | 2023-05-15 15:20:00 | 533.05 | 527.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2023-05-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:25:00 | 525.00 | 527.71 | 0.00 | ORB-short ORB[525.50,530.00] vol=2.2x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 10:30:00 | 523.17 | 527.45 | 0.00 | T1 1.5R @ 523.17 |
| Target hit | 2023-05-17 15:20:00 | 523.80 | 525.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2023-05-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:35:00 | 523.60 | 524.50 | 0.00 | ORB-short ORB[524.20,527.45] vol=2.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-05-18 11:45:00 | 524.69 | 524.28 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-24 10:55:00 | 523.10 | 524.41 | 0.00 | ORB-short ORB[523.45,526.80] vol=1.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2023-05-24 11:25:00 | 523.80 | 524.02 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 10:55:00 | 531.05 | 529.65 | 0.00 | ORB-long ORB[525.55,529.40] vol=2.0x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-05-25 11:15:00 | 529.93 | 529.69 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 09:45:00 | 551.10 | 549.06 | 0.00 | ORB-long ORB[546.70,550.40] vol=3.1x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-30 09:50:00 | 552.81 | 549.58 | 0.00 | T1 1.5R @ 552.81 |
| Stop hit — per-position SL triggered | 2023-05-30 10:00:00 | 551.10 | 549.83 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 11:05:00 | 556.55 | 559.31 | 0.00 | ORB-short ORB[557.55,563.00] vol=1.9x ATR=0.73 |
| Stop hit — per-position SL triggered | 2023-06-05 11:20:00 | 557.28 | 559.12 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 09:45:00 | 548.35 | 549.59 | 0.00 | ORB-short ORB[549.05,554.35] vol=1.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 10:00:00 | 546.41 | 548.58 | 0.00 | T1 1.5R @ 546.41 |
| Target hit | 2023-06-06 12:50:00 | 545.95 | 545.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2023-06-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:20:00 | 555.00 | 554.12 | 0.00 | ORB-long ORB[546.20,553.25] vol=11.0x ATR=1.17 |
| Stop hit — per-position SL triggered | 2023-06-07 10:45:00 | 553.83 | 554.39 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 10:40:00 | 555.55 | 556.85 | 0.00 | ORB-short ORB[556.40,560.00] vol=1.9x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 11:30:00 | 554.07 | 556.37 | 0.00 | T1 1.5R @ 554.07 |
| Target hit | 2023-06-08 15:20:00 | 552.50 | 554.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2023-06-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 10:55:00 | 549.00 | 551.96 | 0.00 | ORB-short ORB[552.80,556.50] vol=1.9x ATR=1.24 |
| Stop hit — per-position SL triggered | 2023-06-09 11:10:00 | 550.24 | 550.87 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:05:00 | 551.55 | 549.54 | 0.00 | ORB-long ORB[546.65,549.70] vol=1.5x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 10:15:00 | 553.32 | 549.90 | 0.00 | T1 1.5R @ 553.32 |
| Target hit | 2023-06-12 13:05:00 | 552.60 | 552.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2023-06-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:00:00 | 556.10 | 553.78 | 0.00 | ORB-long ORB[551.30,555.50] vol=4.0x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:30:00 | 558.23 | 555.07 | 0.00 | T1 1.5R @ 558.23 |
| Target hit | 2023-06-13 15:20:00 | 559.55 | 558.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2023-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 09:30:00 | 558.30 | 559.80 | 0.00 | ORB-short ORB[558.85,561.50] vol=1.7x ATR=1.28 |
| Stop hit — per-position SL triggered | 2023-06-14 09:40:00 | 559.58 | 559.77 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 10:00:00 | 567.30 | 564.73 | 0.00 | ORB-long ORB[561.55,564.60] vol=1.5x ATR=1.00 |
| Stop hit — per-position SL triggered | 2023-06-15 10:05:00 | 566.30 | 564.86 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-06-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 11:00:00 | 567.00 | 570.25 | 0.00 | ORB-short ORB[570.00,573.00] vol=1.5x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 11:30:00 | 565.42 | 569.83 | 0.00 | T1 1.5R @ 565.42 |
| Stop hit — per-position SL triggered | 2023-06-16 11:40:00 | 567.00 | 569.67 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 11:15:00 | 568.00 | 566.13 | 0.00 | ORB-long ORB[564.00,567.75] vol=10.3x ATR=0.93 |
| Stop hit — per-position SL triggered | 2023-06-20 11:20:00 | 567.07 | 566.34 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-06-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:35:00 | 569.45 | 572.46 | 0.00 | ORB-short ORB[570.45,575.50] vol=2.0x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 10:45:00 | 567.70 | 571.64 | 0.00 | T1 1.5R @ 567.70 |
| Stop hit — per-position SL triggered | 2023-06-21 12:20:00 | 569.45 | 570.42 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-06-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 11:05:00 | 565.80 | 563.82 | 0.00 | ORB-long ORB[560.15,565.45] vol=5.2x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 11:45:00 | 567.55 | 564.75 | 0.00 | T1 1.5R @ 567.55 |
| Target hit | 2023-06-26 15:20:00 | 569.75 | 567.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2023-06-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:35:00 | 576.75 | 574.64 | 0.00 | ORB-long ORB[571.00,573.90] vol=3.0x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-06-30 09:40:00 | 575.17 | 574.90 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:50:00 | 584.80 | 581.87 | 0.00 | ORB-long ORB[576.60,583.95] vol=1.6x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:25:00 | 586.87 | 584.05 | 0.00 | T1 1.5R @ 586.87 |
| Target hit | 2023-07-05 11:25:00 | 588.50 | 588.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2023-07-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 11:00:00 | 583.10 | 581.47 | 0.00 | ORB-long ORB[576.80,580.45] vol=3.9x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 11:40:00 | 585.07 | 581.92 | 0.00 | T1 1.5R @ 585.07 |
| Target hit | 2023-07-14 15:20:00 | 587.60 | 584.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2023-07-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 10:20:00 | 577.00 | 577.06 | 0.00 | ORB-short ORB[577.55,582.20] vol=7.5x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 10:35:00 | 575.13 | 576.84 | 0.00 | T1 1.5R @ 575.13 |
| Target hit | 2023-07-18 14:35:00 | 576.70 | 576.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:15:00 | 576.00 | 574.11 | 0.00 | ORB-long ORB[570.05,574.95] vol=1.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-07-20 11:15:00 | 574.91 | 574.63 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 09:50:00 | 574.90 | 571.77 | 0.00 | ORB-long ORB[566.60,574.10] vol=3.3x ATR=2.04 |
| Stop hit — per-position SL triggered | 2023-07-21 11:25:00 | 572.86 | 573.62 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-07-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 10:05:00 | 559.50 | 561.63 | 0.00 | ORB-short ORB[560.00,566.75] vol=3.5x ATR=1.77 |
| Stop hit — per-position SL triggered | 2023-07-24 10:15:00 | 561.27 | 561.60 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:15:00 | 572.25 | 570.21 | 0.00 | ORB-long ORB[568.20,571.90] vol=1.9x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 10:25:00 | 573.70 | 571.05 | 0.00 | T1 1.5R @ 573.70 |
| Stop hit — per-position SL triggered | 2023-07-26 11:25:00 | 572.25 | 572.37 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-07-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 10:55:00 | 575.70 | 574.94 | 0.00 | ORB-long ORB[570.00,575.65] vol=2.3x ATR=1.29 |
| Stop hit — per-position SL triggered | 2023-07-28 11:40:00 | 574.41 | 575.17 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:15:00 | 562.40 | 563.61 | 0.00 | ORB-short ORB[564.00,567.10] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2023-08-08 10:40:00 | 563.58 | 562.97 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 10:45:00 | 565.40 | 563.39 | 0.00 | ORB-long ORB[561.05,564.85] vol=1.9x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 10:55:00 | 567.12 | 564.29 | 0.00 | T1 1.5R @ 567.12 |
| Target hit | 2023-08-09 15:20:00 | 572.00 | 568.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2023-08-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 09:35:00 | 579.15 | 576.57 | 0.00 | ORB-long ORB[570.80,577.95] vol=2.3x ATR=1.54 |
| Stop hit — per-position SL triggered | 2023-08-10 09:45:00 | 577.61 | 576.77 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 10:00:00 | 565.60 | 560.58 | 0.00 | ORB-long ORB[556.00,563.00] vol=3.0x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-08-14 10:05:00 | 563.79 | 561.16 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 10:20:00 | 569.05 | 567.66 | 0.00 | ORB-long ORB[565.25,568.00] vol=1.6x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 10:30:00 | 570.34 | 568.31 | 0.00 | T1 1.5R @ 570.34 |
| Target hit | 2023-08-22 15:20:00 | 575.05 | 573.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2023-08-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 10:45:00 | 567.85 | 570.76 | 0.00 | ORB-short ORB[572.05,575.80] vol=1.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-08-23 10:50:00 | 568.77 | 570.59 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-08-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:05:00 | 558.40 | 562.47 | 0.00 | ORB-short ORB[561.10,568.65] vol=3.5x ATR=1.36 |
| Stop hit — per-position SL triggered | 2023-08-25 10:20:00 | 559.76 | 560.96 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-08-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-28 10:00:00 | 554.85 | 556.60 | 0.00 | ORB-short ORB[555.00,559.85] vol=1.8x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 10:35:00 | 553.40 | 555.90 | 0.00 | T1 1.5R @ 553.40 |
| Stop hit — per-position SL triggered | 2023-08-28 10:55:00 | 554.85 | 555.51 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-08-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 10:00:00 | 549.10 | 550.92 | 0.00 | ORB-short ORB[550.00,553.90] vol=1.7x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 10:35:00 | 547.87 | 549.96 | 0.00 | T1 1.5R @ 547.87 |
| Stop hit — per-position SL triggered | 2023-08-29 12:10:00 | 549.10 | 548.89 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-08-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:10:00 | 549.50 | 552.71 | 0.00 | ORB-short ORB[552.15,555.30] vol=1.6x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 11:25:00 | 548.10 | 552.25 | 0.00 | T1 1.5R @ 548.10 |
| Stop hit — per-position SL triggered | 2023-08-31 12:10:00 | 549.50 | 551.06 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:15:00 | 562.00 | 560.80 | 0.00 | ORB-long ORB[558.00,560.75] vol=3.0x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-09-05 11:35:00 | 561.04 | 561.34 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 10:15:00 | 568.45 | 566.80 | 0.00 | ORB-long ORB[563.55,566.50] vol=1.7x ATR=0.75 |
| Stop hit — per-position SL triggered | 2023-09-08 10:30:00 | 567.70 | 567.15 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 09:35:00 | 556.75 | 559.14 | 0.00 | ORB-short ORB[557.50,565.25] vol=1.9x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 10:00:00 | 554.51 | 557.59 | 0.00 | T1 1.5R @ 554.51 |
| Stop hit — per-position SL triggered | 2023-09-20 10:30:00 | 556.75 | 557.23 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-09-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 10:00:00 | 556.15 | 558.33 | 0.00 | ORB-short ORB[556.30,561.00] vol=2.4x ATR=1.34 |
| Stop hit — per-position SL triggered | 2023-09-21 10:05:00 | 557.49 | 558.16 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-09-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 10:00:00 | 556.50 | 555.87 | 0.00 | ORB-long ORB[553.70,556.00] vol=1.8x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-09-27 10:05:00 | 555.60 | 555.80 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-09-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 10:40:00 | 556.20 | 557.48 | 0.00 | ORB-short ORB[556.40,561.85] vol=4.9x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-09-28 10:45:00 | 557.07 | 557.41 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:30:00 | 547.65 | 551.70 | 0.00 | ORB-short ORB[553.00,558.00] vol=1.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-10-05 10:45:00 | 548.77 | 551.26 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 10:20:00 | 548.20 | 549.44 | 0.00 | ORB-short ORB[548.55,551.90] vol=1.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2023-10-06 10:40:00 | 549.31 | 549.28 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 09:30:00 | 542.20 | 543.24 | 0.00 | ORB-short ORB[542.55,546.20] vol=1.6x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 09:55:00 | 540.92 | 542.56 | 0.00 | T1 1.5R @ 540.92 |
| Target hit | 2023-10-12 11:50:00 | 541.10 | 540.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 48 — SELL (started 2023-10-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 11:00:00 | 538.35 | 540.39 | 0.00 | ORB-short ORB[538.85,542.25] vol=1.9x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 11:05:00 | 537.20 | 540.19 | 0.00 | T1 1.5R @ 537.20 |
| Stop hit — per-position SL triggered | 2023-10-13 11:35:00 | 538.35 | 539.82 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:40:00 | 540.35 | 538.82 | 0.00 | ORB-long ORB[536.00,538.30] vol=4.9x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-10-17 09:50:00 | 539.43 | 538.97 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 11:15:00 | 509.60 | 513.13 | 0.00 | ORB-short ORB[513.10,517.90] vol=3.4x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 11:25:00 | 508.39 | 512.88 | 0.00 | T1 1.5R @ 508.39 |
| Stop hit — per-position SL triggered | 2023-10-26 11:40:00 | 509.60 | 512.52 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 10:05:00 | 527.05 | 524.73 | 0.00 | ORB-long ORB[522.25,525.00] vol=1.8x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 10:15:00 | 528.67 | 525.39 | 0.00 | T1 1.5R @ 528.67 |
| Stop hit — per-position SL triggered | 2023-10-31 10:45:00 | 527.05 | 526.14 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:50:00 | 525.75 | 528.15 | 0.00 | ORB-short ORB[528.15,532.00] vol=1.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 11:35:00 | 524.37 | 527.44 | 0.00 | T1 1.5R @ 524.37 |
| Target hit | 2023-11-01 15:20:00 | 518.10 | 521.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2023-11-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 11:10:00 | 538.80 | 537.24 | 0.00 | ORB-long ORB[533.35,536.90] vol=2.2x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-11-06 11:35:00 | 537.99 | 537.36 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-11-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 11:05:00 | 536.20 | 537.47 | 0.00 | ORB-short ORB[536.80,539.90] vol=4.7x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 13:20:00 | 534.80 | 536.85 | 0.00 | T1 1.5R @ 534.80 |
| Stop hit — per-position SL triggered | 2023-11-07 14:00:00 | 536.20 | 536.36 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:20:00 | 529.50 | 531.43 | 0.00 | ORB-short ORB[531.15,535.80] vol=4.7x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 11:55:00 | 527.99 | 529.51 | 0.00 | T1 1.5R @ 527.99 |
| Target hit | 2023-11-13 14:00:00 | 528.60 | 528.39 | 0.00 | Trail-exit close>VWAP |

### Cycle 56 — BUY (started 2023-11-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 10:00:00 | 539.55 | 536.81 | 0.00 | ORB-long ORB[531.65,536.70] vol=1.7x ATR=1.18 |
| Stop hit — per-position SL triggered | 2023-11-17 10:25:00 | 538.37 | 537.55 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 09:45:00 | 537.90 | 540.12 | 0.00 | ORB-short ORB[540.25,542.00] vol=3.4x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 10:05:00 | 536.61 | 538.92 | 0.00 | T1 1.5R @ 536.61 |
| Stop hit — per-position SL triggered | 2023-11-23 11:15:00 | 537.90 | 538.07 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 09:40:00 | 536.00 | 535.47 | 0.00 | ORB-long ORB[533.35,535.55] vol=3.8x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-11-30 09:45:00 | 535.15 | 535.46 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 09:35:00 | 552.10 | 553.26 | 0.00 | ORB-short ORB[552.45,554.85] vol=2.2x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 10:00:00 | 550.43 | 552.66 | 0.00 | T1 1.5R @ 550.43 |
| Target hit | 2023-12-06 14:15:00 | 551.55 | 551.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — SELL (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 11:15:00 | 545.00 | 545.41 | 0.00 | ORB-short ORB[546.35,549.35] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2023-12-13 15:00:00 | 545.74 | 545.16 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 11:00:00 | 545.10 | 541.65 | 0.00 | ORB-long ORB[540.35,543.05] vol=1.8x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-12-19 11:10:00 | 544.01 | 541.79 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-12-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 11:00:00 | 545.25 | 547.53 | 0.00 | ORB-short ORB[547.70,551.35] vol=1.5x ATR=0.93 |
| Stop hit — per-position SL triggered | 2023-12-20 11:10:00 | 546.18 | 547.37 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 09:45:00 | 550.50 | 548.16 | 0.00 | ORB-long ORB[543.65,548.45] vol=1.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 10:00:00 | 553.22 | 549.28 | 0.00 | T1 1.5R @ 553.22 |
| Target hit | 2023-12-29 13:15:00 | 554.45 | 554.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — SELL (started 2024-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:40:00 | 555.60 | 557.69 | 0.00 | ORB-short ORB[556.95,562.35] vol=1.8x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 11:20:00 | 553.46 | 557.09 | 0.00 | T1 1.5R @ 553.46 |
| Stop hit — per-position SL triggered | 2024-01-02 14:25:00 | 555.60 | 555.93 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 09:30:00 | 559.65 | 558.20 | 0.00 | ORB-long ORB[555.20,558.20] vol=2.1x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-01-03 09:40:00 | 558.44 | 558.38 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 11:15:00 | 565.10 | 567.80 | 0.00 | ORB-short ORB[566.25,571.65] vol=1.7x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-01-05 11:40:00 | 566.18 | 567.64 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:35:00 | 552.95 | 553.71 | 0.00 | ORB-short ORB[553.00,556.00] vol=1.8x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-01-09 10:30:00 | 554.41 | 553.36 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-01-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 11:05:00 | 542.70 | 546.73 | 0.00 | ORB-short ORB[547.85,551.00] vol=1.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-01-10 11:40:00 | 543.76 | 546.26 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-01-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:40:00 | 555.85 | 553.06 | 0.00 | ORB-long ORB[549.00,552.05] vol=6.5x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-01-11 10:45:00 | 554.75 | 553.20 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 10:50:00 | 549.25 | 550.98 | 0.00 | ORB-short ORB[550.30,552.95] vol=2.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-01-12 11:10:00 | 550.16 | 550.84 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:55:00 | 549.40 | 549.78 | 0.00 | ORB-short ORB[549.70,552.60] vol=5.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-01-15 10:00:00 | 550.55 | 549.86 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:50:00 | 559.50 | 557.53 | 0.00 | ORB-long ORB[552.95,557.00] vol=4.4x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 11:00:00 | 561.19 | 558.19 | 0.00 | T1 1.5R @ 561.19 |
| Stop hit — per-position SL triggered | 2024-01-16 11:15:00 | 559.50 | 558.42 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:35:00 | 549.70 | 553.58 | 0.00 | ORB-short ORB[551.25,557.85] vol=1.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-01-17 11:10:00 | 551.12 | 552.81 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:30:00 | 541.40 | 544.02 | 0.00 | ORB-short ORB[542.05,549.95] vol=1.5x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:50:00 | 538.83 | 542.17 | 0.00 | T1 1.5R @ 538.83 |
| Stop hit — per-position SL triggered | 2024-01-18 10:30:00 | 541.40 | 541.02 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:10:00 | 537.95 | 540.22 | 0.00 | ORB-short ORB[542.05,545.25] vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-01-20 10:25:00 | 539.35 | 540.05 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:35:00 | 535.85 | 538.98 | 0.00 | ORB-short ORB[537.25,544.40] vol=1.6x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 10:50:00 | 533.70 | 537.07 | 0.00 | T1 1.5R @ 533.70 |
| Stop hit — per-position SL triggered | 2024-01-23 11:25:00 | 535.85 | 536.84 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-01-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 11:05:00 | 532.85 | 529.40 | 0.00 | ORB-long ORB[522.20,528.05] vol=3.2x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-01-24 11:25:00 | 531.43 | 530.00 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-01-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:55:00 | 529.30 | 533.53 | 0.00 | ORB-short ORB[534.10,537.55] vol=2.0x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-01-25 11:20:00 | 530.32 | 533.08 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-02 10:15:00 | 549.35 | 552.34 | 0.00 | ORB-short ORB[551.60,559.00] vol=2.0x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 10:30:00 | 546.91 | 551.64 | 0.00 | T1 1.5R @ 546.91 |
| Target hit | 2024-02-02 15:20:00 | 544.55 | 545.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — SELL (started 2024-02-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 10:05:00 | 531.25 | 532.43 | 0.00 | ORB-short ORB[532.15,536.00] vol=1.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-02-07 10:10:00 | 532.15 | 532.40 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 10:50:00 | 531.45 | 533.46 | 0.00 | ORB-short ORB[532.80,536.50] vol=2.0x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 11:05:00 | 529.28 | 532.56 | 0.00 | T1 1.5R @ 529.28 |
| Stop hit — per-position SL triggered | 2024-02-09 11:10:00 | 531.45 | 532.56 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 10:55:00 | 541.35 | 542.63 | 0.00 | ORB-short ORB[541.85,547.35] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-02-15 11:10:00 | 542.32 | 542.49 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-02-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-19 10:30:00 | 542.35 | 543.99 | 0.00 | ORB-short ORB[543.10,545.75] vol=1.8x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 10:50:00 | 540.79 | 543.43 | 0.00 | T1 1.5R @ 540.79 |
| Stop hit — per-position SL triggered | 2024-02-19 11:10:00 | 542.35 | 543.22 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 11:15:00 | 545.10 | 546.22 | 0.00 | ORB-short ORB[546.55,551.20] vol=1.7x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 13:50:00 | 543.74 | 545.52 | 0.00 | T1 1.5R @ 543.74 |
| Stop hit — per-position SL triggered | 2024-02-20 14:25:00 | 545.10 | 545.39 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 11:00:00 | 545.85 | 547.57 | 0.00 | ORB-short ORB[547.10,549.85] vol=3.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-02-21 11:10:00 | 546.70 | 547.53 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 10:35:00 | 540.15 | 541.92 | 0.00 | ORB-short ORB[541.55,544.55] vol=4.0x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 11:45:00 | 538.70 | 541.06 | 0.00 | T1 1.5R @ 538.70 |
| Stop hit — per-position SL triggered | 2024-02-23 12:35:00 | 540.15 | 540.62 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 11:05:00 | 529.80 | 532.15 | 0.00 | ORB-short ORB[531.55,536.00] vol=2.1x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-03-06 11:50:00 | 530.87 | 530.55 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-03-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:30:00 | 536.10 | 533.49 | 0.00 | ORB-long ORB[530.50,534.10] vol=1.7x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-03-07 12:15:00 | 534.95 | 534.13 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:50:00 | 527.80 | 528.90 | 0.00 | ORB-short ORB[528.35,531.40] vol=1.9x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:55:00 | 525.74 | 528.73 | 0.00 | T1 1.5R @ 525.74 |
| Target hit | 2024-03-13 15:20:00 | 520.50 | 521.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 90 — SELL (started 2024-03-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:35:00 | 514.80 | 517.19 | 0.00 | ORB-short ORB[517.55,521.00] vol=2.5x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:45:00 | 512.95 | 516.29 | 0.00 | T1 1.5R @ 512.95 |
| Stop hit — per-position SL triggered | 2024-03-20 10:50:00 | 514.80 | 515.87 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-03-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:50:00 | 523.20 | 521.54 | 0.00 | ORB-long ORB[518.50,523.00] vol=1.9x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 13:00:00 | 524.51 | 522.66 | 0.00 | T1 1.5R @ 524.51 |
| Stop hit — per-position SL triggered | 2024-03-21 13:40:00 | 523.20 | 522.78 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 10:55:00 | 519.75 | 521.73 | 0.00 | ORB-short ORB[521.85,524.40] vol=2.0x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-03-27 11:20:00 | 520.51 | 521.45 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-04-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-03 10:55:00 | 529.30 | 529.85 | 0.00 | ORB-short ORB[529.50,532.40] vol=6.1x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-04-03 11:05:00 | 530.09 | 529.79 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-04-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:25:00 | 526.90 | 527.94 | 0.00 | ORB-short ORB[529.10,531.95] vol=3.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-04-04 10:45:00 | 527.98 | 527.90 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 10:15:00 | 505.95 | 506.75 | 0.00 | ORB-short ORB[506.80,510.00] vol=3.4x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-04-05 11:05:00 | 507.49 | 506.66 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-04-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 10:30:00 | 505.60 | 506.89 | 0.00 | ORB-short ORB[505.80,509.85] vol=1.6x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-04-08 12:05:00 | 506.48 | 506.26 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-04-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 11:10:00 | 507.80 | 506.39 | 0.00 | ORB-long ORB[503.10,506.50] vol=5.0x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-04-09 11:20:00 | 507.00 | 506.38 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-15 09:35:00 | 490.00 | 492.72 | 0.00 | ORB-short ORB[492.00,497.00] vol=2.6x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-04-15 09:45:00 | 491.48 | 492.48 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2024-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:40:00 | 500.35 | 498.03 | 0.00 | ORB-long ORB[492.35,498.85] vol=2.0x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 09:50:00 | 502.15 | 498.93 | 0.00 | T1 1.5R @ 502.15 |
| Stop hit — per-position SL triggered | 2024-04-16 10:00:00 | 500.35 | 499.43 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-04-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:30:00 | 501.80 | 504.43 | 0.00 | ORB-short ORB[503.50,507.25] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-04-18 09:45:00 | 502.98 | 503.68 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:15:00 | 509.05 | 506.72 | 0.00 | ORB-long ORB[505.20,507.80] vol=2.5x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-04-23 10:40:00 | 508.21 | 507.26 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2024-04-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:50:00 | 505.15 | 507.91 | 0.00 | ORB-short ORB[508.70,510.25] vol=2.4x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-04-25 11:00:00 | 505.84 | 507.75 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2024-04-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 10:45:00 | 509.15 | 507.86 | 0.00 | ORB-long ORB[505.30,509.10] vol=1.7x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 10:50:00 | 510.37 | 508.48 | 0.00 | T1 1.5R @ 510.37 |
| Stop hit — per-position SL triggered | 2024-04-26 11:05:00 | 509.15 | 508.63 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 10:40:00 | 509.95 | 510.78 | 0.00 | ORB-short ORB[510.75,512.45] vol=2.9x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 11:30:00 | 508.52 | 510.26 | 0.00 | T1 1.5R @ 508.52 |
| Target hit | 2024-04-29 15:20:00 | 506.90 | 508.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 105 — SELL (started 2024-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 11:10:00 | 527.20 | 530.91 | 0.00 | ORB-short ORB[529.30,535.00] vol=3.1x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-05-06 11:15:00 | 529.25 | 530.75 | 0.00 | SL hit |

### Cycle 106 — BUY (started 2024-05-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 10:00:00 | 559.90 | 554.76 | 0.00 | ORB-long ORB[552.25,556.70] vol=1.8x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:15:00 | 562.80 | 556.04 | 0.00 | T1 1.5R @ 562.80 |
| Stop hit — per-position SL triggered | 2024-05-09 10:25:00 | 559.90 | 557.01 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 11:05:00 | 523.35 | 2023-05-15 11:15:00 | 525.26 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-05-15 11:05:00 | 523.35 | 2023-05-15 15:20:00 | 533.05 | TARGET_HIT | 0.50 | 1.85% |
| SELL | retest1 | 2023-05-17 10:25:00 | 525.00 | 2023-05-17 10:30:00 | 523.17 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-05-17 10:25:00 | 525.00 | 2023-05-17 15:20:00 | 523.80 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2023-05-18 10:35:00 | 523.60 | 2023-05-18 11:45:00 | 524.69 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-05-24 10:55:00 | 523.10 | 2023-05-24 11:25:00 | 523.80 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-05-25 10:55:00 | 531.05 | 2023-05-25 11:15:00 | 529.93 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-05-30 09:45:00 | 551.10 | 2023-05-30 09:50:00 | 552.81 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-05-30 09:45:00 | 551.10 | 2023-05-30 10:00:00 | 551.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-05 11:05:00 | 556.55 | 2023-06-05 11:20:00 | 557.28 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-06-06 09:45:00 | 548.35 | 2023-06-06 10:00:00 | 546.41 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-06-06 09:45:00 | 548.35 | 2023-06-06 12:50:00 | 545.95 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2023-06-07 10:20:00 | 555.00 | 2023-06-07 10:45:00 | 553.83 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-06-08 10:40:00 | 555.55 | 2023-06-08 11:30:00 | 554.07 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-06-08 10:40:00 | 555.55 | 2023-06-08 15:20:00 | 552.50 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2023-06-09 10:55:00 | 549.00 | 2023-06-09 11:10:00 | 550.24 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-12 10:05:00 | 551.55 | 2023-06-12 10:15:00 | 553.32 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-06-12 10:05:00 | 551.55 | 2023-06-12 13:05:00 | 552.60 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2023-06-13 10:00:00 | 556.10 | 2023-06-13 10:30:00 | 558.23 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-13 10:00:00 | 556.10 | 2023-06-13 15:20:00 | 559.55 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2023-06-14 09:30:00 | 558.30 | 2023-06-14 09:40:00 | 559.58 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-15 10:00:00 | 567.30 | 2023-06-15 10:05:00 | 566.30 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-16 11:00:00 | 567.00 | 2023-06-16 11:30:00 | 565.42 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-06-16 11:00:00 | 567.00 | 2023-06-16 11:40:00 | 567.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-20 11:15:00 | 568.00 | 2023-06-20 11:20:00 | 567.07 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-06-21 10:35:00 | 569.45 | 2023-06-21 10:45:00 | 567.70 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-06-21 10:35:00 | 569.45 | 2023-06-21 12:20:00 | 569.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-26 11:05:00 | 565.80 | 2023-06-26 11:45:00 | 567.55 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-06-26 11:05:00 | 565.80 | 2023-06-26 15:20:00 | 569.75 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2023-06-30 09:35:00 | 576.75 | 2023-06-30 09:40:00 | 575.17 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-05 09:50:00 | 584.80 | 2023-07-05 10:25:00 | 586.87 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-05 09:50:00 | 584.80 | 2023-07-05 11:25:00 | 588.50 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2023-07-14 11:00:00 | 583.10 | 2023-07-14 11:40:00 | 585.07 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-07-14 11:00:00 | 583.10 | 2023-07-14 15:20:00 | 587.60 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2023-07-18 10:20:00 | 577.00 | 2023-07-18 10:35:00 | 575.13 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-07-18 10:20:00 | 577.00 | 2023-07-18 14:35:00 | 576.70 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2023-07-20 10:15:00 | 576.00 | 2023-07-20 11:15:00 | 574.91 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-07-21 09:50:00 | 574.90 | 2023-07-21 11:25:00 | 572.86 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-07-24 10:05:00 | 559.50 | 2023-07-24 10:15:00 | 561.27 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-07-26 10:15:00 | 572.25 | 2023-07-26 10:25:00 | 573.70 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-07-26 10:15:00 | 572.25 | 2023-07-26 11:25:00 | 572.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-28 10:55:00 | 575.70 | 2023-07-28 11:40:00 | 574.41 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-08 10:15:00 | 562.40 | 2023-08-08 10:40:00 | 563.58 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-09 10:45:00 | 565.40 | 2023-08-09 10:55:00 | 567.12 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-08-09 10:45:00 | 565.40 | 2023-08-09 15:20:00 | 572.00 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2023-08-10 09:35:00 | 579.15 | 2023-08-10 09:45:00 | 577.61 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-08-14 10:00:00 | 565.60 | 2023-08-14 10:05:00 | 563.79 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-22 10:20:00 | 569.05 | 2023-08-22 10:30:00 | 570.34 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-08-22 10:20:00 | 569.05 | 2023-08-22 15:20:00 | 575.05 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2023-08-23 10:45:00 | 567.85 | 2023-08-23 10:50:00 | 568.77 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-25 10:05:00 | 558.40 | 2023-08-25 10:20:00 | 559.76 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-28 10:00:00 | 554.85 | 2023-08-28 10:35:00 | 553.40 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-08-28 10:00:00 | 554.85 | 2023-08-28 10:55:00 | 554.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-29 10:00:00 | 549.10 | 2023-08-29 10:35:00 | 547.87 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-08-29 10:00:00 | 549.10 | 2023-08-29 12:10:00 | 549.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-31 11:10:00 | 549.50 | 2023-08-31 11:25:00 | 548.10 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-08-31 11:10:00 | 549.50 | 2023-08-31 12:10:00 | 549.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-05 10:15:00 | 562.00 | 2023-09-05 11:35:00 | 561.04 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-09-08 10:15:00 | 568.45 | 2023-09-08 10:30:00 | 567.70 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-09-20 09:35:00 | 556.75 | 2023-09-20 10:00:00 | 554.51 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-09-20 09:35:00 | 556.75 | 2023-09-20 10:30:00 | 556.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-21 10:00:00 | 556.15 | 2023-09-21 10:05:00 | 557.49 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-09-27 10:00:00 | 556.50 | 2023-09-27 10:05:00 | 555.60 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-09-28 10:40:00 | 556.20 | 2023-09-28 10:45:00 | 557.07 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-10-05 10:30:00 | 547.65 | 2023-10-05 10:45:00 | 548.77 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-10-06 10:20:00 | 548.20 | 2023-10-06 10:40:00 | 549.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-10-12 09:30:00 | 542.20 | 2023-10-12 09:55:00 | 540.92 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-10-12 09:30:00 | 542.20 | 2023-10-12 11:50:00 | 541.10 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2023-10-13 11:00:00 | 538.35 | 2023-10-13 11:05:00 | 537.20 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-10-13 11:00:00 | 538.35 | 2023-10-13 11:35:00 | 538.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-17 09:40:00 | 540.35 | 2023-10-17 09:50:00 | 539.43 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-10-26 11:15:00 | 509.60 | 2023-10-26 11:25:00 | 508.39 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-10-26 11:15:00 | 509.60 | 2023-10-26 11:40:00 | 509.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-31 10:05:00 | 527.05 | 2023-10-31 10:15:00 | 528.67 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-10-31 10:05:00 | 527.05 | 2023-10-31 10:45:00 | 527.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-01 10:50:00 | 525.75 | 2023-11-01 11:35:00 | 524.37 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-11-01 10:50:00 | 525.75 | 2023-11-01 15:20:00 | 518.10 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2023-11-06 11:10:00 | 538.80 | 2023-11-06 11:35:00 | 537.99 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-11-07 11:05:00 | 536.20 | 2023-11-07 13:20:00 | 534.80 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-11-07 11:05:00 | 536.20 | 2023-11-07 14:00:00 | 536.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-13 10:20:00 | 529.50 | 2023-11-13 11:55:00 | 527.99 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-11-13 10:20:00 | 529.50 | 2023-11-13 14:00:00 | 528.60 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2023-11-17 10:00:00 | 539.55 | 2023-11-17 10:25:00 | 538.37 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-11-23 09:45:00 | 537.90 | 2023-11-23 10:05:00 | 536.61 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-11-23 09:45:00 | 537.90 | 2023-11-23 11:15:00 | 537.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 09:40:00 | 536.00 | 2023-11-30 09:45:00 | 535.15 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-12-06 09:35:00 | 552.10 | 2023-12-06 10:00:00 | 550.43 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-12-06 09:35:00 | 552.10 | 2023-12-06 14:15:00 | 551.55 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2023-12-13 11:15:00 | 545.00 | 2023-12-13 15:00:00 | 545.74 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-12-19 11:00:00 | 545.10 | 2023-12-19 11:10:00 | 544.01 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-12-20 11:00:00 | 545.25 | 2023-12-20 11:10:00 | 546.18 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-12-29 09:45:00 | 550.50 | 2023-12-29 10:00:00 | 553.22 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-12-29 09:45:00 | 550.50 | 2023-12-29 13:15:00 | 554.45 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2024-01-02 10:40:00 | 555.60 | 2024-01-02 11:20:00 | 553.46 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-01-02 10:40:00 | 555.60 | 2024-01-02 14:25:00 | 555.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-03 09:30:00 | 559.65 | 2024-01-03 09:40:00 | 558.44 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-05 11:15:00 | 565.10 | 2024-01-05 11:40:00 | 566.18 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-01-09 09:35:00 | 552.95 | 2024-01-09 10:30:00 | 554.41 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-01-10 11:05:00 | 542.70 | 2024-01-10 11:40:00 | 543.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-01-11 10:40:00 | 555.85 | 2024-01-11 10:45:00 | 554.75 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-01-12 10:50:00 | 549.25 | 2024-01-12 11:10:00 | 550.16 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-01-15 09:55:00 | 549.40 | 2024-01-15 10:00:00 | 550.55 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-01-16 10:50:00 | 559.50 | 2024-01-16 11:00:00 | 561.19 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-01-16 10:50:00 | 559.50 | 2024-01-16 11:15:00 | 559.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-17 10:35:00 | 549.70 | 2024-01-17 11:10:00 | 551.12 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-01-18 09:30:00 | 541.40 | 2024-01-18 09:50:00 | 538.83 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-01-18 09:30:00 | 541.40 | 2024-01-18 10:30:00 | 541.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 10:10:00 | 537.95 | 2024-01-20 10:25:00 | 539.35 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-01-23 09:35:00 | 535.85 | 2024-01-23 10:50:00 | 533.70 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-01-23 09:35:00 | 535.85 | 2024-01-23 11:25:00 | 535.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-24 11:05:00 | 532.85 | 2024-01-24 11:25:00 | 531.43 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-25 10:55:00 | 529.30 | 2024-01-25 11:20:00 | 530.32 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-02-02 10:15:00 | 549.35 | 2024-02-02 10:30:00 | 546.91 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-02-02 10:15:00 | 549.35 | 2024-02-02 15:20:00 | 544.55 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2024-02-07 10:05:00 | 531.25 | 2024-02-07 10:10:00 | 532.15 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-02-09 10:50:00 | 531.45 | 2024-02-09 11:05:00 | 529.28 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-02-09 10:50:00 | 531.45 | 2024-02-09 11:10:00 | 531.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-15 10:55:00 | 541.35 | 2024-02-15 11:10:00 | 542.32 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-02-19 10:30:00 | 542.35 | 2024-02-19 10:50:00 | 540.79 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-02-19 10:30:00 | 542.35 | 2024-02-19 11:10:00 | 542.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-20 11:15:00 | 545.10 | 2024-02-20 13:50:00 | 543.74 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-02-20 11:15:00 | 545.10 | 2024-02-20 14:25:00 | 545.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-21 11:00:00 | 545.85 | 2024-02-21 11:10:00 | 546.70 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-02-23 10:35:00 | 540.15 | 2024-02-23 11:45:00 | 538.70 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-02-23 10:35:00 | 540.15 | 2024-02-23 12:35:00 | 540.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-06 11:05:00 | 529.80 | 2024-03-06 11:50:00 | 530.87 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-03-07 10:30:00 | 536.10 | 2024-03-07 12:15:00 | 534.95 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-03-13 09:50:00 | 527.80 | 2024-03-13 09:55:00 | 525.74 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-03-13 09:50:00 | 527.80 | 2024-03-13 15:20:00 | 520.50 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2024-03-20 10:35:00 | 514.80 | 2024-03-20 10:45:00 | 512.95 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-03-20 10:35:00 | 514.80 | 2024-03-20 10:50:00 | 514.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 10:50:00 | 523.20 | 2024-03-21 13:00:00 | 524.51 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-03-21 10:50:00 | 523.20 | 2024-03-21 13:40:00 | 523.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-27 10:55:00 | 519.75 | 2024-03-27 11:20:00 | 520.51 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-04-03 10:55:00 | 529.30 | 2024-04-03 11:05:00 | 530.09 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-04-04 10:25:00 | 526.90 | 2024-04-04 10:45:00 | 527.98 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-04-05 10:15:00 | 505.95 | 2024-04-05 11:05:00 | 507.49 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-04-08 10:30:00 | 505.60 | 2024-04-08 12:05:00 | 506.48 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-04-09 11:10:00 | 507.80 | 2024-04-09 11:20:00 | 507.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-04-15 09:35:00 | 490.00 | 2024-04-15 09:45:00 | 491.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-16 09:40:00 | 500.35 | 2024-04-16 09:50:00 | 502.15 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-04-16 09:40:00 | 500.35 | 2024-04-16 10:00:00 | 500.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-18 09:30:00 | 501.80 | 2024-04-18 09:45:00 | 502.98 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-23 10:15:00 | 509.05 | 2024-04-23 10:40:00 | 508.21 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-04-25 10:50:00 | 505.15 | 2024-04-25 11:00:00 | 505.84 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-04-26 10:45:00 | 509.15 | 2024-04-26 10:50:00 | 510.37 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-04-26 10:45:00 | 509.15 | 2024-04-26 11:05:00 | 509.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-29 10:40:00 | 509.95 | 2024-04-29 11:30:00 | 508.52 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-04-29 10:40:00 | 509.95 | 2024-04-29 15:20:00 | 506.90 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-06 11:10:00 | 527.20 | 2024-05-06 11:15:00 | 529.25 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-09 10:00:00 | 559.90 | 2024-05-09 10:15:00 | 562.80 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-05-09 10:00:00 | 559.90 | 2024-05-09 10:25:00 | 559.90 | STOP_HIT | 0.50 | 0.00% |
