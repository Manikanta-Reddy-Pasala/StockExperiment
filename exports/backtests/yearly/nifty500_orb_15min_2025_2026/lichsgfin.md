# LIC Housing Finance Ltd. (LICHSGFIN)

## Backtest Summary

- **Window:** 2025-10-08 09:15:00 → 2026-05-08 15:25:00 (7663 bars)
- **Last close:** 581.85
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
| ENTRY1 | 43 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 11 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 32
- **Target hits / Stop hits / Partials:** 11 / 32 / 20
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 8.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 21 | 58.3% | 8 | 15 | 13 | 0.18% | 6.6% |
| BUY @ 2nd Alert (retest1) | 36 | 21 | 58.3% | 8 | 15 | 13 | 0.18% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 10 | 37.0% | 3 | 17 | 7 | 0.05% | 1.4% |
| SELL @ 2nd Alert (retest1) | 27 | 10 | 37.0% | 3 | 17 | 7 | 0.05% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 63 | 31 | 49.2% | 11 | 32 | 20 | 0.13% | 8.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 567.50 | 569.03 | 0.00 | ORB-short ORB[567.90,570.60] vol=6.2x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:35:00 | 565.29 | 568.65 | 0.00 | T1 1.5R @ 565.29 |
| Target hit | 2025-10-08 15:20:00 | 562.50 | 566.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-10-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:45:00 | 567.90 | 564.07 | 0.00 | ORB-long ORB[561.10,565.40] vol=1.8x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 11:05:00 | 569.99 | 565.38 | 0.00 | T1 1.5R @ 569.99 |
| Stop hit — per-position SL triggered | 2025-10-10 11:50:00 | 567.90 | 566.96 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-10-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:50:00 | 568.25 | 566.52 | 0.00 | ORB-long ORB[563.05,567.40] vol=1.6x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:40:00 | 569.91 | 567.88 | 0.00 | T1 1.5R @ 569.91 |
| Target hit | 2025-10-13 15:20:00 | 572.15 | 569.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-10-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:05:00 | 564.20 | 568.28 | 0.00 | ORB-short ORB[571.60,573.90] vol=3.8x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 562.64 | 567.53 | 0.00 | T1 1.5R @ 562.64 |
| Stop hit — per-position SL triggered | 2025-10-14 14:40:00 | 564.20 | 564.23 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-10-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:55:00 | 571.70 | 569.09 | 0.00 | ORB-long ORB[565.80,570.00] vol=1.6x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:55:00 | 572.88 | 569.73 | 0.00 | T1 1.5R @ 572.88 |
| Stop hit — per-position SL triggered | 2025-10-15 12:55:00 | 571.70 | 570.00 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-10-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 11:10:00 | 569.35 | 569.51 | 0.00 | ORB-short ORB[570.00,572.00] vol=2.2x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 570.29 | 569.55 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:50:00 | 565.90 | 567.05 | 0.00 | ORB-short ORB[568.10,570.15] vol=7.6x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-10-17 09:55:00 | 566.97 | 566.90 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-10-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:45:00 | 576.00 | 572.50 | 0.00 | ORB-long ORB[569.65,575.70] vol=2.5x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 12:00:00 | 578.33 | 574.65 | 0.00 | T1 1.5R @ 578.33 |
| Target hit | 2025-10-20 14:15:00 | 576.50 | 576.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:45:00 | 578.60 | 576.32 | 0.00 | ORB-long ORB[573.00,577.90] vol=1.8x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-10-23 10:35:00 | 577.25 | 577.01 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:30:00 | 585.85 | 583.51 | 0.00 | ORB-long ORB[581.10,584.20] vol=2.0x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-10-24 09:45:00 | 584.59 | 583.86 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-10-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:00:00 | 585.15 | 583.66 | 0.00 | ORB-long ORB[578.40,583.70] vol=1.8x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-10-27 10:35:00 | 583.87 | 583.93 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:55:00 | 589.10 | 586.93 | 0.00 | ORB-long ORB[584.65,588.80] vol=1.8x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:25:00 | 591.02 | 587.59 | 0.00 | T1 1.5R @ 591.02 |
| Target hit | 2025-10-29 15:20:00 | 594.45 | 590.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2025-10-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:50:00 | 570.85 | 571.91 | 0.00 | ORB-short ORB[571.00,574.50] vol=1.6x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 572.27 | 571.69 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-11-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:55:00 | 572.50 | 574.57 | 0.00 | ORB-short ORB[575.65,578.00] vol=1.6x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:30:00 | 570.89 | 574.21 | 0.00 | T1 1.5R @ 570.89 |
| Stop hit — per-position SL triggered | 2025-11-04 12:00:00 | 572.50 | 573.94 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 566.50 | 569.00 | 0.00 | ORB-short ORB[567.30,573.60] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-11-06 09:35:00 | 567.58 | 568.87 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-11-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:50:00 | 567.10 | 568.92 | 0.00 | ORB-short ORB[568.35,573.40] vol=1.5x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:10:00 | 565.37 | 567.65 | 0.00 | T1 1.5R @ 565.37 |
| Stop hit — per-position SL triggered | 2025-11-11 11:50:00 | 567.10 | 567.47 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-11-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 11:10:00 | 570.55 | 571.40 | 0.00 | ORB-short ORB[570.85,573.55] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-11-13 11:35:00 | 571.23 | 571.31 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-11-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:00:00 | 560.80 | 563.39 | 0.00 | ORB-short ORB[563.10,567.00] vol=1.7x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 12:55:00 | 559.20 | 561.25 | 0.00 | T1 1.5R @ 559.20 |
| Target hit | 2025-11-20 15:20:00 | 553.80 | 558.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2025-11-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 09:55:00 | 548.35 | 547.30 | 0.00 | ORB-long ORB[544.05,547.45] vol=9.2x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 10:00:00 | 549.83 | 547.39 | 0.00 | T1 1.5R @ 549.83 |
| Stop hit — per-position SL triggered | 2025-11-25 10:55:00 | 548.35 | 548.79 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-11-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:35:00 | 557.35 | 552.62 | 0.00 | ORB-long ORB[547.85,552.25] vol=2.0x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-11-26 10:55:00 | 556.14 | 553.18 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-11-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:55:00 | 552.45 | 554.77 | 0.00 | ORB-short ORB[555.60,557.55] vol=1.6x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:30:00 | 550.83 | 554.36 | 0.00 | T1 1.5R @ 550.83 |
| Target hit | 2025-11-27 15:20:00 | 550.80 | 551.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-12-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 09:40:00 | 551.10 | 550.04 | 0.00 | ORB-long ORB[548.60,551.00] vol=1.8x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 10:05:00 | 552.62 | 551.13 | 0.00 | T1 1.5R @ 552.62 |
| Target hit | 2025-12-02 14:35:00 | 552.65 | 553.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-12-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:30:00 | 551.00 | 548.54 | 0.00 | ORB-long ORB[545.65,549.40] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-12-04 10:50:00 | 550.00 | 548.72 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-12-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:25:00 | 542.15 | 546.19 | 0.00 | ORB-short ORB[546.50,549.05] vol=2.0x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 14:45:00 | 539.68 | 542.86 | 0.00 | T1 1.5R @ 539.68 |
| Stop hit — per-position SL triggered | 2025-12-05 15:05:00 | 542.15 | 542.44 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 534.90 | 537.62 | 0.00 | ORB-short ORB[535.60,540.55] vol=1.5x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-01-08 09:35:00 | 536.14 | 537.49 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:15:00 | 532.80 | 528.25 | 0.00 | ORB-long ORB[524.30,530.00] vol=1.9x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 10:20:00 | 535.72 | 530.34 | 0.00 | T1 1.5R @ 535.72 |
| Target hit | 2026-01-09 11:40:00 | 535.95 | 536.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2026-01-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:30:00 | 534.05 | 532.38 | 0.00 | ORB-long ORB[528.50,532.50] vol=2.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-01-19 10:35:00 | 532.71 | 532.39 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2026-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:45:00 | 525.55 | 529.94 | 0.00 | ORB-short ORB[530.40,534.85] vol=1.9x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-01-20 09:50:00 | 527.08 | 529.70 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2026-01-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:40:00 | 517.65 | 522.60 | 0.00 | ORB-short ORB[519.80,526.30] vol=1.8x ATR=1.93 |
| Stop hit — per-position SL triggered | 2026-01-21 12:10:00 | 519.58 | 520.22 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2026-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 11:10:00 | 523.90 | 521.25 | 0.00 | ORB-long ORB[518.30,523.65] vol=2.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-01-29 11:55:00 | 522.60 | 521.66 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2026-01-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:35:00 | 525.60 | 522.17 | 0.00 | ORB-long ORB[517.50,522.60] vol=1.7x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 11:25:00 | 528.05 | 523.46 | 0.00 | T1 1.5R @ 528.05 |
| Stop hit — per-position SL triggered | 2026-01-30 12:55:00 | 525.60 | 524.66 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2026-02-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:20:00 | 516.15 | 519.85 | 0.00 | ORB-short ORB[518.00,523.30] vol=1.9x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-02-09 10:25:00 | 517.60 | 519.66 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 509.70 | 511.94 | 0.00 | ORB-short ORB[510.75,515.35] vol=1.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 511.04 | 511.37 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 515.20 | 512.42 | 0.00 | ORB-long ORB[509.65,511.85] vol=2.0x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:20:00 | 516.90 | 513.72 | 0.00 | T1 1.5R @ 516.90 |
| Target hit | 2026-02-17 15:20:00 | 518.50 | 517.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 520.90 | 519.96 | 0.00 | ORB-long ORB[517.55,520.65] vol=1.7x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 519.84 | 520.19 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 531.50 | 530.13 | 0.00 | ORB-long ORB[525.60,531.30] vol=6.7x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-02-19 09:50:00 | 529.92 | 530.30 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 527.50 | 525.06 | 0.00 | ORB-long ORB[517.55,521.25] vol=5.9x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-02-20 12:55:00 | 525.56 | 526.82 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 525.45 | 523.59 | 0.00 | ORB-long ORB[521.70,525.05] vol=2.4x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:35:00 | 527.47 | 524.82 | 0.00 | T1 1.5R @ 527.47 |
| Target hit | 2026-02-24 13:30:00 | 527.55 | 527.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 540.10 | 537.18 | 0.00 | ORB-long ORB[531.60,538.00] vol=1.8x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:20:00 | 542.24 | 537.73 | 0.00 | T1 1.5R @ 542.24 |
| Target hit | 2026-02-25 12:45:00 | 543.70 | 543.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 538.65 | 541.64 | 0.00 | ORB-short ORB[541.40,546.25] vol=1.6x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 540.42 | 541.50 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 498.20 | 494.69 | 0.00 | ORB-long ORB[495.00,497.95] vol=1.5x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:25:00 | 500.76 | 495.57 | 0.00 | T1 1.5R @ 500.76 |
| Stop hit — per-position SL triggered | 2026-03-13 12:35:00 | 498.20 | 496.83 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-03-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:20:00 | 494.00 | 494.64 | 0.00 | ORB-short ORB[494.50,499.95] vol=2.3x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-03-19 10:35:00 | 495.74 | 494.61 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-05-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:35:00 | 581.00 | 583.52 | 0.00 | ORB-short ORB[583.00,587.50] vol=2.2x ATR=1.64 |
| Stop hit — per-position SL triggered | 2026-05-08 10:40:00 | 582.64 | 583.49 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-10-08 11:00:00 | 567.50 | 2025-10-08 11:35:00 | 565.29 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-08 11:00:00 | 567.50 | 2025-10-08 15:20:00 | 562.50 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2025-10-10 10:45:00 | 567.90 | 2025-10-10 11:05:00 | 569.99 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-10 10:45:00 | 567.90 | 2025-10-10 11:50:00 | 567.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-13 09:50:00 | 568.25 | 2025-10-13 10:40:00 | 569.91 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-10-13 09:50:00 | 568.25 | 2025-10-13 15:20:00 | 572.15 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-10-14 11:05:00 | 564.20 | 2025-10-14 11:15:00 | 562.64 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-10-14 11:05:00 | 564.20 | 2025-10-14 14:40:00 | 564.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 10:55:00 | 571.70 | 2025-10-15 11:55:00 | 572.88 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-10-15 10:55:00 | 571.70 | 2025-10-15 12:55:00 | 571.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-16 11:10:00 | 569.35 | 2025-10-16 11:15:00 | 570.29 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-17 09:50:00 | 565.90 | 2025-10-17 09:55:00 | 566.97 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-20 10:45:00 | 576.00 | 2025-10-20 12:00:00 | 578.33 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-20 10:45:00 | 576.00 | 2025-10-20 14:15:00 | 576.50 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-10-23 09:45:00 | 578.60 | 2025-10-23 10:35:00 | 577.25 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-24 09:30:00 | 585.85 | 2025-10-24 09:45:00 | 584.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-27 10:00:00 | 585.15 | 2025-10-27 10:35:00 | 583.87 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-29 10:55:00 | 589.10 | 2025-10-29 11:25:00 | 591.02 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-29 10:55:00 | 589.10 | 2025-10-29 15:20:00 | 594.45 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2025-10-31 10:50:00 | 570.85 | 2025-10-31 11:15:00 | 572.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-04 10:55:00 | 572.50 | 2025-11-04 11:30:00 | 570.89 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-04 10:55:00 | 572.50 | 2025-11-04 12:00:00 | 572.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 09:30:00 | 566.50 | 2025-11-06 09:35:00 | 567.58 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-11 09:50:00 | 567.10 | 2025-11-11 11:10:00 | 565.37 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-11-11 09:50:00 | 567.10 | 2025-11-11 11:50:00 | 567.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-13 11:10:00 | 570.55 | 2025-11-13 11:35:00 | 571.23 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-11-20 10:00:00 | 560.80 | 2025-11-20 12:55:00 | 559.20 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-20 10:00:00 | 560.80 | 2025-11-20 15:20:00 | 553.80 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2025-11-25 09:55:00 | 548.35 | 2025-11-25 10:00:00 | 549.83 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-11-25 09:55:00 | 548.35 | 2025-11-25 10:55:00 | 548.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 10:35:00 | 557.35 | 2025-11-26 10:55:00 | 556.14 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-27 10:55:00 | 552.45 | 2025-11-27 11:30:00 | 550.83 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-27 10:55:00 | 552.45 | 2025-11-27 15:20:00 | 550.80 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-02 09:40:00 | 551.10 | 2025-12-02 10:05:00 | 552.62 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-12-02 09:40:00 | 551.10 | 2025-12-02 14:35:00 | 552.65 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-12-04 10:30:00 | 551.00 | 2025-12-04 10:50:00 | 550.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-05 10:25:00 | 542.15 | 2025-12-05 14:45:00 | 539.68 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-12-05 10:25:00 | 542.15 | 2025-12-05 15:05:00 | 542.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 09:30:00 | 534.90 | 2026-01-08 09:35:00 | 536.14 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-09 10:15:00 | 532.80 | 2026-01-09 10:20:00 | 535.72 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-01-09 10:15:00 | 532.80 | 2026-01-09 11:40:00 | 535.95 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2026-01-19 10:30:00 | 534.05 | 2026-01-19 10:35:00 | 532.71 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-20 09:45:00 | 525.55 | 2026-01-20 09:50:00 | 527.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-21 10:40:00 | 517.65 | 2026-01-21 12:10:00 | 519.58 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-01-29 11:10:00 | 523.90 | 2026-01-29 11:55:00 | 522.60 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-30 10:35:00 | 525.60 | 2026-01-30 11:25:00 | 528.05 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-01-30 10:35:00 | 525.60 | 2026-01-30 12:55:00 | 525.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-09 10:20:00 | 516.15 | 2026-02-09 10:25:00 | 517.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-13 09:35:00 | 509.70 | 2026-02-13 09:40:00 | 511.04 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-17 10:25:00 | 515.20 | 2026-02-17 11:20:00 | 516.90 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-17 10:25:00 | 515.20 | 2026-02-17 15:20:00 | 518.50 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-18 09:35:00 | 520.90 | 2026-02-18 09:50:00 | 519.84 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-19 09:30:00 | 531.50 | 2026-02-19 09:50:00 | 529.92 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-20 09:45:00 | 527.50 | 2026-02-20 12:55:00 | 525.56 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-24 09:35:00 | 525.45 | 2026-02-24 10:35:00 | 527.47 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-24 09:35:00 | 525.45 | 2026-02-24 13:30:00 | 527.55 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-25 10:15:00 | 540.10 | 2026-02-25 10:20:00 | 542.24 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-25 10:15:00 | 540.10 | 2026-02-25 12:45:00 | 543.70 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-02-27 10:20:00 | 538.65 | 2026-02-27 10:35:00 | 540.42 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-13 10:50:00 | 498.20 | 2026-03-13 11:25:00 | 500.76 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-13 10:50:00 | 498.20 | 2026-03-13 12:35:00 | 498.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 10:20:00 | 494.00 | 2026-03-19 10:35:00 | 495.74 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-05-08 10:35:00 | 581.00 | 2026-05-08 10:40:00 | 582.64 | STOP_HIT | 1.00 | -0.28% |
