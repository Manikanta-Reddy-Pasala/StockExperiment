# Brigade Enterprises Ltd. (BRIGADE)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2023-08-09 15:25:00 (4725 bars)
- **Last close:** 564.00
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 16
- **Target hits / Stop hits / Partials:** 4 / 16 / 11
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 5.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.05% | 0.6% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.05% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 10 | 55.6% | 3 | 8 | 7 | 0.27% | 4.8% |
| SELL @ 2nd Alert (retest1) | 18 | 10 | 55.6% | 3 | 8 | 7 | 0.27% | 4.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 31 | 15 | 48.4% | 4 | 16 | 11 | 0.17% | 5.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-22 10:30:00 | 532.00 | 536.95 | 0.00 | ORB-short ORB[539.05,542.05] vol=2.2x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-22 10:50:00 | 529.20 | 535.03 | 0.00 | T1 1.5R @ 529.20 |
| Target hit | 2023-05-22 15:20:00 | 527.00 | 531.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2023-05-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 10:10:00 | 534.55 | 531.46 | 0.00 | ORB-long ORB[524.00,531.15] vol=1.8x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 10:45:00 | 536.96 | 532.20 | 0.00 | T1 1.5R @ 536.96 |
| Stop hit — per-position SL triggered | 2023-05-23 11:35:00 | 534.55 | 533.00 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 11:00:00 | 536.10 | 532.05 | 0.00 | ORB-long ORB[530.50,535.50] vol=2.9x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-24 11:25:00 | 538.72 | 533.35 | 0.00 | T1 1.5R @ 538.72 |
| Stop hit — per-position SL triggered | 2023-05-24 11:30:00 | 536.10 | 533.57 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-07 10:05:00 | 569.45 | 571.47 | 0.00 | ORB-short ORB[573.30,576.55] vol=1.6x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 10:45:00 | 566.81 | 570.57 | 0.00 | T1 1.5R @ 566.81 |
| Target hit | 2023-06-07 14:25:00 | 568.50 | 567.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2023-06-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:05:00 | 579.30 | 580.18 | 0.00 | ORB-short ORB[579.65,584.65] vol=2.2x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-06-14 11:40:00 | 580.52 | 580.30 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 10:45:00 | 584.00 | 579.43 | 0.00 | ORB-long ORB[574.55,583.00] vol=1.8x ATR=2.28 |
| Stop hit — per-position SL triggered | 2023-06-15 11:50:00 | 581.72 | 581.62 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 11:05:00 | 583.50 | 589.47 | 0.00 | ORB-short ORB[589.75,594.95] vol=3.0x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 11:10:00 | 580.91 | 588.40 | 0.00 | T1 1.5R @ 580.91 |
| Stop hit — per-position SL triggered | 2023-06-16 11:25:00 | 583.50 | 587.97 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:30:00 | 591.80 | 587.27 | 0.00 | ORB-long ORB[581.85,588.00] vol=3.0x ATR=2.48 |
| Stop hit — per-position SL triggered | 2023-06-19 09:35:00 | 589.32 | 587.73 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 09:40:00 | 572.45 | 575.22 | 0.00 | ORB-short ORB[575.20,579.95] vol=4.2x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 10:05:00 | 568.52 | 573.40 | 0.00 | T1 1.5R @ 568.52 |
| Stop hit — per-position SL triggered | 2023-06-23 10:20:00 | 572.45 | 573.29 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 11:00:00 | 565.10 | 563.49 | 0.00 | ORB-long ORB[562.60,565.00] vol=1.7x ATR=1.70 |
| Stop hit — per-position SL triggered | 2023-06-26 11:10:00 | 563.40 | 563.68 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-07-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 10:30:00 | 562.25 | 565.70 | 0.00 | ORB-short ORB[562.50,570.00] vol=5.0x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:40:00 | 559.61 | 565.21 | 0.00 | T1 1.5R @ 559.61 |
| Stop hit — per-position SL triggered | 2023-07-05 11:00:00 | 562.25 | 564.16 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-07-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 10:20:00 | 552.35 | 559.93 | 0.00 | ORB-short ORB[558.95,564.85] vol=2.6x ATR=2.37 |
| Stop hit — per-position SL triggered | 2023-07-06 10:25:00 | 554.72 | 559.66 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:50:00 | 566.35 | 562.71 | 0.00 | ORB-long ORB[558.70,562.95] vol=3.1x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 10:05:00 | 569.36 | 564.84 | 0.00 | T1 1.5R @ 569.36 |
| Stop hit — per-position SL triggered | 2023-07-12 10:15:00 | 566.35 | 565.54 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 10:35:00 | 580.70 | 575.97 | 0.00 | ORB-long ORB[571.90,577.70] vol=4.6x ATR=2.00 |
| Stop hit — per-position SL triggered | 2023-07-17 10:40:00 | 578.70 | 576.43 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:40:00 | 586.90 | 590.10 | 0.00 | ORB-short ORB[587.55,594.10] vol=1.7x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:45:00 | 583.51 | 588.37 | 0.00 | T1 1.5R @ 583.51 |
| Stop hit — per-position SL triggered | 2023-07-18 11:50:00 | 586.90 | 588.32 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 10:25:00 | 570.35 | 572.53 | 0.00 | ORB-short ORB[571.25,578.40] vol=2.5x ATR=2.25 |
| Stop hit — per-position SL triggered | 2023-07-24 10:35:00 | 572.60 | 572.52 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:35:00 | 588.70 | 585.65 | 0.00 | ORB-long ORB[580.10,585.55] vol=1.7x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 11:55:00 | 592.61 | 588.69 | 0.00 | T1 1.5R @ 592.61 |
| Target hit | 2023-07-31 15:10:00 | 590.65 | 590.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2023-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:30:00 | 585.55 | 589.66 | 0.00 | ORB-short ORB[590.15,595.35] vol=2.4x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-08-01 10:35:00 | 587.37 | 589.58 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 11:15:00 | 580.65 | 586.20 | 0.00 | ORB-short ORB[582.30,587.45] vol=7.8x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 13:00:00 | 577.18 | 583.82 | 0.00 | T1 1.5R @ 577.18 |
| Target hit | 2023-08-02 15:20:00 | 573.20 | 579.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 10:15:00 | 596.10 | 590.46 | 0.00 | ORB-long ORB[583.90,592.15] vol=2.4x ATR=2.54 |
| Stop hit — per-position SL triggered | 2023-08-08 10:20:00 | 593.56 | 591.06 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-22 10:30:00 | 532.00 | 2023-05-22 10:50:00 | 529.20 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-05-22 10:30:00 | 532.00 | 2023-05-22 15:20:00 | 527.00 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2023-05-23 10:10:00 | 534.55 | 2023-05-23 10:45:00 | 536.96 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-05-23 10:10:00 | 534.55 | 2023-05-23 11:35:00 | 534.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-24 11:00:00 | 536.10 | 2023-05-24 11:25:00 | 538.72 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-05-24 11:00:00 | 536.10 | 2023-05-24 11:30:00 | 536.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-07 10:05:00 | 569.45 | 2023-06-07 10:45:00 | 566.81 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-06-07 10:05:00 | 569.45 | 2023-06-07 14:25:00 | 568.50 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2023-06-14 11:05:00 | 579.30 | 2023-06-14 11:40:00 | 580.52 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-15 10:45:00 | 584.00 | 2023-06-15 11:50:00 | 581.72 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-06-16 11:05:00 | 583.50 | 2023-06-16 11:10:00 | 580.91 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-06-16 11:05:00 | 583.50 | 2023-06-16 11:25:00 | 583.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-19 09:30:00 | 591.80 | 2023-06-19 09:35:00 | 589.32 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-06-23 09:40:00 | 572.45 | 2023-06-23 10:05:00 | 568.52 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2023-06-23 09:40:00 | 572.45 | 2023-06-23 10:20:00 | 572.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-26 11:00:00 | 565.10 | 2023-06-26 11:10:00 | 563.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-07-05 10:30:00 | 562.25 | 2023-07-05 10:40:00 | 559.61 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-07-05 10:30:00 | 562.25 | 2023-07-05 11:00:00 | 562.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-06 10:20:00 | 552.35 | 2023-07-06 10:25:00 | 554.72 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-07-12 09:50:00 | 566.35 | 2023-07-12 10:05:00 | 569.36 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-07-12 09:50:00 | 566.35 | 2023-07-12 10:15:00 | 566.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-17 10:35:00 | 580.70 | 2023-07-17 10:40:00 | 578.70 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-18 09:40:00 | 586.90 | 2023-07-18 11:45:00 | 583.51 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2023-07-18 09:40:00 | 586.90 | 2023-07-18 11:50:00 | 586.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-24 10:25:00 | 570.35 | 2023-07-24 10:35:00 | 572.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-07-31 09:35:00 | 588.70 | 2023-07-31 11:55:00 | 592.61 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2023-07-31 09:35:00 | 588.70 | 2023-07-31 15:10:00 | 590.65 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2023-08-01 10:30:00 | 585.55 | 2023-08-01 10:35:00 | 587.37 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-08-02 11:15:00 | 580.65 | 2023-08-02 13:00:00 | 577.18 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2023-08-02 11:15:00 | 580.65 | 2023-08-02 15:20:00 | 573.20 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2023-08-08 10:15:00 | 596.10 | 2023-08-08 10:20:00 | 593.56 | STOP_HIT | 1.00 | -0.43% |
