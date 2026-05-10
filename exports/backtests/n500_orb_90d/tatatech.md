# Tata Technologies Ltd. (TATATECH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 632.05
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 8
- **Target hits / Stop hits / Partials:** 4 / 8 / 7
- **Avg / median % per leg:** 0.27% / 0.35%
- **Sum % (uncompounded):** 5.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 3 | 4 | 5 | 0.27% | 3.2% |
| BUY @ 2nd Alert (retest1) | 12 | 8 | 66.7% | 3 | 4 | 5 | 0.27% | 3.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.28% | 2.0% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.28% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 11 | 57.9% | 4 | 8 | 7 | 0.27% | 5.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 626.10 | 623.65 | 0.00 | ORB-long ORB[617.70,626.00] vol=1.9x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 628.32 | 625.38 | 0.00 | T1 1.5R @ 628.32 |
| Target hit | 2026-02-10 15:05:00 | 630.70 | 631.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 606.45 | 603.07 | 0.00 | ORB-long ORB[596.10,603.75] vol=1.5x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:30:00 | 609.53 | 605.34 | 0.00 | T1 1.5R @ 609.53 |
| Target hit | 2026-02-17 13:50:00 | 607.05 | 607.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 604.80 | 600.18 | 0.00 | ORB-long ORB[593.80,602.80] vol=2.3x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:15:00 | 608.56 | 602.82 | 0.00 | T1 1.5R @ 608.56 |
| Target hit | 2026-02-20 15:20:00 | 607.00 | 606.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 601.45 | 606.11 | 0.00 | ORB-short ORB[606.15,610.25] vol=1.6x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-02-23 12:55:00 | 602.95 | 604.86 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 587.40 | 584.46 | 0.00 | ORB-long ORB[578.50,585.75] vol=2.0x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 585.84 | 584.67 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 544.25 | 546.10 | 0.00 | ORB-short ORB[545.10,551.00] vol=2.0x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:05:00 | 541.52 | 545.17 | 0.00 | T1 1.5R @ 541.52 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 544.25 | 544.73 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:25:00 | 532.20 | 535.11 | 0.00 | ORB-short ORB[535.50,542.50] vol=1.7x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:00:00 | 529.07 | 534.22 | 0.00 | T1 1.5R @ 529.07 |
| Target hit | 2026-03-19 15:20:00 | 522.80 | 529.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-03-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-23 09:35:00 | 535.15 | 531.56 | 0.00 | ORB-long ORB[527.05,533.95] vol=1.7x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:40:00 | 539.21 | 532.62 | 0.00 | T1 1.5R @ 539.21 |
| Stop hit — per-position SL triggered | 2026-03-23 09:45:00 | 535.15 | 532.89 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 558.95 | 561.62 | 0.00 | ORB-short ORB[559.65,564.70] vol=2.2x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-04-10 10:20:00 | 560.94 | 561.51 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:25:00 | 582.90 | 580.19 | 0.00 | ORB-long ORB[573.00,580.45] vol=2.8x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:50:00 | 585.33 | 581.10 | 0.00 | T1 1.5R @ 585.33 |
| Stop hit — per-position SL triggered | 2026-04-21 15:10:00 | 582.90 | 582.71 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 567.30 | 571.65 | 0.00 | ORB-short ORB[569.00,576.00] vol=2.9x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-04-23 11:25:00 | 568.79 | 571.54 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 576.10 | 570.27 | 0.00 | ORB-long ORB[565.50,571.60] vol=4.2x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-04-30 09:35:00 | 573.88 | 571.28 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 626.10 | 2026-02-10 09:40:00 | 628.32 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-10 09:30:00 | 626.10 | 2026-02-10 15:05:00 | 630.70 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2026-02-17 09:40:00 | 606.45 | 2026-02-17 11:30:00 | 609.53 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-17 09:40:00 | 606.45 | 2026-02-17 13:50:00 | 607.05 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-02-20 09:40:00 | 604.80 | 2026-02-20 10:15:00 | 608.56 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-20 09:40:00 | 604.80 | 2026-02-20 15:20:00 | 607.00 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-23 11:15:00 | 601.45 | 2026-02-23 12:55:00 | 602.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-26 10:40:00 | 587.40 | 2026-02-26 10:55:00 | 585.84 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-13 09:45:00 | 544.25 | 2026-03-13 10:05:00 | 541.52 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-13 09:45:00 | 544.25 | 2026-03-13 10:30:00 | 544.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 10:25:00 | 532.20 | 2026-03-19 11:00:00 | 529.07 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-19 10:25:00 | 532.20 | 2026-03-19 15:20:00 | 522.80 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2026-03-23 09:35:00 | 535.15 | 2026-03-23 09:40:00 | 539.21 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-03-23 09:35:00 | 535.15 | 2026-03-23 09:45:00 | 535.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-10 10:05:00 | 558.95 | 2026-04-10 10:20:00 | 560.94 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-21 10:25:00 | 582.90 | 2026-04-21 11:50:00 | 585.33 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-21 10:25:00 | 582.90 | 2026-04-21 15:10:00 | 582.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:15:00 | 567.30 | 2026-04-23 11:25:00 | 568.79 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-30 09:30:00 | 576.10 | 2026-04-30 09:35:00 | 573.88 | STOP_HIT | 1.00 | -0.39% |
