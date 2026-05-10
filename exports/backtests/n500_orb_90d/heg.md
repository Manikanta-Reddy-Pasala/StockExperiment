# H.E.G. Ltd. (HEG)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 596.00
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 15
- **Target hits / Stop hits / Partials:** 2 / 15 / 6
- **Avg / median % per leg:** -0.01% / 0.00%
- **Sum % (uncompounded):** -0.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 8 | 38.1% | 2 | 13 | 6 | 0.04% | 0.8% |
| BUY @ 2nd Alert (retest1) | 21 | 8 | 38.1% | 2 | 13 | 6 | 0.04% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.52% | -1.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.52% | -1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 8 | 34.8% | 2 | 15 | 6 | -0.01% | -0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 559.90 | 552.87 | 0.00 | ORB-long ORB[539.80,545.35] vol=2.9x ATR=4.09 |
| Stop hit — per-position SL triggered | 2026-02-09 10:45:00 | 555.81 | 553.03 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 554.25 | 551.27 | 0.00 | ORB-long ORB[545.25,552.35] vol=4.0x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 558.33 | 552.77 | 0.00 | T1 1.5R @ 558.33 |
| Stop hit — per-position SL triggered | 2026-02-10 10:25:00 | 554.25 | 553.02 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:50:00 | 528.75 | 523.03 | 0.00 | ORB-long ORB[516.00,523.60] vol=3.6x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:55:00 | 531.91 | 524.76 | 0.00 | T1 1.5R @ 531.91 |
| Stop hit — per-position SL triggered | 2026-02-16 11:10:00 | 528.75 | 525.62 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 534.20 | 532.07 | 0.00 | ORB-long ORB[527.50,533.60] vol=4.5x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 537.51 | 533.98 | 0.00 | T1 1.5R @ 537.51 |
| Stop hit — per-position SL triggered | 2026-02-17 10:35:00 | 534.20 | 534.00 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 552.40 | 546.18 | 0.00 | ORB-long ORB[541.00,546.95] vol=2.4x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:35:00 | 556.09 | 549.03 | 0.00 | T1 1.5R @ 556.09 |
| Target hit | 2026-02-20 11:00:00 | 553.90 | 554.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 576.95 | 570.58 | 0.00 | ORB-long ORB[564.75,571.40] vol=2.3x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:10:00 | 581.28 | 574.42 | 0.00 | T1 1.5R @ 581.28 |
| Stop hit — per-position SL triggered | 2026-02-25 12:35:00 | 576.95 | 575.02 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 586.40 | 582.24 | 0.00 | ORB-long ORB[576.55,584.20] vol=3.8x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-02-26 12:20:00 | 583.32 | 584.73 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 543.75 | 548.05 | 0.00 | ORB-short ORB[546.50,554.70] vol=2.8x ATR=3.07 |
| Stop hit — per-position SL triggered | 2026-03-05 10:15:00 | 546.82 | 546.45 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:20:00 | 548.35 | 544.49 | 0.00 | ORB-long ORB[539.20,546.80] vol=2.1x ATR=2.16 |
| Stop hit — per-position SL triggered | 2026-03-06 10:30:00 | 546.19 | 544.65 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 513.15 | 508.35 | 0.00 | ORB-long ORB[503.20,510.00] vol=3.1x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 510.65 | 509.18 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:55:00 | 483.30 | 488.18 | 0.00 | ORB-short ORB[484.30,490.95] vol=1.7x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-03-24 11:05:00 | 485.64 | 488.13 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 503.35 | 498.60 | 0.00 | ORB-long ORB[492.65,499.60] vol=2.7x ATR=2.49 |
| Stop hit — per-position SL triggered | 2026-03-25 09:35:00 | 500.86 | 498.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:40:00 | 569.30 | 562.47 | 0.00 | ORB-long ORB[555.50,563.00] vol=5.8x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-04-08 10:50:00 | 566.55 | 563.67 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:40:00 | 559.30 | 552.07 | 0.00 | ORB-long ORB[545.00,552.60] vol=1.8x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:55:00 | 562.75 | 553.76 | 0.00 | T1 1.5R @ 562.75 |
| Target hit | 2026-04-13 15:20:00 | 566.40 | 562.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 587.90 | 581.86 | 0.00 | ORB-long ORB[572.00,579.05] vol=8.5x ATR=3.20 |
| Stop hit — per-position SL triggered | 2026-04-15 09:35:00 | 584.70 | 582.94 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 602.00 | 595.52 | 0.00 | ORB-long ORB[594.00,601.60] vol=2.2x ATR=2.39 |
| Stop hit — per-position SL triggered | 2026-05-07 11:05:00 | 599.61 | 596.09 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:25:00 | 603.40 | 600.26 | 0.00 | ORB-long ORB[593.70,602.65] vol=2.2x ATR=3.95 |
| Stop hit — per-position SL triggered | 2026-05-08 10:55:00 | 599.45 | 600.36 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 559.90 | 2026-02-09 10:45:00 | 555.81 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2026-02-10 09:45:00 | 554.25 | 2026-02-10 10:15:00 | 558.33 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-02-10 09:45:00 | 554.25 | 2026-02-10 10:25:00 | 554.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:50:00 | 528.75 | 2026-02-16 10:55:00 | 531.91 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-16 10:50:00 | 528.75 | 2026-02-16 11:10:00 | 528.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:30:00 | 534.20 | 2026-02-17 10:30:00 | 537.51 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-17 09:30:00 | 534.20 | 2026-02-17 10:35:00 | 534.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 09:30:00 | 552.40 | 2026-02-20 09:35:00 | 556.09 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-02-20 09:30:00 | 552.40 | 2026-02-20 11:00:00 | 553.90 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-25 10:50:00 | 576.95 | 2026-02-25 12:10:00 | 581.28 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-02-25 10:50:00 | 576.95 | 2026-02-25 12:35:00 | 576.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:30:00 | 586.40 | 2026-02-26 12:20:00 | 583.32 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-03-05 09:30:00 | 543.75 | 2026-03-05 10:15:00 | 546.82 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-03-06 10:20:00 | 548.35 | 2026-03-06 10:30:00 | 546.19 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-18 09:30:00 | 513.15 | 2026-03-18 09:55:00 | 510.65 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-24 10:55:00 | 483.30 | 2026-03-24 11:05:00 | 485.64 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-25 09:30:00 | 503.35 | 2026-03-25 09:35:00 | 500.86 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-08 10:40:00 | 569.30 | 2026-04-08 10:50:00 | 566.55 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-13 10:40:00 | 559.30 | 2026-04-13 10:55:00 | 562.75 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-13 10:40:00 | 559.30 | 2026-04-13 15:20:00 | 566.40 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2026-04-15 09:30:00 | 587.90 | 2026-04-15 09:35:00 | 584.70 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-05-07 11:00:00 | 602.00 | 2026-05-07 11:05:00 | 599.61 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-08 10:25:00 | 603.40 | 2026-05-08 10:55:00 | 599.45 | STOP_HIT | 1.00 | -0.65% |
