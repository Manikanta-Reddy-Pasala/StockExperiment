# JBM Auto Ltd. (JBMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 649.00
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 9
- **Target hits / Stop hits / Partials:** 4 / 9 / 5
- **Avg / median % per leg:** 0.14% / 0.09%
- **Sum % (uncompounded):** 2.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.04% | -0.4% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.04% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.37% | 3.0% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.37% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 9 | 50.0% | 4 | 9 | 5 | 0.14% | 2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 593.00 | 589.50 | 0.00 | ORB-long ORB[583.55,592.00] vol=1.7x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 15:00:00 | 599.09 | 592.05 | 0.00 | T1 1.5R @ 599.09 |
| Target hit | 2026-02-09 15:20:00 | 595.50 | 592.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 581.40 | 577.17 | 0.00 | ORB-long ORB[572.90,578.95] vol=2.3x ATR=2.16 |
| Stop hit — per-position SL triggered | 2026-02-16 09:35:00 | 579.24 | 577.60 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:15:00 | 570.00 | 574.20 | 0.00 | ORB-short ORB[570.95,577.70] vol=1.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:25:00 | 567.41 | 573.38 | 0.00 | T1 1.5R @ 567.41 |
| Target hit | 2026-02-23 13:40:00 | 569.50 | 568.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 555.60 | 557.68 | 0.00 | ORB-short ORB[557.65,562.35] vol=2.7x ATR=2.32 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 557.92 | 557.65 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 553.65 | 555.58 | 0.00 | ORB-short ORB[555.90,560.65] vol=3.3x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-02-27 11:10:00 | 555.12 | 555.50 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 500.95 | 504.69 | 0.00 | ORB-short ORB[505.70,511.90] vol=5.3x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 496.64 | 502.46 | 0.00 | T1 1.5R @ 496.64 |
| Target hit | 2026-03-13 15:20:00 | 495.00 | 498.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:55:00 | 623.95 | 619.33 | 0.00 | ORB-long ORB[613.75,621.10] vol=4.8x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:00:00 | 627.37 | 622.61 | 0.00 | T1 1.5R @ 627.37 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 623.95 | 623.16 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 635.90 | 629.57 | 0.00 | ORB-long ORB[622.15,629.00] vol=6.1x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 633.43 | 631.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 636.75 | 631.34 | 0.00 | ORB-long ORB[622.20,630.40] vol=6.4x ATR=2.71 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 634.04 | 631.97 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 626.50 | 621.43 | 0.00 | ORB-long ORB[615.25,622.65] vol=3.4x ATR=1.95 |
| Stop hit — per-position SL triggered | 2026-04-27 11:20:00 | 624.55 | 622.23 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 632.50 | 627.33 | 0.00 | ORB-long ORB[621.35,628.90] vol=2.9x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 629.92 | 628.04 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 621.75 | 625.10 | 0.00 | ORB-short ORB[622.10,628.00] vol=2.2x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:00:00 | 618.43 | 623.05 | 0.00 | T1 1.5R @ 618.43 |
| Target hit | 2026-05-06 14:50:00 | 618.40 | 617.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 632.70 | 624.23 | 0.00 | ORB-long ORB[616.00,623.75] vol=5.9x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 629.60 | 625.11 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 593.00 | 2026-02-09 15:00:00 | 599.09 | PARTIAL | 0.50 | 1.03% |
| BUY | retest1 | 2026-02-09 10:35:00 | 593.00 | 2026-02-09 15:20:00 | 595.50 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-16 09:30:00 | 581.40 | 2026-02-16 09:35:00 | 579.24 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-23 10:15:00 | 570.00 | 2026-02-23 10:25:00 | 567.41 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-23 10:15:00 | 570.00 | 2026-02-23 13:40:00 | 569.50 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2026-02-25 09:40:00 | 555.60 | 2026-02-25 09:45:00 | 557.92 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-27 11:05:00 | 553.65 | 2026-02-27 11:10:00 | 555.12 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-13 09:40:00 | 500.95 | 2026-03-13 10:10:00 | 496.64 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2026-03-13 09:40:00 | 500.95 | 2026-03-13 15:20:00 | 495.00 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2026-04-21 09:55:00 | 623.95 | 2026-04-21 10:00:00 | 627.37 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-21 09:55:00 | 623.95 | 2026-04-21 10:15:00 | 623.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:35:00 | 635.90 | 2026-04-22 09:40:00 | 633.43 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-23 09:35:00 | 636.75 | 2026-04-23 09:40:00 | 634.04 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-27 10:55:00 | 626.50 | 2026-04-27 11:20:00 | 624.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-05 09:35:00 | 632.50 | 2026-05-05 09:40:00 | 629.92 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-05-06 09:35:00 | 621.75 | 2026-05-06 10:00:00 | 618.43 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-06 09:35:00 | 621.75 | 2026-05-06 14:50:00 | 618.40 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-08 09:50:00 | 632.70 | 2026-05-08 09:55:00 | 629.60 | STOP_HIT | 1.00 | -0.49% |
