# Brainbees Solutions Ltd. (FIRSTCRY)

## Backtest Summary

- **Window:** 2024-08-13 09:45:00 → 2026-05-08 15:25:00 (30394 bars)
- **Last close:** 234.91
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 6
- **Avg / median % per leg:** 0.28% / 0.00%
- **Sum % (uncompounded):** 5.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.16% | 1.9% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.16% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.45% | 3.6% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.45% | 3.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.28% | 5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:30:00 | 622.65 | 630.55 | 0.00 | ORB-short ORB[631.90,640.95] vol=2.4x ATR=2.82 |
| Stop hit — per-position SL triggered | 2024-08-28 10:45:00 | 625.47 | 629.27 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:45:00 | 618.85 | 615.18 | 0.00 | ORB-long ORB[610.20,618.25] vol=2.3x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-09-05 09:50:00 | 616.58 | 615.26 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:30:00 | 621.80 | 616.98 | 0.00 | ORB-long ORB[611.00,618.00] vol=3.5x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:50:00 | 626.31 | 620.09 | 0.00 | T1 1.5R @ 626.31 |
| Stop hit — per-position SL triggered | 2024-09-06 10:10:00 | 621.80 | 621.92 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-09-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:35:00 | 655.80 | 649.83 | 0.00 | ORB-long ORB[641.50,649.90] vol=1.5x ATR=2.63 |
| Stop hit — per-position SL triggered | 2024-09-24 10:45:00 | 653.17 | 650.87 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-09-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 10:35:00 | 650.90 | 653.34 | 0.00 | ORB-short ORB[651.25,656.95] vol=2.1x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 11:10:00 | 648.91 | 653.04 | 0.00 | T1 1.5R @ 648.91 |
| Target hit | 2024-09-27 15:20:00 | 624.25 | 636.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-10-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:30:00 | 687.75 | 678.75 | 0.00 | ORB-long ORB[669.00,678.95] vol=2.7x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:35:00 | 693.84 | 681.28 | 0.00 | T1 1.5R @ 693.84 |
| Stop hit — per-position SL triggered | 2024-10-11 10:40:00 | 687.75 | 682.46 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 09:45:00 | 706.80 | 712.04 | 0.00 | ORB-short ORB[712.15,720.15] vol=2.9x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:50:00 | 701.98 | 710.38 | 0.00 | T1 1.5R @ 701.98 |
| Target hit | 2024-10-16 12:20:00 | 703.25 | 700.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2024-11-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:30:00 | 557.05 | 554.03 | 0.00 | ORB-long ORB[548.30,556.55] vol=2.1x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-11-25 09:45:00 | 553.80 | 554.13 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-12-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:20:00 | 614.95 | 613.17 | 0.00 | ORB-long ORB[605.65,614.75] vol=2.0x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-12-04 10:35:00 | 612.27 | 613.13 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-12-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:05:00 | 612.35 | 605.27 | 0.00 | ORB-long ORB[594.65,603.75] vol=2.7x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 13:10:00 | 616.66 | 611.46 | 0.00 | T1 1.5R @ 616.66 |
| Target hit | 2024-12-16 15:20:00 | 616.75 | 612.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2025-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:10:00 | 374.35 | 377.33 | 0.00 | ORB-short ORB[375.95,380.00] vol=1.7x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-03-12 11:35:00 | 376.02 | 376.91 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:05:00 | 333.30 | 335.53 | 0.00 | ORB-short ORB[333.55,338.60] vol=4.4x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-04-17 10:25:00 | 334.86 | 335.13 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:50:00 | 368.40 | 363.81 | 0.00 | ORB-long ORB[356.50,362.00] vol=2.6x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 11:20:00 | 370.89 | 365.02 | 0.00 | T1 1.5R @ 370.89 |
| Stop hit — per-position SL triggered | 2025-04-22 11:30:00 | 368.40 | 365.39 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 09:45:00 | 314.60 | 318.30 | 0.00 | ORB-short ORB[316.75,321.50] vol=2.7x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-05-07 09:55:00 | 316.42 | 317.91 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-08-28 10:30:00 | 622.65 | 2024-08-28 10:45:00 | 625.47 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-09-05 09:45:00 | 618.85 | 2024-09-05 09:50:00 | 616.58 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-06 09:30:00 | 621.80 | 2024-09-06 09:50:00 | 626.31 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-09-06 09:30:00 | 621.80 | 2024-09-06 10:10:00 | 621.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-24 10:35:00 | 655.80 | 2024-09-24 10:45:00 | 653.17 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-27 10:35:00 | 650.90 | 2024-09-27 11:10:00 | 648.91 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-09-27 10:35:00 | 650.90 | 2024-09-27 15:20:00 | 624.25 | TARGET_HIT | 0.50 | 4.09% |
| BUY | retest1 | 2024-10-11 10:30:00 | 687.75 | 2024-10-11 10:35:00 | 693.84 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2024-10-11 10:30:00 | 687.75 | 2024-10-11 10:40:00 | 687.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 09:45:00 | 706.80 | 2024-10-16 09:50:00 | 701.98 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-10-16 09:45:00 | 706.80 | 2024-10-16 12:20:00 | 703.25 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-11-25 09:30:00 | 557.05 | 2024-11-25 09:45:00 | 553.80 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-12-04 10:20:00 | 614.95 | 2024-12-04 10:35:00 | 612.27 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-12-16 10:05:00 | 612.35 | 2024-12-16 13:10:00 | 616.66 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-12-16 10:05:00 | 612.35 | 2024-12-16 15:20:00 | 616.75 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2025-03-12 11:10:00 | 374.35 | 2025-03-12 11:35:00 | 376.02 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-04-17 10:05:00 | 333.30 | 2025-04-17 10:25:00 | 334.86 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-04-22 10:50:00 | 368.40 | 2025-04-22 11:20:00 | 370.89 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-04-22 10:50:00 | 368.40 | 2025-04-22 11:30:00 | 368.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-07 09:45:00 | 314.60 | 2025-05-07 09:55:00 | 316.42 | STOP_HIT | 1.00 | -0.58% |
