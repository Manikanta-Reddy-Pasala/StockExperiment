# Clean Science and Technology Ltd. (CLEAN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 891.90
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
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 14
- **Target hits / Stop hits / Partials:** 6 / 14 / 9
- **Avg / median % per leg:** 0.38% / 0.14%
- **Sum % (uncompounded):** 10.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 9 | 52.9% | 4 | 8 | 5 | 0.32% | 5.5% |
| BUY @ 2nd Alert (retest1) | 17 | 9 | 52.9% | 4 | 8 | 5 | 0.32% | 5.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.46% | 5.5% |
| SELL @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.46% | 5.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 15 | 51.7% | 6 | 14 | 9 | 0.38% | 11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 787.50 | 793.66 | 0.00 | ORB-short ORB[793.40,804.25] vol=2.0x ATR=2.56 |
| Stop hit — per-position SL triggered | 2026-02-11 10:20:00 | 790.06 | 791.18 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 752.30 | 748.62 | 0.00 | ORB-long ORB[745.20,750.00] vol=1.7x ATR=3.01 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 749.29 | 748.90 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 751.85 | 747.24 | 0.00 | ORB-long ORB[737.55,747.20] vol=2.2x ATR=2.56 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 749.29 | 747.38 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 748.75 | 752.38 | 0.00 | ORB-short ORB[755.10,760.00] vol=2.3x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 745.62 | 750.40 | 0.00 | T1 1.5R @ 745.62 |
| Target hit | 2026-02-19 15:20:00 | 733.95 | 744.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:20:00 | 721.00 | 723.24 | 0.00 | ORB-short ORB[722.45,729.90] vol=1.6x ATR=1.78 |
| Stop hit — per-position SL triggered | 2026-02-25 10:25:00 | 722.78 | 723.32 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 735.15 | 728.47 | 0.00 | ORB-long ORB[721.50,726.40] vol=2.7x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 732.70 | 728.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 11:05:00 | 735.55 | 731.86 | 0.00 | ORB-long ORB[725.00,733.65] vol=2.7x ATR=2.48 |
| Stop hit — per-position SL triggered | 2026-03-04 11:25:00 | 733.07 | 732.07 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:50:00 | 748.65 | 739.49 | 0.00 | ORB-long ORB[734.30,740.00] vol=1.5x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:55:00 | 753.48 | 743.65 | 0.00 | T1 1.5R @ 753.48 |
| Target hit | 2026-03-05 15:20:00 | 773.00 | 763.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 736.80 | 739.13 | 0.00 | ORB-short ORB[737.50,744.00] vol=2.4x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-03-10 09:45:00 | 739.33 | 739.07 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 719.50 | 722.48 | 0.00 | ORB-short ORB[722.15,728.65] vol=2.0x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:05:00 | 715.58 | 721.37 | 0.00 | T1 1.5R @ 715.58 |
| Target hit | 2026-03-13 15:20:00 | 701.65 | 707.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-03-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:20:00 | 689.10 | 696.36 | 0.00 | ORB-short ORB[690.05,698.25] vol=1.5x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:50:00 | 684.29 | 690.59 | 0.00 | T1 1.5R @ 684.29 |
| Stop hit — per-position SL triggered | 2026-03-16 12:50:00 | 689.10 | 684.63 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:55:00 | 695.00 | 701.29 | 0.00 | ORB-short ORB[699.70,704.85] vol=2.2x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:25:00 | 691.11 | 699.47 | 0.00 | T1 1.5R @ 691.11 |
| Stop hit — per-position SL triggered | 2026-03-17 14:00:00 | 695.00 | 694.55 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:30:00 | 677.65 | 672.86 | 0.00 | ORB-long ORB[666.00,675.00] vol=1.6x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-03-30 09:35:00 | 674.34 | 673.14 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 728.80 | 724.98 | 0.00 | ORB-long ORB[718.55,726.70] vol=3.4x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:15:00 | 733.04 | 728.14 | 0.00 | T1 1.5R @ 733.04 |
| Stop hit — per-position SL triggered | 2026-04-10 10:25:00 | 728.80 | 728.30 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 742.00 | 736.46 | 0.00 | ORB-long ORB[730.00,739.50] vol=2.0x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:55:00 | 746.98 | 739.72 | 0.00 | T1 1.5R @ 746.98 |
| Target hit | 2026-04-15 15:20:00 | 752.00 | 747.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 759.10 | 757.09 | 0.00 | ORB-long ORB[752.75,758.80] vol=1.6x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:10:00 | 762.98 | 759.63 | 0.00 | T1 1.5R @ 762.98 |
| Target hit | 2026-04-17 10:30:00 | 760.20 | 760.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 749.60 | 746.83 | 0.00 | ORB-long ORB[741.70,749.15] vol=3.3x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:10:00 | 753.35 | 748.95 | 0.00 | T1 1.5R @ 753.35 |
| Target hit | 2026-04-21 11:35:00 | 752.70 | 753.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 813.95 | 804.86 | 0.00 | ORB-long ORB[798.00,810.00] vol=5.1x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-04-29 13:45:00 | 810.60 | 811.42 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 815.00 | 818.10 | 0.00 | ORB-short ORB[817.20,825.65] vol=2.7x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-05-05 11:35:00 | 817.24 | 818.02 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:35:00 | 878.15 | 869.85 | 0.00 | ORB-long ORB[861.40,874.30] vol=1.6x ATR=3.83 |
| Stop hit — per-position SL triggered | 2026-05-07 15:20:00 | 875.60 | 872.56 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:35:00 | 787.50 | 2026-02-11 10:20:00 | 790.06 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-16 09:30:00 | 752.30 | 2026-02-16 09:40:00 | 749.29 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-17 09:45:00 | 751.85 | 2026-02-17 09:50:00 | 749.29 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-19 10:35:00 | 748.75 | 2026-02-19 11:15:00 | 745.62 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-19 10:35:00 | 748.75 | 2026-02-19 15:20:00 | 733.95 | TARGET_HIT | 0.50 | 1.98% |
| SELL | retest1 | 2026-02-25 10:20:00 | 721.00 | 2026-02-25 10:25:00 | 722.78 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-26 09:50:00 | 735.15 | 2026-02-26 09:55:00 | 732.70 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-04 11:05:00 | 735.55 | 2026-03-04 11:25:00 | 733.07 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-05 09:50:00 | 748.65 | 2026-03-05 09:55:00 | 753.48 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-03-05 09:50:00 | 748.65 | 2026-03-05 15:20:00 | 773.00 | TARGET_HIT | 0.50 | 3.25% |
| SELL | retest1 | 2026-03-10 09:30:00 | 736.80 | 2026-03-10 09:45:00 | 739.33 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-13 09:45:00 | 719.50 | 2026-03-13 10:05:00 | 715.58 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-13 09:45:00 | 719.50 | 2026-03-13 15:20:00 | 701.65 | TARGET_HIT | 0.50 | 2.48% |
| SELL | retest1 | 2026-03-16 10:20:00 | 689.10 | 2026-03-16 10:50:00 | 684.29 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-03-16 10:20:00 | 689.10 | 2026-03-16 12:50:00 | 689.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 10:55:00 | 695.00 | 2026-03-17 11:25:00 | 691.11 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-17 10:55:00 | 695.00 | 2026-03-17 14:00:00 | 695.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-30 09:30:00 | 677.65 | 2026-03-30 09:35:00 | 674.34 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-10 09:30:00 | 728.80 | 2026-04-10 10:15:00 | 733.04 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-10 09:30:00 | 728.80 | 2026-04-10 10:25:00 | 728.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:35:00 | 742.00 | 2026-04-15 09:55:00 | 746.98 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-04-15 09:35:00 | 742.00 | 2026-04-15 15:20:00 | 752.00 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2026-04-17 09:50:00 | 759.10 | 2026-04-17 10:10:00 | 762.98 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-17 09:50:00 | 759.10 | 2026-04-17 10:30:00 | 760.20 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2026-04-21 09:35:00 | 749.60 | 2026-04-21 10:10:00 | 753.35 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-21 09:35:00 | 749.60 | 2026-04-21 11:35:00 | 752.70 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-29 10:00:00 | 813.95 | 2026-04-29 13:45:00 | 810.60 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-05-05 11:15:00 | 815.00 | 2026-05-05 11:35:00 | 817.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-07 10:35:00 | 878.15 | 2026-05-07 15:20:00 | 875.60 | STOP_HIT | 1.00 | -0.29% |
