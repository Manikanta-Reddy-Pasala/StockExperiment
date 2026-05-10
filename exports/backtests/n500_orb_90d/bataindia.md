# Bata India Ltd. (BATAINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 722.80
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 7
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 1.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.14% | -1.2% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | -0.14% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.15% | 2.8% |
| SELL @ 2nd Alert (retest1) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.15% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 10 | 38.5% | 3 | 16 | 7 | 0.06% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 866.05 | 870.39 | 0.00 | ORB-short ORB[867.90,880.70] vol=7.3x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:20:00 | 862.00 | 869.78 | 0.00 | T1 1.5R @ 862.00 |
| Stop hit — per-position SL triggered | 2026-02-13 12:05:00 | 866.05 | 868.89 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 841.00 | 846.40 | 0.00 | ORB-short ORB[842.45,851.90] vol=8.2x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:15:00 | 836.84 | 845.38 | 0.00 | T1 1.5R @ 836.84 |
| Target hit | 2026-02-16 15:20:00 | 828.95 | 834.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 837.25 | 828.47 | 0.00 | ORB-long ORB[823.75,829.00] vol=1.5x ATR=2.81 |
| Stop hit — per-position SL triggered | 2026-02-17 09:40:00 | 834.44 | 829.79 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 832.60 | 834.67 | 0.00 | ORB-short ORB[833.90,839.90] vol=4.8x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 834.32 | 834.62 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 801.30 | 805.84 | 0.00 | ORB-short ORB[803.80,815.00] vol=2.3x ATR=2.94 |
| Stop hit — per-position SL triggered | 2026-02-24 10:05:00 | 804.24 | 805.12 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 793.25 | 796.35 | 0.00 | ORB-short ORB[794.15,800.45] vol=2.5x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:20:00 | 790.37 | 795.12 | 0.00 | T1 1.5R @ 790.37 |
| Stop hit — per-position SL triggered | 2026-02-27 12:25:00 | 793.25 | 792.44 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 753.85 | 757.24 | 0.00 | ORB-short ORB[755.00,762.55] vol=4.2x ATR=3.34 |
| Stop hit — per-position SL triggered | 2026-03-04 10:00:00 | 757.19 | 756.37 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 748.50 | 745.10 | 0.00 | ORB-long ORB[738.80,746.80] vol=2.6x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-03-06 10:00:00 | 745.32 | 745.98 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 705.85 | 708.05 | 0.00 | ORB-short ORB[707.60,713.60] vol=2.5x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 702.50 | 706.93 | 0.00 | T1 1.5R @ 702.50 |
| Target hit | 2026-03-13 11:50:00 | 703.30 | 702.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 683.10 | 678.34 | 0.00 | ORB-long ORB[672.90,683.00] vol=3.2x ATR=2.92 |
| Stop hit — per-position SL triggered | 2026-03-18 11:05:00 | 680.18 | 678.58 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 697.50 | 703.15 | 0.00 | ORB-short ORB[701.55,709.00] vol=2.3x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:40:00 | 693.54 | 701.81 | 0.00 | T1 1.5R @ 693.54 |
| Stop hit — per-position SL triggered | 2026-04-09 10:40:00 | 697.50 | 698.01 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 713.45 | 711.22 | 0.00 | ORB-long ORB[702.00,709.85] vol=1.7x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-04-10 11:40:00 | 711.34 | 711.68 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:05:00 | 741.90 | 746.76 | 0.00 | ORB-short ORB[748.15,753.85] vol=1.7x ATR=2.37 |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 744.27 | 746.67 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 762.15 | 759.47 | 0.00 | ORB-long ORB[753.95,761.40] vol=2.6x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:35:00 | 765.62 | 761.16 | 0.00 | T1 1.5R @ 765.62 |
| Target hit | 2026-04-21 10:55:00 | 766.55 | 766.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2026-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:40:00 | 762.40 | 768.08 | 0.00 | ORB-short ORB[766.75,774.30] vol=1.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-04-23 11:05:00 | 764.75 | 767.72 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 727.30 | 724.21 | 0.00 | ORB-long ORB[721.35,726.85] vol=2.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2026-05-04 11:30:00 | 725.29 | 724.54 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 726.85 | 724.06 | 0.00 | ORB-long ORB[717.00,725.80] vol=5.3x ATR=3.06 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 723.79 | 725.90 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 716.80 | 723.86 | 0.00 | ORB-short ORB[721.05,728.40] vol=3.0x ATR=2.02 |
| Stop hit — per-position SL triggered | 2026-05-06 11:35:00 | 718.82 | 722.57 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 720.85 | 724.82 | 0.00 | ORB-short ORB[724.55,730.20] vol=2.3x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:45:00 | 716.96 | 722.58 | 0.00 | T1 1.5R @ 716.96 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 720.85 | 722.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 11:15:00 | 866.05 | 2026-02-13 11:20:00 | 862.00 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-13 11:15:00 | 866.05 | 2026-02-13 12:05:00 | 866.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-16 11:05:00 | 841.00 | 2026-02-16 11:15:00 | 836.84 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-16 11:05:00 | 841.00 | 2026-02-16 15:20:00 | 828.95 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2026-02-17 09:35:00 | 837.25 | 2026-02-17 09:40:00 | 834.44 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-19 11:15:00 | 832.60 | 2026-02-19 11:25:00 | 834.32 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-24 09:30:00 | 801.30 | 2026-02-24 10:05:00 | 804.24 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-27 10:15:00 | 793.25 | 2026-02-27 10:20:00 | 790.37 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-27 10:15:00 | 793.25 | 2026-02-27 12:25:00 | 793.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:30:00 | 753.85 | 2026-03-04 10:00:00 | 757.19 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-06 09:30:00 | 748.50 | 2026-03-06 10:00:00 | 745.32 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-13 09:50:00 | 705.85 | 2026-03-13 10:10:00 | 702.50 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-13 09:50:00 | 705.85 | 2026-03-13 11:50:00 | 703.30 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-03-18 11:00:00 | 683.10 | 2026-03-18 11:05:00 | 680.18 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-09 09:30:00 | 697.50 | 2026-04-09 09:40:00 | 693.54 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-09 09:30:00 | 697.50 | 2026-04-09 10:40:00 | 697.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 10:50:00 | 713.45 | 2026-04-10 11:40:00 | 711.34 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-16 11:05:00 | 741.90 | 2026-04-16 11:15:00 | 744.27 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-21 09:30:00 | 762.15 | 2026-04-21 09:35:00 | 765.62 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-21 09:30:00 | 762.15 | 2026-04-21 10:55:00 | 766.55 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-23 10:40:00 | 762.40 | 2026-04-23 11:05:00 | 764.75 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-04 11:15:00 | 727.30 | 2026-05-04 11:30:00 | 725.29 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-05-05 09:30:00 | 726.85 | 2026-05-05 10:10:00 | 723.79 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-05-06 10:55:00 | 716.80 | 2026-05-06 11:35:00 | 718.82 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-08 09:35:00 | 720.85 | 2026-05-08 09:45:00 | 716.96 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-05-08 09:35:00 | 720.85 | 2026-05-08 09:50:00 | 720.85 | STOP_HIT | 0.50 | 0.00% |
