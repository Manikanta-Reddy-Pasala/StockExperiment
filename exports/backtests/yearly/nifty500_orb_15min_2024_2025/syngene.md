# Syngene International Ltd. (SYNGENE)

## Backtest Summary

- **Window:** 2025-02-05 09:15:00 → 2026-05-08 15:25:00 (23038 bars)
- **Last close:** 459.50
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 4
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 0.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 1 | 8 | 2 | 0.02% | 0.2% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 1 | 8 | 2 | 0.02% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.08% | 0.4% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 0.08% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 5 | 31.2% | 1 | 11 | 4 | 0.04% | 0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 11:00:00 | 729.65 | 735.80 | 0.00 | ORB-short ORB[736.10,745.60] vol=2.3x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:40:00 | 726.65 | 733.05 | 0.00 | T1 1.5R @ 726.65 |
| Stop hit — per-position SL triggered | 2025-02-10 13:25:00 | 729.65 | 728.85 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:10:00 | 694.50 | 699.38 | 0.00 | ORB-short ORB[695.00,702.90] vol=2.4x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-02-18 10:15:00 | 697.32 | 698.98 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:50:00 | 708.55 | 700.75 | 0.00 | ORB-long ORB[693.55,703.40] vol=1.8x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-02-20 11:10:00 | 706.48 | 702.80 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:45:00 | 674.45 | 671.64 | 0.00 | ORB-long ORB[663.90,671.70] vol=2.6x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 09:55:00 | 678.40 | 672.17 | 0.00 | T1 1.5R @ 678.40 |
| Stop hit — per-position SL triggered | 2025-03-17 10:05:00 | 674.45 | 672.19 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:30:00 | 694.75 | 690.52 | 0.00 | ORB-long ORB[684.95,691.55] vol=3.1x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 09:45:00 | 697.79 | 692.92 | 0.00 | T1 1.5R @ 697.79 |
| Target hit | 2025-03-19 15:20:00 | 705.75 | 699.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 713.65 | 709.54 | 0.00 | ORB-long ORB[700.55,708.70] vol=1.7x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 711.38 | 710.87 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-03-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 10:45:00 | 726.70 | 722.07 | 0.00 | ORB-long ORB[720.40,724.70] vol=4.6x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-03-28 11:10:00 | 724.35 | 722.46 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-04-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:45:00 | 729.05 | 728.79 | 0.00 | ORB-long ORB[721.10,727.45] vol=2.5x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-04-16 11:05:00 | 726.48 | 728.74 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-04-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:45:00 | 728.00 | 730.52 | 0.00 | ORB-short ORB[730.00,738.95] vol=4.3x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 12:05:00 | 725.11 | 729.43 | 0.00 | T1 1.5R @ 725.11 |
| Stop hit — per-position SL triggered | 2025-04-17 14:10:00 | 728.00 | 728.21 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:40:00 | 745.90 | 742.65 | 0.00 | ORB-long ORB[740.00,745.00] vol=2.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 743.17 | 742.29 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:45:00 | 639.50 | 633.88 | 0.00 | ORB-long ORB[626.00,633.50] vol=3.0x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-04-30 11:25:00 | 636.95 | 634.96 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:45:00 | 634.50 | 630.07 | 0.00 | ORB-long ORB[625.90,632.20] vol=1.7x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-05-06 09:50:00 | 632.52 | 630.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-02-10 11:00:00 | 729.65 | 2025-02-10 11:40:00 | 726.65 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-02-10 11:00:00 | 729.65 | 2025-02-10 13:25:00 | 729.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 10:10:00 | 694.50 | 2025-02-18 10:15:00 | 697.32 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-02-20 10:50:00 | 708.55 | 2025-02-20 11:10:00 | 706.48 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-17 09:45:00 | 674.45 | 2025-03-17 09:55:00 | 678.40 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-17 09:45:00 | 674.45 | 2025-03-17 10:05:00 | 674.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 09:30:00 | 694.75 | 2025-03-19 09:45:00 | 697.79 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-03-19 09:30:00 | 694.75 | 2025-03-19 15:20:00 | 705.75 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2025-03-21 09:35:00 | 713.65 | 2025-03-21 09:50:00 | 711.38 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-28 10:45:00 | 726.70 | 2025-03-28 11:10:00 | 724.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-16 10:45:00 | 729.05 | 2025-04-16 11:05:00 | 726.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-17 10:45:00 | 728.00 | 2025-04-17 12:05:00 | 725.11 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-04-17 10:45:00 | 728.00 | 2025-04-17 14:10:00 | 728.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-23 09:40:00 | 745.90 | 2025-04-23 09:45:00 | 743.17 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-30 10:45:00 | 639.50 | 2025-04-30 11:25:00 | 636.95 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-05-06 09:45:00 | 634.50 | 2025-05-06 09:50:00 | 632.52 | STOP_HIT | 1.00 | -0.31% |
