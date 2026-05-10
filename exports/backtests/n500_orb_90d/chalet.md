# Chalet Hotels Ltd. (CHALET)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 787.00
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
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 7
- **Target hits / Stop hits / Partials:** 6 / 7 / 7
- **Avg / median % per leg:** 0.41% / 0.47%
- **Sum % (uncompounded):** 8.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 11 | 64.7% | 5 | 6 | 6 | 0.35% | 5.9% |
| BUY @ 2nd Alert (retest1) | 17 | 11 | 64.7% | 5 | 6 | 6 | 0.35% | 5.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.78% | 2.3% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.78% | 2.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 13 | 65.0% | 6 | 7 | 7 | 0.41% | 8.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:30:00 | 874.90 | 870.81 | 0.00 | ORB-long ORB[864.55,874.00] vol=1.9x ATR=2.54 |
| Stop hit — per-position SL triggered | 2026-02-10 10:35:00 | 872.36 | 871.05 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:10:00 | 880.80 | 873.52 | 0.00 | ORB-long ORB[865.55,878.70] vol=2.1x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-02-12 11:30:00 | 877.35 | 877.87 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 862.65 | 856.37 | 0.00 | ORB-long ORB[851.80,862.20] vol=1.7x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 866.96 | 857.58 | 0.00 | T1 1.5R @ 866.96 |
| Target hit | 2026-02-17 13:05:00 | 871.70 | 871.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 867.70 | 870.51 | 0.00 | ORB-short ORB[869.00,879.80] vol=1.9x ATR=2.64 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 870.34 | 870.02 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 839.60 | 842.08 | 0.00 | ORB-short ORB[840.80,846.40] vol=4.7x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:00:00 | 835.57 | 839.77 | 0.00 | T1 1.5R @ 835.57 |
| Target hit | 2026-02-25 11:50:00 | 821.40 | 820.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-03-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:20:00 | 770.80 | 769.23 | 0.00 | ORB-long ORB[763.60,770.50] vol=14.7x ATR=2.99 |
| Stop hit — per-position SL triggered | 2026-03-06 10:35:00 | 767.81 | 769.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 09:45:00 | 726.00 | 721.36 | 0.00 | ORB-long ORB[715.00,724.40] vol=4.8x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 10:00:00 | 732.22 | 722.13 | 0.00 | T1 1.5R @ 732.22 |
| Target hit | 2026-04-07 15:20:00 | 740.00 | 733.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 768.85 | 762.82 | 0.00 | ORB-long ORB[756.00,761.15] vol=3.2x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:55:00 | 772.43 | 766.36 | 0.00 | T1 1.5R @ 772.43 |
| Stop hit — per-position SL triggered | 2026-04-10 11:35:00 | 768.85 | 771.78 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 774.25 | 771.87 | 0.00 | ORB-long ORB[765.55,773.95] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 772.46 | 771.90 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 803.60 | 797.11 | 0.00 | ORB-long ORB[784.35,793.75] vol=1.6x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 799.82 | 797.27 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 809.70 | 805.62 | 0.00 | ORB-long ORB[798.80,805.80] vol=2.0x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:05:00 | 814.14 | 811.42 | 0.00 | T1 1.5R @ 814.14 |
| Target hit | 2026-04-22 14:15:00 | 813.30 | 813.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 803.35 | 798.22 | 0.00 | ORB-long ORB[792.00,802.60] vol=2.5x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:55:00 | 809.45 | 800.76 | 0.00 | T1 1.5R @ 809.45 |
| Target hit | 2026-04-27 12:05:00 | 805.00 | 808.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2026-05-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:10:00 | 764.10 | 758.79 | 0.00 | ORB-long ORB[754.35,760.85] vol=2.0x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:10:00 | 767.96 | 769.01 | 0.00 | T1 1.5R @ 767.96 |
| Target hit | 2026-05-06 11:15:00 | 767.15 | 769.02 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:30:00 | 874.90 | 2026-02-10 10:35:00 | 872.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-12 10:10:00 | 880.80 | 2026-02-12 11:30:00 | 877.35 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-17 10:20:00 | 862.65 | 2026-02-17 10:30:00 | 866.96 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-17 10:20:00 | 862.65 | 2026-02-17 13:05:00 | 871.70 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2026-02-18 10:00:00 | 867.70 | 2026-02-18 10:55:00 | 870.34 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-25 09:40:00 | 839.60 | 2026-02-25 10:00:00 | 835.57 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-25 09:40:00 | 839.60 | 2026-02-25 11:50:00 | 821.40 | TARGET_HIT | 0.50 | 2.17% |
| BUY | retest1 | 2026-03-06 10:20:00 | 770.80 | 2026-03-06 10:35:00 | 767.81 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-07 09:45:00 | 726.00 | 2026-04-07 10:00:00 | 732.22 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2026-04-07 09:45:00 | 726.00 | 2026-04-07 15:20:00 | 740.00 | TARGET_HIT | 0.50 | 1.93% |
| BUY | retest1 | 2026-04-10 09:35:00 | 768.85 | 2026-04-10 09:55:00 | 772.43 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-10 09:35:00 | 768.85 | 2026-04-10 11:35:00 | 768.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:55:00 | 774.25 | 2026-04-17 11:00:00 | 772.46 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-21 09:35:00 | 803.60 | 2026-04-21 09:40:00 | 799.82 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-22 09:35:00 | 809.70 | 2026-04-22 11:05:00 | 814.14 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-22 09:35:00 | 809.70 | 2026-04-22 14:15:00 | 813.30 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-27 09:40:00 | 803.35 | 2026-04-27 09:55:00 | 809.45 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-04-27 09:40:00 | 803.35 | 2026-04-27 12:05:00 | 805.00 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2026-05-06 10:10:00 | 764.10 | 2026-05-06 11:10:00 | 767.96 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-06 10:10:00 | 764.10 | 2026-05-06 11:15:00 | 767.15 | TARGET_HIT | 0.50 | 0.40% |
