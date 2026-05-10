# Life Insurance Corporation of India (LICI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 802.45
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 18
- **Target hits / Stop hits / Partials:** 1 / 18 / 4
- **Avg / median % per leg:** -0.08% / -0.19%
- **Sum % (uncompounded):** -1.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.23% | -1.4% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.23% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 5 | 29.4% | 1 | 12 | 4 | -0.02% | -0.4% |
| SELL @ 2nd Alert (retest1) | 17 | 5 | 29.4% | 1 | 12 | 4 | -0.02% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 5 | 21.7% | 1 | 18 | 4 | -0.08% | -1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 890.60 | 894.72 | 0.00 | ORB-short ORB[895.45,902.80] vol=2.7x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-02-10 11:00:00 | 892.32 | 894.64 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:45:00 | 872.45 | 877.49 | 0.00 | ORB-short ORB[874.55,886.95] vol=1.9x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:00:00 | 868.61 | 875.95 | 0.00 | T1 1.5R @ 868.61 |
| Stop hit — per-position SL triggered | 2026-02-11 10:35:00 | 872.45 | 873.08 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:10:00 | 866.00 | 871.29 | 0.00 | ORB-short ORB[870.40,875.70] vol=1.5x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 867.86 | 870.57 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:40:00 | 857.45 | 859.87 | 0.00 | ORB-short ORB[860.10,863.50] vol=1.5x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:40:00 | 853.94 | 858.42 | 0.00 | T1 1.5R @ 853.94 |
| Stop hit — per-position SL triggered | 2026-02-16 11:30:00 | 857.45 | 857.65 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 873.00 | 869.86 | 0.00 | ORB-long ORB[864.05,870.80] vol=2.5x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 870.55 | 871.21 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:30:00 | 871.90 | 873.75 | 0.00 | ORB-short ORB[872.40,876.90] vol=3.0x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-02-18 10:40:00 | 873.57 | 873.69 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:30:00 | 874.20 | 877.42 | 0.00 | ORB-short ORB[877.50,882.50] vol=1.9x ATR=1.55 |
| Stop hit — per-position SL triggered | 2026-02-19 10:45:00 | 875.75 | 877.14 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:10:00 | 884.70 | 880.85 | 0.00 | ORB-long ORB[873.65,882.90] vol=1.9x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-02-23 11:30:00 | 882.49 | 881.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 872.20 | 875.19 | 0.00 | ORB-short ORB[875.10,879.00] vol=1.6x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:00:00 | 869.41 | 874.69 | 0.00 | T1 1.5R @ 869.41 |
| Target hit | 2026-02-26 15:20:00 | 871.50 | 871.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 856.50 | 860.89 | 0.00 | ORB-short ORB[858.80,870.75] vol=1.6x ATR=2.10 |
| Stop hit — per-position SL triggered | 2026-02-27 09:55:00 | 858.60 | 860.41 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:55:00 | 825.10 | 829.71 | 0.00 | ORB-short ORB[831.00,839.95] vol=2.2x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-03-05 10:10:00 | 827.08 | 828.83 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:00:00 | 801.65 | 795.14 | 0.00 | ORB-long ORB[791.55,799.90] vol=1.8x ATR=1.90 |
| Stop hit — per-position SL triggered | 2026-03-12 11:05:00 | 799.75 | 795.30 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 788.45 | 792.47 | 0.00 | ORB-short ORB[791.60,798.00] vol=2.1x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:05:00 | 785.59 | 790.35 | 0.00 | T1 1.5R @ 785.59 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 788.45 | 789.75 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-03-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:05:00 | 765.05 | 767.33 | 0.00 | ORB-short ORB[765.25,774.35] vol=1.6x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 767.48 | 767.23 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 778.30 | 776.00 | 0.00 | ORB-long ORB[769.85,778.05] vol=1.9x ATR=2.52 |
| Stop hit — per-position SL triggered | 2026-03-20 09:40:00 | 775.78 | 776.20 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:10:00 | 820.00 | 818.30 | 0.00 | ORB-long ORB[812.40,818.80] vol=1.5x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 818.64 | 818.59 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:45:00 | 821.60 | 819.40 | 0.00 | ORB-long ORB[816.10,820.40] vol=2.7x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 820.35 | 819.48 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:30:00 | 799.05 | 804.16 | 0.00 | ORB-short ORB[805.25,812.05] vol=1.9x ATR=1.81 |
| Stop hit — per-position SL triggered | 2026-04-30 12:15:00 | 800.86 | 802.20 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 802.90 | 806.01 | 0.00 | ORB-short ORB[805.25,810.00] vol=1.7x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 804.50 | 805.77 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:55:00 | 890.60 | 2026-02-10 11:00:00 | 892.32 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-11 09:45:00 | 872.45 | 2026-02-11 10:00:00 | 868.61 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-11 09:45:00 | 872.45 | 2026-02-11 10:35:00 | 872.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 11:10:00 | 866.00 | 2026-02-13 11:50:00 | 867.86 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-16 09:40:00 | 857.45 | 2026-02-16 10:40:00 | 853.94 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-16 09:40:00 | 857.45 | 2026-02-16 11:30:00 | 857.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:35:00 | 873.00 | 2026-02-17 10:00:00 | 870.55 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-18 10:30:00 | 871.90 | 2026-02-18 10:40:00 | 873.57 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-19 10:30:00 | 874.20 | 2026-02-19 10:45:00 | 875.75 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-23 11:10:00 | 884.70 | 2026-02-23 11:30:00 | 882.49 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-26 10:50:00 | 872.20 | 2026-02-26 11:00:00 | 869.41 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-26 10:50:00 | 872.20 | 2026-02-26 15:20:00 | 871.50 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2026-02-27 09:45:00 | 856.50 | 2026-02-27 09:55:00 | 858.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 09:55:00 | 825.10 | 2026-03-05 10:10:00 | 827.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-12 11:00:00 | 801.65 | 2026-03-12 11:05:00 | 799.75 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-13 09:50:00 | 788.45 | 2026-03-13 10:05:00 | 785.59 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-03-13 09:50:00 | 788.45 | 2026-03-13 10:30:00 | 788.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 11:05:00 | 765.05 | 2026-03-16 11:15:00 | 767.48 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-20 09:30:00 | 778.30 | 2026-03-20 09:40:00 | 775.78 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-27 11:10:00 | 820.00 | 2026-04-27 11:25:00 | 818.64 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-04-28 10:45:00 | 821.60 | 2026-04-28 11:00:00 | 820.35 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-04-30 10:30:00 | 799.05 | 2026-04-30 12:15:00 | 800.86 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-08 09:45:00 | 802.90 | 2026-05-08 09:55:00 | 804.50 | STOP_HIT | 1.00 | -0.20% |
