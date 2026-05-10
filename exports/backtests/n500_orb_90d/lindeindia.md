# Linde India Ltd. (LINDEINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 7765.00
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
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 6
- **Avg / median % per leg:** 0.28% / 0.29%
- **Sum % (uncompounded):** 5.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 9 | 64.3% | 4 | 5 | 5 | 0.45% | 6.3% |
| BUY @ 2nd Alert (retest1) | 14 | 9 | 64.3% | 4 | 5 | 5 | 0.45% | 6.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.13% | -0.8% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.13% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 10 | 50.0% | 4 | 10 | 6 | 0.28% | 5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 6795.00 | 6818.37 | 0.00 | ORB-short ORB[6799.50,6880.00] vol=1.5x ATR=17.35 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 6812.35 | 6817.59 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 6811.00 | 6778.34 | 0.00 | ORB-long ORB[6725.50,6798.50] vol=1.6x ATR=25.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:00:00 | 6848.63 | 6818.17 | 0.00 | T1 1.5R @ 6848.63 |
| Target hit | 2026-02-20 10:55:00 | 6844.00 | 6845.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 6645.50 | 6673.20 | 0.00 | ORB-short ORB[6656.00,6734.50] vol=1.7x ATR=26.24 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 6671.74 | 6671.77 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 6811.50 | 6841.50 | 0.00 | ORB-short ORB[6829.00,6910.50] vol=1.9x ATR=18.36 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 6829.86 | 6840.33 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:25:00 | 6649.00 | 6579.68 | 0.00 | ORB-long ORB[6530.00,6605.00] vol=2.4x ATR=26.88 |
| Stop hit — per-position SL triggered | 2026-03-04 10:50:00 | 6622.12 | 6602.04 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 6665.00 | 6687.33 | 0.00 | ORB-short ORB[6669.00,6726.00] vol=1.7x ATR=17.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:50:00 | 6638.44 | 6682.05 | 0.00 | T1 1.5R @ 6638.44 |
| Stop hit — per-position SL triggered | 2026-03-05 12:20:00 | 6665.00 | 6680.49 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 7257.50 | 7198.39 | 0.00 | ORB-long ORB[7150.00,7250.00] vol=2.6x ATR=39.97 |
| Stop hit — per-position SL triggered | 2026-03-17 10:40:00 | 7217.53 | 7199.83 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:05:00 | 7293.50 | 7224.88 | 0.00 | ORB-long ORB[7155.00,7259.00] vol=2.3x ATR=33.00 |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 7260.50 | 7228.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 7350.00 | 7305.30 | 0.00 | ORB-long ORB[7251.00,7304.00] vol=5.9x ATR=25.25 |
| Stop hit — per-position SL triggered | 2026-04-15 09:45:00 | 7324.75 | 7313.71 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 7118.00 | 7077.12 | 0.00 | ORB-long ORB[7021.00,7104.00] vol=3.1x ATR=21.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:50:00 | 7150.85 | 7091.60 | 0.00 | T1 1.5R @ 7150.85 |
| Target hit | 2026-04-22 15:20:00 | 7286.50 | 7201.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 10:05:00 | 7321.50 | 7286.45 | 0.00 | ORB-long ORB[7261.00,7307.50] vol=1.6x ATR=23.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:45:00 | 7356.19 | 7315.56 | 0.00 | T1 1.5R @ 7356.19 |
| Target hit | 2026-04-24 10:55:00 | 7343.00 | 7345.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 7234.00 | 7249.84 | 0.00 | ORB-short ORB[7235.00,7276.50] vol=1.9x ATR=17.97 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 7251.97 | 7249.63 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 7575.00 | 7531.32 | 0.00 | ORB-long ORB[7459.00,7549.00] vol=5.7x ATR=26.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:35:00 | 7615.14 | 7555.33 | 0.00 | T1 1.5R @ 7615.14 |
| Stop hit — per-position SL triggered | 2026-05-06 09:50:00 | 7575.00 | 7567.27 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:35:00 | 7665.00 | 7622.10 | 0.00 | ORB-long ORB[7580.00,7651.50] vol=2.1x ATR=24.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:45:00 | 7702.14 | 7635.16 | 0.00 | T1 1.5R @ 7702.14 |
| Target hit | 2026-05-07 12:15:00 | 7850.00 | 7854.64 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 10:45:00 | 6795.00 | 2026-02-18 10:50:00 | 6812.35 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-20 09:40:00 | 6811.00 | 2026-02-20 10:00:00 | 6848.63 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-20 09:40:00 | 6811.00 | 2026-02-20 10:55:00 | 6844.00 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-24 09:30:00 | 6645.50 | 2026-02-24 09:35:00 | 6671.74 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-27 11:10:00 | 6811.50 | 2026-02-27 11:15:00 | 6829.86 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-04 10:25:00 | 6649.00 | 2026-03-04 10:50:00 | 6622.12 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-05 11:00:00 | 6665.00 | 2026-03-05 11:50:00 | 6638.44 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-03-05 11:00:00 | 6665.00 | 2026-03-05 12:20:00 | 6665.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:35:00 | 7257.50 | 2026-03-17 10:40:00 | 7217.53 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-04-08 10:05:00 | 7293.50 | 2026-04-08 10:15:00 | 7260.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-15 09:35:00 | 7350.00 | 2026-04-15 09:45:00 | 7324.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-22 09:45:00 | 7118.00 | 2026-04-22 09:50:00 | 7150.85 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-22 09:45:00 | 7118.00 | 2026-04-22 15:20:00 | 7286.50 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2026-04-24 10:05:00 | 7321.50 | 2026-04-24 10:45:00 | 7356.19 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-24 10:05:00 | 7321.50 | 2026-04-24 10:55:00 | 7343.00 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2026-04-29 09:55:00 | 7234.00 | 2026-04-29 10:20:00 | 7251.97 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-06 09:30:00 | 7575.00 | 2026-05-06 09:35:00 | 7615.14 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-05-06 09:30:00 | 7575.00 | 2026-05-06 09:50:00 | 7575.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 10:35:00 | 7665.00 | 2026-05-07 10:45:00 | 7702.14 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-05-07 10:35:00 | 7665.00 | 2026-05-07 12:15:00 | 7850.00 | TARGET_HIT | 0.50 | 2.41% |
