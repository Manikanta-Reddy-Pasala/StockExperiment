# J.K. Cement Ltd. (JKCEMENT)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-06-09 15:25:00 (1575 bars)
- **Last close:** 5774.00
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 5
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 2.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.14% | 1.4% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.14% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.29% | 1.2% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.29% | 1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 6 | 42.9% | 1 | 8 | 5 | 0.18% | 2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-12 11:05:00 | 5156.00 | 5193.20 | 0.00 | ORB-short ORB[5160.00,5214.50] vol=5.1x ATR=22.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 11:55:00 | 5122.17 | 5179.21 | 0.00 | T1 1.5R @ 5122.17 |
| Stop hit — per-position SL triggered | 2025-05-12 12:40:00 | 5156.00 | 5166.40 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:30:00 | 5204.50 | 5155.20 | 0.00 | ORB-long ORB[5112.00,5189.50] vol=2.2x ATR=16.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 11:25:00 | 5229.48 | 5182.47 | 0.00 | T1 1.5R @ 5229.48 |
| Stop hit — per-position SL triggered | 2025-05-13 12:30:00 | 5204.50 | 5194.96 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:35:00 | 5292.00 | 5245.32 | 0.00 | ORB-long ORB[5202.00,5276.50] vol=1.9x ATR=15.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 11:00:00 | 5315.80 | 5263.27 | 0.00 | T1 1.5R @ 5315.80 |
| Stop hit — per-position SL triggered | 2025-05-14 12:25:00 | 5292.00 | 5282.72 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:55:00 | 5384.50 | 5340.69 | 0.00 | ORB-long ORB[5294.00,5362.00] vol=2.7x ATR=18.68 |
| Stop hit — per-position SL triggered | 2025-05-15 11:00:00 | 5365.82 | 5341.30 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 10:40:00 | 5209.50 | 5214.78 | 0.00 | ORB-short ORB[5230.50,5285.00] vol=2.3x ATR=17.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 11:50:00 | 5182.87 | 5208.59 | 0.00 | T1 1.5R @ 5182.87 |
| Stop hit — per-position SL triggered | 2025-05-16 14:50:00 | 5209.50 | 5191.84 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 10:55:00 | 5310.00 | 5273.28 | 0.00 | ORB-long ORB[5228.50,5306.50] vol=3.2x ATR=12.10 |
| Stop hit — per-position SL triggered | 2025-05-20 11:20:00 | 5297.90 | 5279.96 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:55:00 | 5498.00 | 5488.53 | 0.00 | ORB-long ORB[5423.00,5497.50] vol=17.4x ATR=16.22 |
| Stop hit — per-position SL triggered | 2025-06-03 13:00:00 | 5481.78 | 5489.66 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:35:00 | 5651.00 | 5622.47 | 0.00 | ORB-long ORB[5591.50,5620.00] vol=8.1x ATR=13.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 10:55:00 | 5671.66 | 5626.03 | 0.00 | T1 1.5R @ 5671.66 |
| Target hit | 2025-06-05 15:20:00 | 5726.50 | 5665.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-06-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:50:00 | 5819.50 | 5791.77 | 0.00 | ORB-long ORB[5722.50,5775.00] vol=1.8x ATR=20.93 |
| Stop hit — per-position SL triggered | 2025-06-06 09:55:00 | 5798.57 | 5793.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-12 11:05:00 | 5156.00 | 2025-05-12 11:55:00 | 5122.17 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-05-12 11:05:00 | 5156.00 | 2025-05-12 12:40:00 | 5156.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-13 10:30:00 | 5204.50 | 2025-05-13 11:25:00 | 5229.48 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-13 10:30:00 | 5204.50 | 2025-05-13 12:30:00 | 5204.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-14 10:35:00 | 5292.00 | 2025-05-14 11:00:00 | 5315.80 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-14 10:35:00 | 5292.00 | 2025-05-14 12:25:00 | 5292.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 10:55:00 | 5384.50 | 2025-05-15 11:00:00 | 5365.82 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-05-16 10:40:00 | 5209.50 | 2025-05-16 11:50:00 | 5182.87 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-05-16 10:40:00 | 5209.50 | 2025-05-16 14:50:00 | 5209.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-20 10:55:00 | 5310.00 | 2025-05-20 11:20:00 | 5297.90 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-03 10:55:00 | 5498.00 | 2025-06-03 13:00:00 | 5481.78 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-05 10:35:00 | 5651.00 | 2025-06-05 10:55:00 | 5671.66 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-06-05 10:35:00 | 5651.00 | 2025-06-05 15:20:00 | 5726.50 | TARGET_HIT | 0.50 | 1.34% |
| BUY | retest1 | 2025-06-06 09:50:00 | 5819.50 | 2025-06-06 09:55:00 | 5798.57 | STOP_HIT | 1.00 | -0.36% |
