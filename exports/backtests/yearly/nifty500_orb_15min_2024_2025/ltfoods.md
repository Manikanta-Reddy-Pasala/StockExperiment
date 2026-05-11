# LT Foods Ltd. (LTFOODS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-07-09 15:25:00 (3021 bars)
- **Last close:** 274.20
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
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 5
- **Avg / median % per leg:** 0.38% / 0.22%
- **Sum % (uncompounded):** 5.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.38% | 2.3% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.38% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 5 | 62.5% | 1 | 3 | 4 | 0.38% | 3.0% |
| SELL @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 1 | 3 | 4 | 0.38% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.38% | 5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:35:00 | 215.60 | 214.55 | 0.00 | ORB-long ORB[212.00,215.00] vol=1.8x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 10:50:00 | 217.27 | 215.75 | 0.00 | T1 1.5R @ 217.27 |
| Target hit | 2024-05-14 15:20:00 | 222.85 | 219.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:40:00 | 231.30 | 228.60 | 0.00 | ORB-long ORB[227.70,229.70] vol=2.8x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-05-17 10:45:00 | 230.36 | 228.91 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:40:00 | 210.70 | 211.61 | 0.00 | ORB-short ORB[211.30,213.55] vol=2.9x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:15:00 | 209.62 | 211.25 | 0.00 | T1 1.5R @ 209.62 |
| Stop hit — per-position SL triggered | 2024-05-23 11:20:00 | 210.70 | 211.22 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:50:00 | 207.80 | 208.64 | 0.00 | ORB-short ORB[208.25,211.00] vol=5.2x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 09:55:00 | 206.20 | 207.50 | 0.00 | T1 1.5R @ 206.20 |
| Target hit | 2024-05-24 10:40:00 | 207.35 | 207.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2024-05-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:10:00 | 207.55 | 206.71 | 0.00 | ORB-long ORB[205.00,206.80] vol=6.1x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-05-28 11:55:00 | 206.70 | 207.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:40:00 | 260.70 | 261.43 | 0.00 | ORB-short ORB[261.20,264.13] vol=4.4x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:15:00 | 258.43 | 260.10 | 0.00 | T1 1.5R @ 258.43 |
| Stop hit — per-position SL triggered | 2024-06-26 10:35:00 | 260.70 | 260.03 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 262.30 | 260.85 | 0.00 | ORB-long ORB[257.50,261.39] vol=2.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-06-27 09:35:00 | 260.81 | 261.01 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:15:00 | 259.74 | 257.79 | 0.00 | ORB-long ORB[256.00,259.30] vol=2.4x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-06-28 10:20:00 | 258.48 | 257.88 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:40:00 | 250.00 | 251.37 | 0.00 | ORB-short ORB[250.20,253.30] vol=2.7x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 11:15:00 | 248.34 | 249.96 | 0.00 | T1 1.5R @ 248.34 |
| Stop hit — per-position SL triggered | 2024-07-05 12:05:00 | 250.00 | 248.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:35:00 | 215.60 | 2024-05-14 10:50:00 | 217.27 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-05-14 09:35:00 | 215.60 | 2024-05-14 15:20:00 | 222.85 | TARGET_HIT | 0.50 | 3.36% |
| BUY | retest1 | 2024-05-17 10:40:00 | 231.30 | 2024-05-17 10:45:00 | 230.36 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-05-23 10:40:00 | 210.70 | 2024-05-23 11:15:00 | 209.62 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-23 10:40:00 | 210.70 | 2024-05-23 11:20:00 | 210.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-24 09:50:00 | 207.80 | 2024-05-24 09:55:00 | 206.20 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-05-24 09:50:00 | 207.80 | 2024-05-24 10:40:00 | 207.35 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-05-28 10:10:00 | 207.55 | 2024-05-28 11:55:00 | 206.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-06-26 09:40:00 | 260.70 | 2024-06-26 10:15:00 | 258.43 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2024-06-26 09:40:00 | 260.70 | 2024-06-26 10:35:00 | 260.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 09:30:00 | 262.30 | 2024-06-27 09:35:00 | 260.81 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-06-28 10:15:00 | 259.74 | 2024-06-28 10:20:00 | 258.48 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-07-05 09:40:00 | 250.00 | 2024-07-05 11:15:00 | 248.34 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-07-05 09:40:00 | 250.00 | 2024-07-05 12:05:00 | 250.00 | STOP_HIT | 0.50 | 0.00% |
