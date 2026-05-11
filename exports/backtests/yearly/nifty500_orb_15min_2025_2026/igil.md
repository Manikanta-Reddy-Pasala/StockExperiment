# International Gemmological Institute (India) Ltd. (IGIL)

## Backtest Summary

- **Window:** 2026-01-06 09:15:00 → 2026-05-08 15:25:00 (4650 bars)
- **Last close:** 352.90
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 2
- **Avg / median % per leg:** -0.16% / -0.38%
- **Sum % (uncompounded):** -1.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.20% | -1.2% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.20% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.11% | -0.6% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.11% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 2 | 18.2% | 0 | 9 | 2 | -0.16% | -1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 314.60 | 317.41 | 0.00 | ORB-short ORB[314.95,319.30] vol=1.9x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:05:00 | 312.98 | 316.78 | 0.00 | T1 1.5R @ 312.98 |
| Stop hit — per-position SL triggered | 2026-01-08 14:35:00 | 314.60 | 315.19 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-01-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 10:10:00 | 304.55 | 302.50 | 0.00 | ORB-long ORB[298.20,302.00] vol=1.7x ATR=1.55 |
| Stop hit — per-position SL triggered | 2026-01-22 12:05:00 | 303.00 | 303.09 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 325.00 | 327.72 | 0.00 | ORB-short ORB[327.65,331.80] vol=4.1x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-02-12 13:45:00 | 326.33 | 325.93 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:55:00 | 333.25 | 331.20 | 0.00 | ORB-long ORB[327.55,331.45] vol=2.9x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:55:00 | 335.43 | 332.39 | 0.00 | T1 1.5R @ 335.43 |
| Stop hit — per-position SL triggered | 2026-02-19 12:40:00 | 333.25 | 333.31 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:00:00 | 334.20 | 332.91 | 0.00 | ORB-long ORB[329.65,333.50] vol=2.0x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 332.92 | 333.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:30:00 | 339.70 | 336.97 | 0.00 | ORB-long ORB[331.80,336.80] vol=4.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-02-23 09:40:00 | 338.17 | 337.67 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 325.20 | 327.48 | 0.00 | ORB-short ORB[327.15,331.50] vol=1.6x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-02-25 09:50:00 | 326.04 | 327.41 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:10:00 | 335.60 | 331.43 | 0.00 | ORB-long ORB[327.85,331.50] vol=3.9x ATR=1.76 |
| Stop hit — per-position SL triggered | 2026-03-25 10:25:00 | 333.84 | 332.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-02 10:15:00 | 314.50 | 316.60 | 0.00 | ORB-short ORB[315.40,319.85] vol=1.8x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-04-02 10:50:00 | 315.80 | 316.31 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-01-08 11:10:00 | 314.60 | 2026-01-08 12:05:00 | 312.98 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-01-08 11:10:00 | 314.60 | 2026-01-08 14:35:00 | 314.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-22 10:10:00 | 304.55 | 2026-01-22 12:05:00 | 303.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-02-12 10:45:00 | 325.00 | 2026-02-12 13:45:00 | 326.33 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-19 09:55:00 | 333.25 | 2026-02-19 10:55:00 | 335.43 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-02-19 09:55:00 | 333.25 | 2026-02-19 12:40:00 | 333.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:00:00 | 334.20 | 2026-02-20 10:15:00 | 332.92 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-23 09:30:00 | 339.70 | 2026-02-23 09:40:00 | 338.17 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-02-25 09:45:00 | 325.20 | 2026-02-25 09:50:00 | 326.04 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-25 10:10:00 | 335.60 | 2026-03-25 10:25:00 | 333.84 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-04-02 10:15:00 | 314.50 | 2026-04-02 10:50:00 | 315.80 | STOP_HIT | 1.00 | -0.41% |
