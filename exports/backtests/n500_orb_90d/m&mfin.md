# Mahindra & Mahindra Financial Services Ltd. (M&MFIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 339.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** 0.03% / -0.25%
- **Sum % (uncompounded):** 0.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.33% | 1.6% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.33% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.35% | -1.4% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.35% | -1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.03% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 383.85 | 383.02 | 0.00 | ORB-long ORB[379.20,383.75] vol=2.1x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-02-18 11:50:00 | 382.87 | 383.07 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 375.70 | 378.90 | 0.00 | ORB-short ORB[376.00,380.00] vol=3.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-02-20 12:35:00 | 377.33 | 377.55 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 375.20 | 371.04 | 0.00 | ORB-long ORB[364.70,369.15] vol=1.7x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:30:00 | 377.46 | 372.72 | 0.00 | T1 1.5R @ 377.46 |
| Target hit | 2026-02-25 14:40:00 | 378.50 | 378.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:35:00 | 300.30 | 299.04 | 0.00 | ORB-long ORB[297.65,300.00] vol=1.5x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:00:00 | 301.54 | 299.58 | 0.00 | T1 1.5R @ 301.54 |
| Stop hit — per-position SL triggered | 2026-04-17 11:05:00 | 300.30 | 299.59 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:00:00 | 295.95 | 297.83 | 0.00 | ORB-short ORB[296.90,300.45] vol=4.0x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-04-23 11:55:00 | 296.77 | 297.40 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 294.00 | 297.02 | 0.00 | ORB-short ORB[296.05,299.30] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 294.97 | 296.79 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 299.25 | 301.89 | 0.00 | ORB-short ORB[300.80,304.20] vol=2.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-05-05 12:05:00 | 300.31 | 300.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-18 11:15:00 | 383.85 | 2026-02-18 11:50:00 | 382.87 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-20 10:55:00 | 375.70 | 2026-02-20 12:35:00 | 377.33 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-25 09:55:00 | 375.20 | 2026-02-25 10:30:00 | 377.46 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-25 09:55:00 | 375.20 | 2026-02-25 14:40:00 | 378.50 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2026-04-17 10:35:00 | 300.30 | 2026-04-17 11:00:00 | 301.54 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-17 10:35:00 | 300.30 | 2026-04-17 11:05:00 | 300.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:00:00 | 295.95 | 2026-04-23 11:55:00 | 296.77 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-24 11:00:00 | 294.00 | 2026-04-24 11:30:00 | 294.97 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-05-05 11:00:00 | 299.25 | 2026-05-05 12:05:00 | 300.31 | STOP_HIT | 1.00 | -0.35% |
