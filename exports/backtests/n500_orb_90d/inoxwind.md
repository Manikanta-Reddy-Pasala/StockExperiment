# Inox Wind Ltd. (INOXWIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 103.65
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
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 5
- **Avg / median % per leg:** 0.28% / 0.48%
- **Sum % (uncompounded):** 3.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.63% | 2.5% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.63% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.10% | 0.8% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.10% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 6 | 50.0% | 1 | 6 | 5 | 0.28% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 109.22 | 109.78 | 0.00 | ORB-short ORB[109.89,110.99] vol=5.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-02-12 13:00:00 | 109.50 | 109.65 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:30:00 | 95.22 | 96.24 | 0.00 | ORB-short ORB[96.55,97.67] vol=1.6x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:40:00 | 94.73 | 96.04 | 0.00 | T1 1.5R @ 94.73 |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 95.22 | 95.49 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 84.95 | 85.61 | 0.00 | ORB-short ORB[85.75,87.00] vol=1.5x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 84.46 | 85.45 | 0.00 | T1 1.5R @ 84.46 |
| Stop hit — per-position SL triggered | 2026-03-05 12:25:00 | 84.95 | 85.09 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 80.33 | 79.60 | 0.00 | ORB-long ORB[78.39,79.50] vol=2.2x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:35:00 | 80.74 | 79.73 | 0.00 | T1 1.5R @ 80.74 |
| Stop hit — per-position SL triggered | 2026-03-18 12:05:00 | 80.33 | 79.86 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:55:00 | 98.35 | 97.93 | 0.00 | ORB-long ORB[97.26,98.29] vol=1.7x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:00:00 | 98.86 | 98.03 | 0.00 | T1 1.5R @ 98.86 |
| Target hit | 2026-04-21 11:00:00 | 99.81 | 99.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 100.74 | 101.66 | 0.00 | ORB-short ORB[101.32,102.78] vol=2.7x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-04-24 10:05:00 | 101.24 | 101.18 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 103.34 | 104.11 | 0.00 | ORB-short ORB[103.81,105.14] vol=3.5x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 102.84 | 104.03 | 0.00 | T1 1.5R @ 102.84 |
| Stop hit — per-position SL triggered | 2026-04-28 14:40:00 | 103.34 | 103.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:15:00 | 109.22 | 2026-02-12 13:00:00 | 109.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-23 10:30:00 | 95.22 | 2026-02-23 10:40:00 | 94.73 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-23 10:30:00 | 95.22 | 2026-02-23 12:15:00 | 95.22 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:40:00 | 84.95 | 2026-03-05 11:25:00 | 84.46 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-05 10:40:00 | 84.95 | 2026-03-05 12:25:00 | 84.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 11:15:00 | 80.33 | 2026-03-18 11:35:00 | 80.74 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-18 11:15:00 | 80.33 | 2026-03-18 12:05:00 | 80.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:55:00 | 98.35 | 2026-04-21 10:00:00 | 98.86 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-21 09:55:00 | 98.35 | 2026-04-21 11:00:00 | 99.81 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2026-04-24 09:30:00 | 100.74 | 2026-04-24 10:05:00 | 101.24 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-04-28 11:05:00 | 103.34 | 2026-04-28 11:25:00 | 102.84 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-28 11:05:00 | 103.34 | 2026-04-28 14:40:00 | 103.34 | STOP_HIT | 0.50 | 0.00% |
