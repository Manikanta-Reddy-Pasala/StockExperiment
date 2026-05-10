# BPCL (BPCL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 303.20
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 4
- **Avg / median % per leg:** 0.02% / 0.00%
- **Sum % (uncompounded):** 0.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.14% | -0.7% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.14% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.10% | 0.9% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.10% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.02% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 388.95 | 385.93 | 0.00 | ORB-long ORB[382.90,387.00] vol=1.5x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-02-09 12:00:00 | 386.96 | 386.56 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:20:00 | 390.30 | 388.31 | 0.00 | ORB-long ORB[386.00,389.35] vol=1.9x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-02-11 10:30:00 | 389.38 | 388.53 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 369.95 | 370.82 | 0.00 | ORB-short ORB[371.10,373.50] vol=1.8x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-02-16 11:55:00 | 370.81 | 370.62 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 370.00 | 371.88 | 0.00 | ORB-short ORB[371.40,374.95] vol=1.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:15:00 | 368.61 | 370.51 | 0.00 | T1 1.5R @ 368.61 |
| Target hit | 2026-02-17 11:45:00 | 369.30 | 369.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 373.40 | 370.30 | 0.00 | ORB-long ORB[366.65,371.20] vol=2.8x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:00:00 | 375.02 | 371.39 | 0.00 | T1 1.5R @ 375.02 |
| Stop hit — per-position SL triggered | 2026-02-23 11:45:00 | 373.40 | 372.56 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:10:00 | 356.25 | 357.98 | 0.00 | ORB-short ORB[356.60,361.35] vol=3.0x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 13:20:00 | 353.93 | 356.83 | 0.00 | T1 1.5R @ 353.93 |
| Stop hit — per-position SL triggered | 2026-03-05 14:40:00 | 356.25 | 356.32 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:05:00 | 323.45 | 325.94 | 0.00 | ORB-short ORB[325.00,328.90] vol=1.9x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-03-11 12:20:00 | 324.33 | 325.53 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:05:00 | 320.80 | 321.88 | 0.00 | ORB-short ORB[322.25,325.35] vol=3.5x ATR=1.00 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 321.80 | 321.79 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 314.30 | 310.51 | 0.00 | ORB-long ORB[306.70,311.25] vol=2.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 313.15 | 311.40 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 296.25 | 298.17 | 0.00 | ORB-short ORB[296.75,300.45] vol=2.2x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:55:00 | 294.75 | 297.78 | 0.00 | T1 1.5R @ 294.75 |
| Stop hit — per-position SL triggered | 2026-05-05 12:00:00 | 296.25 | 296.53 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 388.95 | 2026-02-09 12:00:00 | 386.96 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-02-11 10:20:00 | 390.30 | 2026-02-11 10:30:00 | 389.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-16 11:15:00 | 369.95 | 2026-02-16 11:55:00 | 370.81 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-17 09:30:00 | 370.00 | 2026-02-17 10:15:00 | 368.61 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-17 09:30:00 | 370.00 | 2026-02-17 11:45:00 | 369.30 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2026-02-23 10:50:00 | 373.40 | 2026-02-23 11:00:00 | 375.02 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-23 10:50:00 | 373.40 | 2026-02-23 11:45:00 | 373.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 11:10:00 | 356.25 | 2026-03-05 13:20:00 | 353.93 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-05 11:10:00 | 356.25 | 2026-03-05 14:40:00 | 356.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:05:00 | 323.45 | 2026-03-11 12:20:00 | 324.33 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-13 11:05:00 | 320.80 | 2026-03-13 11:25:00 | 321.80 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-27 09:50:00 | 314.30 | 2026-04-27 10:05:00 | 313.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-05-05 10:35:00 | 296.25 | 2026-05-05 10:55:00 | 294.75 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-05-05 10:35:00 | 296.25 | 2026-05-05 12:00:00 | 296.25 | STOP_HIT | 0.50 | 0.00% |
