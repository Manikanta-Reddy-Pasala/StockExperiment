# Karur Vysya Bank Ltd. (KARURVYSYA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 304.80
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
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 0.44% / 0.00%
- **Sum % (uncompounded):** 6.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.65% | 6.5% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.65% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.07% | -0.3% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.07% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.44% | 6.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 319.90 | 322.12 | 0.00 | ORB-short ORB[322.65,326.90] vol=1.9x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:15:00 | 318.33 | 321.50 | 0.00 | T1 1.5R @ 318.33 |
| Stop hit — per-position SL triggered | 2026-02-10 11:25:00 | 319.90 | 320.87 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 319.30 | 322.32 | 0.00 | ORB-short ORB[321.50,325.85] vol=4.5x ATR=1.01 |
| Stop hit — per-position SL triggered | 2026-02-23 11:35:00 | 320.31 | 321.85 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 339.60 | 336.95 | 0.00 | ORB-long ORB[334.60,338.60] vol=1.7x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:35:00 | 341.62 | 338.66 | 0.00 | T1 1.5R @ 341.62 |
| Stop hit — per-position SL triggered | 2026-02-25 09:50:00 | 339.60 | 339.60 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 339.40 | 337.81 | 0.00 | ORB-long ORB[336.00,338.40] vol=1.8x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-02-26 09:45:00 | 338.27 | 338.24 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 286.70 | 285.50 | 0.00 | ORB-long ORB[283.35,285.75] vol=2.7x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-03-18 09:45:00 | 285.53 | 285.73 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:40:00 | 283.15 | 279.70 | 0.00 | ORB-long ORB[274.30,278.50] vol=3.9x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:10:00 | 285.57 | 280.84 | 0.00 | T1 1.5R @ 285.57 |
| Target hit | 2026-03-25 14:35:00 | 285.95 | 286.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 284.25 | 280.49 | 0.00 | ORB-long ORB[277.10,281.00] vol=1.9x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:55:00 | 286.23 | 281.53 | 0.00 | T1 1.5R @ 286.23 |
| Target hit | 2026-03-27 15:20:00 | 298.15 | 290.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 276.55 | 275.35 | 0.00 | ORB-long ORB[273.30,275.85] vol=8.1x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-04-21 09:45:00 | 275.42 | 275.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 301.25 | 298.89 | 0.00 | ORB-long ORB[297.05,299.60] vol=4.0x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 300.15 | 299.23 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:55:00 | 291.50 | 294.30 | 0.00 | ORB-short ORB[293.45,297.55] vol=2.3x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 292.82 | 294.11 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:50:00 | 319.90 | 2026-02-10 11:15:00 | 318.33 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-10 10:50:00 | 319.90 | 2026-02-10 11:25:00 | 319.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:15:00 | 319.30 | 2026-02-23 11:35:00 | 320.31 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-25 09:30:00 | 339.60 | 2026-02-25 09:35:00 | 341.62 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-25 09:30:00 | 339.60 | 2026-02-25 09:50:00 | 339.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:30:00 | 339.40 | 2026-02-26 09:45:00 | 338.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-18 09:30:00 | 286.70 | 2026-03-18 09:45:00 | 285.53 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-25 09:40:00 | 283.15 | 2026-03-25 10:10:00 | 285.57 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2026-03-25 09:40:00 | 283.15 | 2026-03-25 14:35:00 | 285.95 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2026-03-27 11:10:00 | 284.25 | 2026-03-27 11:55:00 | 286.23 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-27 11:10:00 | 284.25 | 2026-03-27 15:20:00 | 298.15 | TARGET_HIT | 0.50 | 4.89% |
| BUY | retest1 | 2026-04-21 09:30:00 | 276.55 | 2026-04-21 09:45:00 | 275.42 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-27 09:50:00 | 301.25 | 2026-04-27 10:05:00 | 300.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-30 10:55:00 | 291.50 | 2026-04-30 11:15:00 | 292.82 | STOP_HIT | 1.00 | -0.45% |
