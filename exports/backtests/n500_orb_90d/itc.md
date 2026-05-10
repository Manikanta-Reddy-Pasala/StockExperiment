# ITC Ltd. (ITC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 307.20
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
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 6
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.12% | 1.7% |
| BUY @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.12% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.10% | 0.6% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.10% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.11% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 317.30 | 315.27 | 0.00 | ORB-long ORB[313.15,314.90] vol=1.5x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:35:00 | 318.15 | 315.67 | 0.00 | T1 1.5R @ 318.15 |
| Stop hit — per-position SL triggered | 2026-02-16 11:50:00 | 317.30 | 315.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 321.75 | 320.00 | 0.00 | ORB-long ORB[318.15,321.00] vol=2.9x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:40:00 | 322.91 | 320.81 | 0.00 | T1 1.5R @ 322.91 |
| Target hit | 2026-02-17 11:00:00 | 324.30 | 324.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 330.50 | 328.49 | 0.00 | ORB-long ORB[324.80,328.20] vol=1.6x ATR=0.81 |
| Stop hit — per-position SL triggered | 2026-02-18 10:20:00 | 329.69 | 328.94 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 323.25 | 324.43 | 0.00 | ORB-short ORB[324.60,326.60] vol=1.6x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:50:00 | 322.19 | 323.82 | 0.00 | T1 1.5R @ 322.19 |
| Target hit | 2026-02-25 15:20:00 | 319.50 | 321.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 309.75 | 310.20 | 0.00 | ORB-short ORB[309.80,313.60] vol=2.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 310.46 | 310.20 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:10:00 | 304.15 | 305.31 | 0.00 | ORB-short ORB[305.25,309.00] vol=2.2x ATR=0.76 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 304.91 | 305.22 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 305.65 | 306.64 | 0.00 | ORB-short ORB[305.70,309.95] vol=1.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2026-03-17 11:50:00 | 306.23 | 306.33 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:15:00 | 307.15 | 304.53 | 0.00 | ORB-long ORB[299.70,301.60] vol=2.4x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:35:00 | 308.13 | 305.14 | 0.00 | T1 1.5R @ 308.13 |
| Stop hit — per-position SL triggered | 2026-03-20 12:05:00 | 307.15 | 305.74 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:25:00 | 296.15 | 295.34 | 0.00 | ORB-long ORB[292.25,294.75] vol=1.6x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:40:00 | 297.18 | 295.49 | 0.00 | T1 1.5R @ 297.18 |
| Target hit | 2026-03-25 12:55:00 | 296.55 | 296.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2026-03-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:40:00 | 291.85 | 293.30 | 0.00 | ORB-short ORB[293.20,295.25] vol=1.7x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-03-27 10:45:00 | 292.46 | 293.18 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:50:00 | 304.90 | 303.60 | 0.00 | ORB-long ORB[302.80,304.05] vol=1.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 304.36 | 303.80 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:00:00 | 303.05 | 302.09 | 0.00 | ORB-long ORB[300.65,302.85] vol=1.6x ATR=0.67 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 302.38 | 302.20 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:10:00 | 303.70 | 302.68 | 0.00 | ORB-long ORB[302.00,303.45] vol=1.5x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:20:00 | 304.37 | 303.01 | 0.00 | T1 1.5R @ 304.37 |
| Stop hit — per-position SL triggered | 2026-04-16 11:45:00 | 303.70 | 303.20 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 304.95 | 304.55 | 0.00 | ORB-long ORB[302.55,304.55] vol=4.4x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-04-28 11:25:00 | 304.59 | 304.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 11:05:00 | 317.30 | 2026-02-16 11:35:00 | 318.15 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-16 11:05:00 | 317.30 | 2026-02-16 11:50:00 | 317.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:30:00 | 321.75 | 2026-02-17 09:40:00 | 322.91 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-17 09:30:00 | 321.75 | 2026-02-17 11:00:00 | 324.30 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2026-02-18 10:00:00 | 330.50 | 2026-02-18 10:20:00 | 329.69 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-25 09:45:00 | 323.25 | 2026-02-25 10:50:00 | 322.19 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-25 09:45:00 | 323.25 | 2026-02-25 15:20:00 | 319.50 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2026-03-05 11:00:00 | 309.75 | 2026-03-05 11:15:00 | 310.46 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-10 10:10:00 | 304.15 | 2026-03-10 10:15:00 | 304.91 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-17 11:15:00 | 305.65 | 2026-03-17 11:50:00 | 306.23 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-03-20 11:15:00 | 307.15 | 2026-03-20 11:35:00 | 308.13 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-03-20 11:15:00 | 307.15 | 2026-03-20 12:05:00 | 307.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 10:25:00 | 296.15 | 2026-03-25 10:40:00 | 297.18 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-03-25 10:25:00 | 296.15 | 2026-03-25 12:55:00 | 296.55 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-03-27 10:40:00 | 291.85 | 2026-03-27 10:45:00 | 292.46 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-04-10 09:50:00 | 304.90 | 2026-04-10 10:05:00 | 304.36 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-04-15 10:00:00 | 303.05 | 2026-04-15 10:15:00 | 302.38 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-16 11:10:00 | 303.70 | 2026-04-16 11:20:00 | 304.37 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2026-04-16 11:10:00 | 303.70 | 2026-04-16 11:45:00 | 303.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 11:15:00 | 304.95 | 2026-04-28 11:25:00 | 304.59 | STOP_HIT | 1.00 | -0.12% |
