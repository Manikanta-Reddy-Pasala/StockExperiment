# Oil India Ltd. (OIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 453.60
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 5
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 3.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.38% | 3.4% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.38% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.06% | 0.4% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.06% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 8 | 47.1% | 3 | 9 | 5 | 0.23% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 489.10 | 492.72 | 0.00 | ORB-short ORB[495.95,501.00] vol=1.7x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-02-09 10:35:00 | 491.90 | 492.37 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 476.85 | 473.97 | 0.00 | ORB-long ORB[469.75,474.80] vol=1.9x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-02-26 09:40:00 | 475.43 | 474.13 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 477.90 | 474.76 | 0.00 | ORB-long ORB[470.70,476.95] vol=1.9x ATR=1.92 |
| Stop hit — per-position SL triggered | 2026-02-27 09:45:00 | 475.98 | 475.32 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:15:00 | 475.50 | 483.02 | 0.00 | ORB-short ORB[483.65,489.05] vol=1.8x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 477.13 | 482.03 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:05:00 | 471.45 | 468.12 | 0.00 | ORB-long ORB[466.75,471.15] vol=2.9x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:30:00 | 473.61 | 469.06 | 0.00 | T1 1.5R @ 473.61 |
| Target hit | 2026-03-11 15:20:00 | 481.80 | 478.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:40:00 | 485.90 | 481.73 | 0.00 | ORB-long ORB[476.45,483.55] vol=2.0x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:00:00 | 489.40 | 483.71 | 0.00 | T1 1.5R @ 489.40 |
| Stop hit — per-position SL triggered | 2026-03-12 10:10:00 | 485.90 | 484.00 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 468.20 | 468.64 | 0.00 | ORB-short ORB[470.55,475.45] vol=3.4x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:20:00 | 465.66 | 468.48 | 0.00 | T1 1.5R @ 465.66 |
| Stop hit — per-position SL triggered | 2026-03-27 12:10:00 | 468.20 | 468.09 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 464.05 | 465.40 | 0.00 | ORB-short ORB[465.50,468.95] vol=9.0x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-04-22 12:05:00 | 464.95 | 464.99 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 478.90 | 476.99 | 0.00 | ORB-long ORB[474.70,478.45] vol=1.5x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:40:00 | 481.09 | 478.22 | 0.00 | T1 1.5R @ 481.09 |
| Target hit | 2026-04-24 11:05:00 | 481.30 | 482.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 484.00 | 486.97 | 0.00 | ORB-short ORB[485.50,490.80] vol=2.0x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:55:00 | 481.75 | 485.42 | 0.00 | T1 1.5R @ 481.75 |
| Target hit | 2026-05-04 11:35:00 | 480.05 | 479.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2026-05-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:30:00 | 470.90 | 473.31 | 0.00 | ORB-short ORB[472.35,476.45] vol=2.3x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-05-06 10:35:00 | 472.17 | 473.16 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 456.00 | 454.47 | 0.00 | ORB-long ORB[450.50,455.55] vol=2.0x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 454.86 | 454.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-09 10:25:00 | 489.10 | 2026-02-09 10:35:00 | 491.90 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-02-26 09:35:00 | 476.85 | 2026-02-26 09:40:00 | 475.43 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-27 09:35:00 | 477.90 | 2026-02-27 09:45:00 | 475.98 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-06 11:15:00 | 475.50 | 2026-03-06 11:35:00 | 477.13 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-11 11:05:00 | 471.45 | 2026-03-11 11:30:00 | 473.61 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-11 11:05:00 | 471.45 | 2026-03-11 15:20:00 | 481.80 | TARGET_HIT | 0.50 | 2.20% |
| BUY | retest1 | 2026-03-12 09:40:00 | 485.90 | 2026-03-12 10:00:00 | 489.40 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-03-12 09:40:00 | 485.90 | 2026-03-12 10:10:00 | 485.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 11:00:00 | 468.20 | 2026-03-27 11:20:00 | 465.66 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-27 11:00:00 | 468.20 | 2026-03-27 12:10:00 | 468.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 10:55:00 | 464.05 | 2026-04-22 12:05:00 | 464.95 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-04-24 09:35:00 | 478.90 | 2026-04-24 09:40:00 | 481.09 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-24 09:35:00 | 478.90 | 2026-04-24 11:05:00 | 481.30 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-05-04 09:45:00 | 484.00 | 2026-05-04 09:55:00 | 481.75 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-05-04 09:45:00 | 484.00 | 2026-05-04 11:35:00 | 480.05 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2026-05-06 10:30:00 | 470.90 | 2026-05-06 10:35:00 | 472.17 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-07 11:05:00 | 456.00 | 2026-05-07 11:15:00 | 454.86 | STOP_HIT | 1.00 | -0.25% |
