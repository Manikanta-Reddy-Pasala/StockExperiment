# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 282.50
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / Stop hits / Partials:** 2 / 6 / 4
- **Avg / median % per leg:** 0.30% / 0.36%
- **Sum % (uncompounded):** 3.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.38% | 2.6% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.38% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.19% | 1.0% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.19% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.30% | 3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:55:00 | 258.95 | 259.29 | 0.00 | ORB-short ORB[259.15,262.00] vol=2.0x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:10:00 | 257.65 | 259.17 | 0.00 | T1 1.5R @ 257.65 |
| Target hit | 2026-02-11 15:20:00 | 257.95 | 258.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 249.90 | 248.41 | 0.00 | ORB-long ORB[247.00,249.25] vol=4.0x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-02-26 10:50:00 | 248.70 | 248.71 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:50:00 | 226.60 | 229.16 | 0.00 | ORB-short ORB[229.05,230.68] vol=3.2x ATR=0.68 |
| Stop hit — per-position SL triggered | 2026-03-06 10:55:00 | 227.28 | 228.73 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:40:00 | 225.99 | 225.56 | 0.00 | ORB-long ORB[223.85,225.44] vol=7.3x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:20:00 | 227.43 | 226.01 | 0.00 | T1 1.5R @ 227.43 |
| Target hit | 2026-03-10 15:20:00 | 232.12 | 228.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 250.06 | 248.65 | 0.00 | ORB-long ORB[247.00,249.75] vol=2.8x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-04-17 11:10:00 | 249.22 | 248.70 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 260.13 | 256.78 | 0.00 | ORB-long ORB[253.21,257.04] vol=4.4x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-04-22 11:30:00 | 259.09 | 258.19 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 259.00 | 261.28 | 0.00 | ORB-short ORB[261.00,262.98] vol=2.0x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:30:00 | 258.06 | 261.04 | 0.00 | T1 1.5R @ 258.06 |
| Stop hit — per-position SL triggered | 2026-04-28 13:55:00 | 259.00 | 259.45 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 264.25 | 263.25 | 0.00 | ORB-long ORB[261.80,263.85] vol=2.5x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:35:00 | 265.54 | 263.57 | 0.00 | T1 1.5R @ 265.54 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 264.25 | 264.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:55:00 | 258.95 | 2026-02-11 10:10:00 | 257.65 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-11 09:55:00 | 258.95 | 2026-02-11 15:20:00 | 257.95 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-26 10:40:00 | 249.90 | 2026-02-26 10:50:00 | 248.70 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-06 10:50:00 | 226.60 | 2026-03-06 10:55:00 | 227.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-10 09:40:00 | 225.99 | 2026-03-10 10:20:00 | 227.43 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-03-10 09:40:00 | 225.99 | 2026-03-10 15:20:00 | 232.12 | TARGET_HIT | 0.50 | 2.71% |
| BUY | retest1 | 2026-04-17 11:00:00 | 250.06 | 2026-04-17 11:10:00 | 249.22 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-22 10:45:00 | 260.13 | 2026-04-22 11:30:00 | 259.09 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-28 11:10:00 | 259.00 | 2026-04-28 11:30:00 | 258.06 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-28 11:10:00 | 259.00 | 2026-04-28 13:55:00 | 259.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:30:00 | 264.25 | 2026-05-04 09:35:00 | 265.54 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-05-04 09:30:00 | 264.25 | 2026-05-04 10:10:00 | 264.25 | STOP_HIT | 0.50 | 0.00% |
