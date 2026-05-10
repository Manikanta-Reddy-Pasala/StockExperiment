# Sonata Software Ltd. (SONATSOFTW)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 296.65
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 3
- **Avg / median % per leg:** 0.14% / 0.27%
- **Sum % (uncompounded):** 1.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.14% | 1.0% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.14% | 1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.15% | 0.6% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.15% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.14% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 315.45 | 318.29 | 0.00 | ORB-short ORB[317.00,321.65] vol=3.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-11 09:50:00 | 316.51 | 317.49 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 262.75 | 258.89 | 0.00 | ORB-long ORB[256.35,259.80] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-03-04 09:50:00 | 261.21 | 259.13 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 246.05 | 249.07 | 0.00 | ORB-short ORB[249.20,252.50] vol=1.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 246.97 | 249.01 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:35:00 | 233.70 | 235.30 | 0.00 | ORB-short ORB[234.80,238.00] vol=2.6x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 14:10:00 | 231.79 | 234.08 | 0.00 | T1 1.5R @ 231.79 |
| Target hit | 2026-03-19 15:20:00 | 232.55 | 233.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:00:00 | 273.75 | 269.69 | 0.00 | ORB-long ORB[267.19,270.45] vol=4.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-04-23 11:05:00 | 272.65 | 270.16 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:00:00 | 260.60 | 258.70 | 0.00 | ORB-long ORB[257.00,259.99] vol=1.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-04-28 10:40:00 | 259.56 | 259.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 262.70 | 260.19 | 0.00 | ORB-long ORB[257.15,260.54] vol=3.8x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:50:00 | 264.42 | 263.98 | 0.00 | T1 1.5R @ 264.42 |
| Target hit | 2026-04-29 10:00:00 | 264.64 | 264.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 259.75 | 258.53 | 0.00 | ORB-long ORB[255.50,258.45] vol=2.0x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:35:00 | 261.57 | 258.72 | 0.00 | T1 1.5R @ 261.57 |
| Target hit | 2026-05-04 12:10:00 | 260.45 | 260.81 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 315.45 | 2026-02-11 09:50:00 | 316.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-04 09:45:00 | 262.75 | 2026-03-04 09:50:00 | 261.21 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-03-06 10:45:00 | 246.05 | 2026-03-06 10:50:00 | 246.97 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-19 09:35:00 | 233.70 | 2026-03-19 14:10:00 | 231.79 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2026-03-19 09:35:00 | 233.70 | 2026-03-19 15:20:00 | 232.55 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-23 11:00:00 | 273.75 | 2026-04-23 11:05:00 | 272.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-28 10:00:00 | 260.60 | 2026-04-28 10:40:00 | 259.56 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-29 09:45:00 | 262.70 | 2026-04-29 09:50:00 | 264.42 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-29 09:45:00 | 262.70 | 2026-04-29 10:00:00 | 264.64 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-05-04 10:30:00 | 259.75 | 2026-05-04 10:35:00 | 261.57 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-05-04 10:30:00 | 259.75 | 2026-05-04 12:10:00 | 260.45 | TARGET_HIT | 0.50 | 0.27% |
