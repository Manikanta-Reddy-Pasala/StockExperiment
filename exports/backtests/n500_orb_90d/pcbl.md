# PCBL Chemical Ltd. (PCBL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 306.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 5
- **Avg / median % per leg:** 1.01% / 0.58%
- **Sum % (uncompounded):** 11.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 2.12% | 8.5% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 2.12% | 8.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.38% | 2.7% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.38% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 7 | 63.6% | 2 | 4 | 5 | 1.01% | 11.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 278.40 | 275.33 | 0.00 | ORB-long ORB[272.75,275.75] vol=2.1x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:45:00 | 280.30 | 276.46 | 0.00 | T1 1.5R @ 280.30 |
| Target hit | 2026-02-09 15:20:00 | 298.45 | 297.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 266.80 | 268.89 | 0.00 | ORB-short ORB[269.00,271.35] vol=1.8x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 265.25 | 268.38 | 0.00 | T1 1.5R @ 265.25 |
| Stop hit — per-position SL triggered | 2026-03-13 10:25:00 | 266.80 | 268.14 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:50:00 | 248.80 | 251.07 | 0.00 | ORB-short ORB[249.40,253.00] vol=1.9x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 10:05:00 | 246.51 | 250.01 | 0.00 | T1 1.5R @ 246.51 |
| Target hit | 2026-03-19 15:20:00 | 246.00 | 246.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 284.01 | 286.07 | 0.00 | ORB-short ORB[284.50,287.99] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-04-17 13:45:00 | 285.12 | 285.38 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 296.99 | 293.23 | 0.00 | ORB-long ORB[290.65,294.95] vol=4.5x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:00:00 | 298.70 | 295.84 | 0.00 | T1 1.5R @ 298.70 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 296.99 | 296.12 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 303.10 | 305.32 | 0.00 | ORB-short ORB[304.50,308.00] vol=2.4x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 12:20:00 | 301.73 | 304.91 | 0.00 | T1 1.5R @ 301.73 |
| Stop hit — per-position SL triggered | 2026-05-06 13:50:00 | 303.10 | 304.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 278.40 | 2026-02-09 10:45:00 | 280.30 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-02-09 10:40:00 | 278.40 | 2026-02-09 15:20:00 | 298.45 | TARGET_HIT | 0.50 | 7.20% |
| SELL | retest1 | 2026-03-13 10:00:00 | 266.80 | 2026-03-13 10:10:00 | 265.25 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-13 10:00:00 | 266.80 | 2026-03-13 10:25:00 | 266.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 09:50:00 | 248.80 | 2026-03-19 10:05:00 | 246.51 | PARTIAL | 0.50 | 0.92% |
| SELL | retest1 | 2026-03-19 09:50:00 | 248.80 | 2026-03-19 15:20:00 | 246.00 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2026-04-17 11:05:00 | 284.01 | 2026-04-17 13:45:00 | 285.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-28 10:55:00 | 296.99 | 2026-04-28 11:00:00 | 298.70 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-28 10:55:00 | 296.99 | 2026-04-28 11:05:00 | 296.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:05:00 | 303.10 | 2026-05-06 12:20:00 | 301.73 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-05-06 11:05:00 | 303.10 | 2026-05-06 13:50:00 | 303.10 | STOP_HIT | 0.50 | 0.00% |
