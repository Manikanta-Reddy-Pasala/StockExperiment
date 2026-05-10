# Transformers And Rectifiers (India) Ltd. (TARIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 325.05
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** 0.31% / -0.54%
- **Sum % (uncompounded):** 2.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.47% | 2.8% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.47% | 2.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.61% | -0.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.61% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.31% | 2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 305.78 | 303.79 | 0.00 | ORB-long ORB[301.20,304.77] vol=4.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 303.96 | 304.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:35:00 | 290.95 | 293.16 | 0.00 | ORB-short ORB[291.55,295.90] vol=1.8x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-03-05 10:05:00 | 292.72 | 292.39 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-04-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:50:00 | 280.00 | 275.84 | 0.00 | ORB-long ORB[272.00,275.65] vol=1.5x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:05:00 | 282.67 | 276.92 | 0.00 | T1 1.5R @ 282.67 |
| Target hit | 2026-04-13 15:20:00 | 291.25 | 289.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 349.20 | 345.45 | 0.00 | ORB-long ORB[343.10,347.50] vol=5.0x ATR=1.66 |
| Stop hit — per-position SL triggered | 2026-04-28 11:10:00 | 347.54 | 345.59 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 336.00 | 333.30 | 0.00 | ORB-long ORB[329.20,333.80] vol=4.3x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 334.13 | 333.70 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 320.20 | 316.44 | 0.00 | ORB-long ORB[313.50,317.70] vol=1.7x ATR=1.73 |
| Stop hit — per-position SL triggered | 2026-05-07 09:45:00 | 318.47 | 316.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-24 09:35:00 | 305.78 | 2026-02-24 09:40:00 | 303.96 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2026-03-05 09:35:00 | 290.95 | 2026-03-05 10:05:00 | 292.72 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2026-04-13 09:50:00 | 280.00 | 2026-04-13 10:05:00 | 282.67 | PARTIAL | 0.50 | 0.96% |
| BUY | retest1 | 2026-04-13 09:50:00 | 280.00 | 2026-04-13 15:20:00 | 291.25 | TARGET_HIT | 0.50 | 4.02% |
| BUY | retest1 | 2026-04-28 11:05:00 | 349.20 | 2026-04-28 11:10:00 | 347.54 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-05-05 09:35:00 | 336.00 | 2026-05-05 09:45:00 | 334.13 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2026-05-07 09:40:00 | 320.20 | 2026-05-07 09:45:00 | 318.47 | STOP_HIT | 1.00 | -0.54% |
