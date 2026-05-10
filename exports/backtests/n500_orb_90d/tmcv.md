# Tata Motors Ltd. (TMCV)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 431.15
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
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 6
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 3.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.03% | -0.2% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.03% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.32% | 3.5% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.32% | 3.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.18% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 496.75 | 493.21 | 0.00 | ORB-long ORB[483.65,491.10] vol=1.7x ATR=2.56 |
| Stop hit — per-position SL triggered | 2026-02-12 11:30:00 | 494.19 | 493.62 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 479.85 | 483.00 | 0.00 | ORB-short ORB[480.65,486.75] vol=4.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 481.33 | 482.69 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 482.80 | 484.61 | 0.00 | ORB-short ORB[485.00,490.00] vol=1.7x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:45:00 | 480.40 | 483.88 | 0.00 | T1 1.5R @ 480.40 |
| Target hit | 2026-02-19 15:20:00 | 473.25 | 480.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:15:00 | 477.35 | 478.83 | 0.00 | ORB-short ORB[478.40,483.35] vol=2.9x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:30:00 | 474.45 | 478.10 | 0.00 | T1 1.5R @ 474.45 |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 477.35 | 476.86 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:15:00 | 477.25 | 478.36 | 0.00 | ORB-short ORB[478.00,482.95] vol=1.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 478.28 | 478.32 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:35:00 | 389.30 | 387.23 | 0.00 | ORB-long ORB[384.50,389.00] vol=2.8x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 10:45:00 | 391.84 | 387.48 | 0.00 | T1 1.5R @ 391.84 |
| Stop hit — per-position SL triggered | 2026-04-07 11:00:00 | 389.30 | 388.04 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:30:00 | 440.00 | 435.45 | 0.00 | ORB-long ORB[431.05,436.80] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-04-10 10:35:00 | 437.79 | 435.66 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 446.15 | 444.38 | 0.00 | ORB-long ORB[440.60,445.80] vol=2.5x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:45:00 | 448.12 | 444.90 | 0.00 | T1 1.5R @ 448.12 |
| Stop hit — per-position SL triggered | 2026-04-22 10:00:00 | 446.15 | 445.24 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:55:00 | 440.90 | 444.58 | 0.00 | ORB-short ORB[444.90,448.50] vol=1.8x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-04-23 10:00:00 | 442.26 | 444.38 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 11:15:00 | 422.40 | 424.72 | 0.00 | ORB-short ORB[423.55,428.10] vol=2.1x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:20:00 | 420.81 | 424.45 | 0.00 | T1 1.5R @ 420.81 |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 422.40 | 424.07 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 427.20 | 425.02 | 0.00 | ORB-long ORB[420.75,425.90] vol=1.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 425.86 | 425.37 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:55:00 | 410.00 | 412.65 | 0.00 | ORB-short ORB[412.20,416.70] vol=1.5x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:05:00 | 407.54 | 411.92 | 0.00 | T1 1.5R @ 407.54 |
| Target hit | 2026-04-30 13:25:00 | 408.80 | 408.79 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 10:50:00 | 496.75 | 2026-02-12 11:30:00 | 494.19 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-02-17 11:05:00 | 479.85 | 2026-02-17 11:15:00 | 481.33 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-19 10:35:00 | 482.80 | 2026-02-19 11:45:00 | 480.40 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-19 10:35:00 | 482.80 | 2026-02-19 15:20:00 | 473.25 | TARGET_HIT | 0.50 | 1.98% |
| SELL | retest1 | 2026-02-24 10:15:00 | 477.35 | 2026-02-24 12:30:00 | 474.45 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-02-24 10:15:00 | 477.35 | 2026-02-24 15:15:00 | 477.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 11:15:00 | 477.25 | 2026-02-25 11:30:00 | 478.28 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-07 10:35:00 | 389.30 | 2026-04-07 10:45:00 | 391.84 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-07 10:35:00 | 389.30 | 2026-04-07 11:00:00 | 389.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 10:30:00 | 440.00 | 2026-04-10 10:35:00 | 437.79 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-22 09:35:00 | 446.15 | 2026-04-22 09:45:00 | 448.12 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-22 09:35:00 | 446.15 | 2026-04-22 10:00:00 | 446.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 09:55:00 | 440.90 | 2026-04-23 10:00:00 | 442.26 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-27 11:15:00 | 422.40 | 2026-04-27 11:20:00 | 420.81 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-27 11:15:00 | 422.40 | 2026-04-27 12:15:00 | 422.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:30:00 | 427.20 | 2026-04-28 09:45:00 | 425.86 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-30 09:55:00 | 410.00 | 2026-04-30 10:05:00 | 407.54 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-30 09:55:00 | 410.00 | 2026-04-30 13:25:00 | 408.80 | TARGET_HIT | 0.50 | 0.29% |
