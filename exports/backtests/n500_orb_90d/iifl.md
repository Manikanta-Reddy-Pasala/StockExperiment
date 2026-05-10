# IIFL Finance Ltd. (IIFL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 460.10
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 6
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 3.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.20% | 2.6% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.20% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.14% | 1.1% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 2 | 4 | 2 | 0.14% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 9 | 42.9% | 3 | 12 | 6 | 0.18% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 513.70 | 510.13 | 0.00 | ORB-long ORB[506.55,511.40] vol=2.2x ATR=1.91 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 511.79 | 510.53 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 511.75 | 517.46 | 0.00 | ORB-short ORB[517.25,523.00] vol=2.2x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:15:00 | 508.37 | 515.28 | 0.00 | T1 1.5R @ 508.37 |
| Target hit | 2026-02-18 13:35:00 | 507.50 | 507.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 504.00 | 498.93 | 0.00 | ORB-long ORB[496.45,503.00] vol=1.9x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-02-24 10:20:00 | 501.66 | 500.55 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 503.25 | 506.56 | 0.00 | ORB-short ORB[505.15,511.90] vol=4.1x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-02-26 10:30:00 | 504.92 | 505.64 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 496.60 | 497.48 | 0.00 | ORB-short ORB[497.50,504.90] vol=1.7x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-02-27 15:00:00 | 498.37 | 497.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 489.25 | 484.46 | 0.00 | ORB-long ORB[479.10,486.00] vol=2.8x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 487.82 | 485.38 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 489.00 | 492.15 | 0.00 | ORB-short ORB[490.10,497.45] vol=5.2x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-03-06 09:50:00 | 491.28 | 492.02 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 479.75 | 475.83 | 0.00 | ORB-long ORB[473.50,479.00] vol=2.2x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:30:00 | 482.23 | 476.49 | 0.00 | T1 1.5R @ 482.23 |
| Target hit | 2026-03-10 15:20:00 | 492.50 | 483.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 479.15 | 476.56 | 0.00 | ORB-long ORB[472.50,477.50] vol=1.5x ATR=2.76 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 476.39 | 476.78 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:00:00 | 452.90 | 453.02 | 0.00 | ORB-short ORB[453.70,459.65] vol=2.0x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:15:00 | 449.49 | 452.43 | 0.00 | T1 1.5R @ 449.49 |
| Target hit | 2026-03-27 12:25:00 | 450.30 | 450.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2026-04-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:50:00 | 430.00 | 431.68 | 0.00 | ORB-short ORB[431.55,437.45] vol=2.4x ATR=2.27 |
| Stop hit — per-position SL triggered | 2026-04-07 10:00:00 | 432.27 | 431.75 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:55:00 | 452.00 | 445.07 | 0.00 | ORB-long ORB[442.25,448.00] vol=2.0x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-04-08 11:00:00 | 449.42 | 445.35 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:55:00 | 450.50 | 445.57 | 0.00 | ORB-long ORB[440.55,446.10] vol=2.2x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 11:20:00 | 453.19 | 446.41 | 0.00 | T1 1.5R @ 453.19 |
| Stop hit — per-position SL triggered | 2026-04-13 11:30:00 | 450.50 | 446.77 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 474.60 | 471.77 | 0.00 | ORB-long ORB[467.45,473.00] vol=1.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:35:00 | 477.69 | 472.84 | 0.00 | T1 1.5R @ 477.69 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 474.60 | 473.93 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 468.00 | 464.86 | 0.00 | ORB-long ORB[460.25,467.00] vol=2.3x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:35:00 | 470.32 | 466.28 | 0.00 | T1 1.5R @ 470.32 |
| Stop hit — per-position SL triggered | 2026-05-08 09:40:00 | 468.00 | 466.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:25:00 | 513.70 | 2026-02-17 10:40:00 | 511.79 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 10:55:00 | 511.75 | 2026-02-18 11:15:00 | 508.37 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-02-18 10:55:00 | 511.75 | 2026-02-18 13:35:00 | 507.50 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2026-02-24 09:45:00 | 504.00 | 2026-02-24 10:20:00 | 501.66 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-26 10:15:00 | 503.25 | 2026-02-26 10:30:00 | 504.92 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-27 10:45:00 | 496.60 | 2026-02-27 15:00:00 | 498.37 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-05 11:15:00 | 489.25 | 2026-03-05 11:45:00 | 487.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 09:45:00 | 489.00 | 2026-03-06 09:50:00 | 491.28 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-10 11:10:00 | 479.75 | 2026-03-10 11:30:00 | 482.23 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-10 11:10:00 | 479.75 | 2026-03-10 15:20:00 | 492.50 | TARGET_HIT | 0.50 | 2.66% |
| BUY | retest1 | 2026-03-20 09:30:00 | 479.15 | 2026-03-20 09:50:00 | 476.39 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-03-27 10:00:00 | 452.90 | 2026-03-27 10:15:00 | 449.49 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2026-03-27 10:00:00 | 452.90 | 2026-03-27 12:25:00 | 450.30 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-07 09:50:00 | 430.00 | 2026-04-07 10:00:00 | 432.27 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-08 10:55:00 | 452.00 | 2026-04-08 11:00:00 | 449.42 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-13 10:55:00 | 450.50 | 2026-04-13 11:20:00 | 453.19 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-13 10:55:00 | 450.50 | 2026-04-13 11:30:00 | 450.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:30:00 | 474.60 | 2026-04-21 09:35:00 | 477.69 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-21 09:30:00 | 474.60 | 2026-04-21 09:50:00 | 474.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 09:30:00 | 468.00 | 2026-05-08 09:35:00 | 470.32 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-08 09:30:00 | 468.00 | 2026-05-08 09:40:00 | 468.00 | STOP_HIT | 0.50 | 0.00% |
