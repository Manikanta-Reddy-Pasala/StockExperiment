# Berger Paints India Ltd. (BERGEPAINT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 515.00
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
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 9
- **Target hits / Stop hits / Partials:** 6 / 9 / 9
- **Avg / median % per leg:** 0.40% / 0.35%
- **Sum % (uncompounded):** 9.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 3 | 3 | 3 | 0.60% | 5.4% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 3 | 3 | 3 | 0.60% | 5.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 9 | 60.0% | 3 | 6 | 6 | 0.27% | 4.1% |
| SELL @ 2nd Alert (retest1) | 15 | 9 | 60.0% | 3 | 6 | 6 | 0.27% | 4.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 15 | 62.5% | 6 | 9 | 9 | 0.40% | 9.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 452.90 | 456.24 | 0.00 | ORB-short ORB[454.25,459.95] vol=2.8x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:25:00 | 450.94 | 455.51 | 0.00 | T1 1.5R @ 450.94 |
| Stop hit — per-position SL triggered | 2026-02-11 12:05:00 | 452.90 | 455.08 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 464.35 | 462.82 | 0.00 | ORB-long ORB[458.00,464.15] vol=1.7x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-02-16 11:40:00 | 462.75 | 463.08 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 464.90 | 463.87 | 0.00 | ORB-long ORB[462.50,464.50] vol=2.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-02-17 09:40:00 | 463.86 | 463.91 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 462.10 | 462.89 | 0.00 | ORB-short ORB[462.25,465.00] vol=2.2x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:20:00 | 460.72 | 462.28 | 0.00 | T1 1.5R @ 460.72 |
| Target hit | 2026-02-18 13:05:00 | 460.80 | 460.68 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 459.15 | 460.27 | 0.00 | ORB-short ORB[459.45,462.60] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-02-19 10:00:00 | 460.25 | 459.98 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 461.95 | 462.62 | 0.00 | ORB-short ORB[462.05,464.55] vol=1.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-02-25 11:00:00 | 462.82 | 462.61 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:00:00 | 438.00 | 432.11 | 0.00 | ORB-long ORB[429.80,434.75] vol=2.1x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 12:05:00 | 439.86 | 433.18 | 0.00 | T1 1.5R @ 439.86 |
| Target hit | 2026-03-06 15:20:00 | 439.40 | 435.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 436.45 | 437.75 | 0.00 | ORB-short ORB[437.95,444.10] vol=12.4x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:10:00 | 433.66 | 437.51 | 0.00 | T1 1.5R @ 433.66 |
| Target hit | 2026-03-11 15:20:00 | 431.10 | 436.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:50:00 | 454.40 | 447.20 | 0.00 | ORB-long ORB[439.70,446.50] vol=1.5x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 12:25:00 | 457.36 | 448.94 | 0.00 | T1 1.5R @ 457.36 |
| Target hit | 2026-04-13 15:20:00 | 462.95 | 453.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 475.40 | 476.67 | 0.00 | ORB-short ORB[477.00,481.95] vol=2.8x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:30:00 | 473.63 | 476.47 | 0.00 | T1 1.5R @ 473.63 |
| Target hit | 2026-04-16 15:20:00 | 472.10 | 473.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:05:00 | 472.00 | 474.45 | 0.00 | ORB-short ORB[473.05,479.45] vol=2.2x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-04-22 11:55:00 | 473.32 | 473.44 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 475.40 | 473.53 | 0.00 | ORB-long ORB[470.50,475.20] vol=1.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-04-23 09:35:00 | 474.08 | 473.77 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 461.50 | 463.73 | 0.00 | ORB-short ORB[462.95,469.00] vol=1.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:40:00 | 459.15 | 461.78 | 0.00 | T1 1.5R @ 459.15 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 461.50 | 461.51 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 459.00 | 460.81 | 0.00 | ORB-short ORB[459.95,462.85] vol=2.0x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:30:00 | 457.41 | 459.68 | 0.00 | T1 1.5R @ 457.41 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 459.00 | 459.27 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:20:00 | 461.70 | 458.70 | 0.00 | ORB-long ORB[456.80,460.55] vol=2.4x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:30:00 | 464.36 | 459.37 | 0.00 | T1 1.5R @ 464.36 |
| Target hit | 2026-04-30 15:20:00 | 472.80 | 468.31 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:10:00 | 452.90 | 2026-02-11 11:25:00 | 450.94 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-11 11:10:00 | 452.90 | 2026-02-11 12:05:00 | 452.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:15:00 | 464.35 | 2026-02-16 11:40:00 | 462.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-17 09:35:00 | 464.90 | 2026-02-17 09:40:00 | 463.86 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-18 09:45:00 | 462.10 | 2026-02-18 10:20:00 | 460.72 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-18 09:45:00 | 462.10 | 2026-02-18 13:05:00 | 460.80 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-19 09:30:00 | 459.15 | 2026-02-19 10:00:00 | 460.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-25 10:50:00 | 461.95 | 2026-02-25 11:00:00 | 462.82 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-03-06 11:00:00 | 438.00 | 2026-03-06 12:05:00 | 439.86 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-06 11:00:00 | 438.00 | 2026-03-06 15:20:00 | 439.40 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2026-03-11 10:10:00 | 436.45 | 2026-03-11 11:10:00 | 433.66 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-03-11 10:10:00 | 436.45 | 2026-03-11 15:20:00 | 431.10 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2026-04-13 10:50:00 | 454.40 | 2026-04-13 12:25:00 | 457.36 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-13 10:50:00 | 454.40 | 2026-04-13 15:20:00 | 462.95 | TARGET_HIT | 0.50 | 1.88% |
| SELL | retest1 | 2026-04-16 11:15:00 | 475.40 | 2026-04-16 11:30:00 | 473.63 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-16 11:15:00 | 475.40 | 2026-04-16 15:20:00 | 472.10 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2026-04-22 10:05:00 | 472.00 | 2026-04-22 11:55:00 | 473.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-23 09:30:00 | 475.40 | 2026-04-23 09:35:00 | 474.08 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-28 09:30:00 | 461.50 | 2026-04-28 09:40:00 | 459.15 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-28 09:30:00 | 461.50 | 2026-04-28 09:50:00 | 461.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:55:00 | 459.00 | 2026-04-29 10:30:00 | 457.41 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-29 09:55:00 | 459.00 | 2026-04-29 11:15:00 | 459.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-30 10:20:00 | 461.70 | 2026-04-30 10:30:00 | 464.36 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-30 10:20:00 | 461.70 | 2026-04-30 15:20:00 | 472.80 | TARGET_HIT | 0.50 | 2.40% |
