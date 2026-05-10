# Varun Beverages Ltd. (VBL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 508.35
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
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 6
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 4.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 5 | 33.3% | 1 | 10 | 4 | 0.16% | 2.4% |
| BUY @ 2nd Alert (retest1) | 15 | 5 | 33.3% | 1 | 10 | 4 | 0.16% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.39% | 2.0% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.39% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 8 | 40.0% | 2 | 12 | 6 | 0.22% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 444.90 | 442.25 | 0.00 | ORB-long ORB[439.10,443.45] vol=1.6x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:35:00 | 447.88 | 443.91 | 0.00 | T1 1.5R @ 447.88 |
| Target hit | 2026-02-09 15:20:00 | 457.05 | 452.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:20:00 | 460.35 | 458.35 | 0.00 | ORB-long ORB[454.70,460.30] vol=3.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 458.82 | 458.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 451.70 | 454.27 | 0.00 | ORB-short ORB[455.00,458.20] vol=3.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-12 11:00:00 | 452.61 | 453.73 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 450.05 | 452.20 | 0.00 | ORB-short ORB[450.70,453.75] vol=1.5x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:40:00 | 448.11 | 451.42 | 0.00 | T1 1.5R @ 448.11 |
| Stop hit — per-position SL triggered | 2026-02-13 09:45:00 | 450.05 | 451.34 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 457.50 | 456.64 | 0.00 | ORB-long ORB[454.35,456.80] vol=2.0x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 12:50:00 | 458.94 | 457.46 | 0.00 | T1 1.5R @ 458.94 |
| Stop hit — per-position SL triggered | 2026-02-17 13:00:00 | 457.50 | 457.54 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 458.60 | 460.66 | 0.00 | ORB-short ORB[460.75,464.00] vol=2.9x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:05:00 | 457.09 | 459.86 | 0.00 | T1 1.5R @ 457.09 |
| Target hit | 2026-02-19 15:20:00 | 452.20 | 457.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-02-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:20:00 | 460.65 | 458.05 | 0.00 | ORB-long ORB[455.75,458.80] vol=1.9x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-26 10:30:00 | 459.53 | 458.32 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 438.35 | 436.20 | 0.00 | ORB-long ORB[432.60,437.60] vol=1.5x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:20:00 | 440.60 | 437.40 | 0.00 | T1 1.5R @ 440.60 |
| Stop hit — per-position SL triggered | 2026-03-05 11:40:00 | 438.35 | 437.58 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:35:00 | 439.35 | 438.01 | 0.00 | ORB-long ORB[434.30,438.80] vol=9.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:00:00 | 441.58 | 438.62 | 0.00 | T1 1.5R @ 441.58 |
| Stop hit — per-position SL triggered | 2026-03-11 10:05:00 | 439.35 | 438.71 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 409.20 | 407.89 | 0.00 | ORB-long ORB[405.80,409.10] vol=1.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-03-20 09:35:00 | 408.02 | 407.90 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 397.30 | 395.80 | 0.00 | ORB-long ORB[391.10,397.00] vol=1.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-03-25 09:35:00 | 395.74 | 395.92 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:10:00 | 455.70 | 451.15 | 0.00 | ORB-long ORB[446.55,453.00] vol=1.9x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 454.31 | 451.72 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:00:00 | 492.90 | 490.13 | 0.00 | ORB-long ORB[484.35,491.45] vol=2.0x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-04-22 10:10:00 | 490.86 | 490.26 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 527.35 | 523.60 | 0.00 | ORB-long ORB[520.05,524.80] vol=3.2x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 525.49 | 524.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 444.90 | 2026-02-09 11:35:00 | 447.88 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-02-09 10:35:00 | 444.90 | 2026-02-09 15:20:00 | 457.05 | TARGET_HIT | 0.50 | 2.73% |
| BUY | retest1 | 2026-02-10 10:20:00 | 460.35 | 2026-02-10 10:40:00 | 458.82 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-12 10:45:00 | 451.70 | 2026-02-12 11:00:00 | 452.61 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-13 09:30:00 | 450.05 | 2026-02-13 09:40:00 | 448.11 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-13 09:30:00 | 450.05 | 2026-02-13 09:45:00 | 450.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:00:00 | 457.50 | 2026-02-17 12:50:00 | 458.94 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-17 11:00:00 | 457.50 | 2026-02-17 13:00:00 | 457.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 458.60 | 2026-02-19 12:05:00 | 457.09 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-19 11:15:00 | 458.60 | 2026-02-19 15:20:00 | 452.20 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2026-02-26 10:20:00 | 460.65 | 2026-02-26 10:30:00 | 459.53 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-05 11:05:00 | 438.35 | 2026-03-05 11:20:00 | 440.60 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-05 11:05:00 | 438.35 | 2026-03-05 11:40:00 | 438.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:35:00 | 439.35 | 2026-03-11 10:00:00 | 441.58 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-11 09:35:00 | 439.35 | 2026-03-11 10:05:00 | 439.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 09:30:00 | 409.20 | 2026-03-20 09:35:00 | 408.02 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-25 09:30:00 | 397.30 | 2026-03-25 09:35:00 | 395.74 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-16 11:10:00 | 455.70 | 2026-04-16 11:25:00 | 454.31 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-22 10:00:00 | 492.90 | 2026-04-22 10:10:00 | 490.86 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-29 10:20:00 | 527.35 | 2026-04-29 10:30:00 | 525.49 | STOP_HIT | 1.00 | -0.35% |
