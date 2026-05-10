# Hindustan Zinc Ltd. (HINDZINC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 634.75
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 1
- **Avg / median % per leg:** -0.21% / -0.31%
- **Sum % (uncompounded):** -2.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.04% | -0.2% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.04% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.35% | -2.1% |
| SELL @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.35% | -2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 2 | 18.2% | 1 | 9 | 1 | -0.21% | -2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 584.65 | 586.24 | 0.00 | ORB-short ORB[585.10,590.00] vol=1.6x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 586.44 | 586.05 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 615.25 | 609.70 | 0.00 | ORB-long ORB[602.85,609.80] vol=1.7x ATR=2.10 |
| Stop hit — per-position SL triggered | 2026-02-25 10:50:00 | 613.15 | 610.29 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 590.75 | 593.27 | 0.00 | ORB-short ORB[592.30,596.80] vol=1.7x ATR=1.85 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 592.60 | 592.81 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:15:00 | 514.55 | 519.12 | 0.00 | ORB-short ORB[517.20,524.95] vol=7.3x ATR=2.39 |
| Stop hit — per-position SL triggered | 2026-03-19 13:30:00 | 516.94 | 517.33 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:40:00 | 486.25 | 490.82 | 0.00 | ORB-short ORB[492.00,499.00] vol=1.5x ATR=2.16 |
| Stop hit — per-position SL triggered | 2026-03-24 11:20:00 | 488.41 | 489.76 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 11:00:00 | 560.35 | 555.34 | 0.00 | ORB-long ORB[550.00,558.35] vol=1.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 14:30:00 | 563.05 | 557.68 | 0.00 | T1 1.5R @ 563.05 |
| Target hit | 2026-04-13 15:20:00 | 561.80 | 559.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:30:00 | 596.00 | 593.61 | 0.00 | ORB-long ORB[590.30,595.85] vol=1.8x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 594.46 | 594.00 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 589.50 | 595.14 | 0.00 | ORB-short ORB[593.15,598.50] vol=1.6x ATR=1.78 |
| Stop hit — per-position SL triggered | 2026-04-24 11:05:00 | 591.28 | 594.80 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 610.55 | 605.25 | 0.00 | ORB-long ORB[597.80,606.75] vol=1.5x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 608.34 | 605.41 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 618.25 | 620.40 | 0.00 | ORB-short ORB[618.65,625.55] vol=1.6x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-05-06 11:35:00 | 619.75 | 620.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-17 09:35:00 | 584.65 | 2026-02-17 09:50:00 | 586.44 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-25 10:25:00 | 615.25 | 2026-02-25 10:50:00 | 613.15 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-06 10:45:00 | 590.75 | 2026-03-06 11:35:00 | 592.60 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-19 11:15:00 | 514.55 | 2026-03-19 13:30:00 | 516.94 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-24 10:40:00 | 486.25 | 2026-03-24 11:20:00 | 488.41 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-13 11:00:00 | 560.35 | 2026-04-13 14:30:00 | 563.05 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-13 11:00:00 | 560.35 | 2026-04-13 15:20:00 | 561.80 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-04-21 10:30:00 | 596.00 | 2026-04-21 11:00:00 | 594.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-24 10:55:00 | 589.50 | 2026-04-24 11:05:00 | 591.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-04 09:45:00 | 610.55 | 2026-05-04 09:50:00 | 608.34 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-06 11:05:00 | 618.25 | 2026-05-06 11:35:00 | 619.75 | STOP_HIT | 1.00 | -0.24% |
