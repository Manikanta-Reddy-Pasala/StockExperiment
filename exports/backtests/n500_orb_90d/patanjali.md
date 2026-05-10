# Patanjali Foods Ltd. (PATANJALI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 459.90
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 4
- **Avg / median % per leg:** 0.07% / -0.23%
- **Sum % (uncompounded):** 1.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.35% | -1.7% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.35% | -1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 6 | 40.0% | 2 | 9 | 4 | 0.21% | 3.1% |
| SELL @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 2 | 9 | 4 | 0.21% | 3.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 6 | 30.0% | 2 | 14 | 4 | 0.07% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 532.00 | 527.37 | 0.00 | ORB-long ORB[521.85,525.40] vol=2.1x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-02-17 09:55:00 | 530.25 | 529.60 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 528.95 | 531.17 | 0.00 | ORB-short ORB[530.75,535.95] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-02-23 11:30:00 | 530.10 | 530.90 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 521.20 | 524.15 | 0.00 | ORB-short ORB[522.50,525.15] vol=1.5x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-02-25 11:00:00 | 522.59 | 523.96 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 499.55 | 496.96 | 0.00 | ORB-long ORB[493.10,497.75] vol=2.2x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-03-04 09:35:00 | 496.85 | 497.60 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 499.25 | 502.54 | 0.00 | ORB-short ORB[503.05,508.85] vol=1.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-03-05 12:55:00 | 500.67 | 500.53 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:05:00 | 496.45 | 500.42 | 0.00 | ORB-short ORB[500.50,504.50] vol=1.8x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:50:00 | 494.65 | 499.94 | 0.00 | T1 1.5R @ 494.65 |
| Stop hit — per-position SL triggered | 2026-03-11 12:30:00 | 496.45 | 499.47 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 491.05 | 489.25 | 0.00 | ORB-long ORB[484.15,490.90] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-03-18 11:05:00 | 489.94 | 489.34 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:50:00 | 482.00 | 479.49 | 0.00 | ORB-long ORB[476.30,481.60] vol=2.4x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-03-20 11:00:00 | 480.68 | 479.65 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 467.45 | 470.08 | 0.00 | ORB-short ORB[468.25,473.00] vol=3.2x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:10:00 | 465.06 | 469.95 | 0.00 | T1 1.5R @ 465.06 |
| Target hit | 2026-03-30 15:20:00 | 456.25 | 460.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-04-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:45:00 | 461.20 | 461.89 | 0.00 | ORB-short ORB[463.60,470.05] vol=1.8x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-04-09 11:10:00 | 462.73 | 461.89 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 461.85 | 464.94 | 0.00 | ORB-short ORB[464.05,470.35] vol=1.5x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-04-16 10:05:00 | 463.12 | 464.57 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 471.05 | 469.70 | 0.00 | ORB-long ORB[463.55,469.25] vol=7.2x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-04-17 10:35:00 | 469.35 | 470.24 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:30:00 | 461.00 | 463.89 | 0.00 | ORB-short ORB[463.05,468.95] vol=1.8x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 09:35:00 | 458.61 | 463.12 | 0.00 | T1 1.5R @ 458.61 |
| Stop hit — per-position SL triggered | 2026-04-20 09:50:00 | 461.00 | 462.39 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 459.75 | 460.92 | 0.00 | ORB-short ORB[460.00,463.80] vol=16.2x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-04-21 09:55:00 | 461.24 | 460.91 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:55:00 | 461.15 | 462.91 | 0.00 | ORB-short ORB[461.45,464.45] vol=4.4x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 12:10:00 | 459.52 | 462.33 | 0.00 | T1 1.5R @ 459.52 |
| Target hit | 2026-05-04 15:20:00 | 456.80 | 458.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-05-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:10:00 | 456.00 | 457.60 | 0.00 | ORB-short ORB[457.90,462.20] vol=3.2x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-05-06 10:30:00 | 457.37 | 457.39 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:45:00 | 532.00 | 2026-02-17 09:55:00 | 530.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-23 11:15:00 | 528.95 | 2026-02-23 11:30:00 | 530.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-25 10:50:00 | 521.20 | 2026-02-25 11:00:00 | 522.59 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-04 09:30:00 | 499.55 | 2026-03-04 09:35:00 | 496.85 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-03-05 11:15:00 | 499.25 | 2026-03-05 12:55:00 | 500.67 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-11 11:05:00 | 496.45 | 2026-03-11 11:50:00 | 494.65 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-03-11 11:05:00 | 496.45 | 2026-03-11 12:30:00 | 496.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 11:00:00 | 491.05 | 2026-03-18 11:05:00 | 489.94 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-03-20 10:50:00 | 482.00 | 2026-03-20 11:00:00 | 480.68 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-30 10:55:00 | 467.45 | 2026-03-30 11:10:00 | 465.06 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-03-30 10:55:00 | 467.45 | 2026-03-30 15:20:00 | 456.25 | TARGET_HIT | 0.50 | 2.40% |
| SELL | retest1 | 2026-04-09 10:45:00 | 461.20 | 2026-04-09 11:10:00 | 462.73 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-16 09:45:00 | 461.85 | 2026-04-16 10:05:00 | 463.12 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-17 10:05:00 | 471.05 | 2026-04-17 10:35:00 | 469.35 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-20 09:30:00 | 461.00 | 2026-04-20 09:35:00 | 458.61 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-04-20 09:30:00 | 461.00 | 2026-04-20 09:50:00 | 461.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 09:50:00 | 459.75 | 2026-04-21 09:55:00 | 461.24 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-04 10:55:00 | 461.15 | 2026-05-04 12:10:00 | 459.52 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-05-04 10:55:00 | 461.15 | 2026-05-04 15:20:00 | 456.80 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2026-05-06 10:10:00 | 456.00 | 2026-05-06 10:30:00 | 457.37 | STOP_HIT | 1.00 | -0.30% |
