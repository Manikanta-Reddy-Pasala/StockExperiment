# ICICI Prudential Life Insurance Company Ltd. (ICICIPRULI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 565.25
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
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 5
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 1.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.06% | 0.7% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.06% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.37% | 1.1% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.37% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.12% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 653.90 | 651.95 | 0.00 | ORB-long ORB[647.70,652.50] vol=2.4x ATR=1.80 |
| Stop hit — per-position SL triggered | 2026-02-09 11:20:00 | 652.10 | 652.13 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 641.40 | 639.29 | 0.00 | ORB-long ORB[636.70,641.20] vol=1.7x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 639.84 | 639.49 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:05:00 | 657.05 | 653.28 | 0.00 | ORB-long ORB[642.85,648.55] vol=1.5x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:25:00 | 660.41 | 654.55 | 0.00 | T1 1.5R @ 660.41 |
| Stop hit — per-position SL triggered | 2026-02-20 11:20:00 | 657.05 | 657.79 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 662.45 | 659.72 | 0.00 | ORB-long ORB[652.00,660.35] vol=2.2x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:10:00 | 664.87 | 660.32 | 0.00 | T1 1.5R @ 664.87 |
| Stop hit — per-position SL triggered | 2026-02-23 11:25:00 | 662.45 | 661.49 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 618.30 | 620.74 | 0.00 | ORB-short ORB[620.75,624.40] vol=2.5x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-03-06 11:50:00 | 619.88 | 619.84 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 595.10 | 590.50 | 0.00 | ORB-long ORB[581.30,587.85] vol=1.9x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:05:00 | 597.82 | 592.52 | 0.00 | T1 1.5R @ 597.82 |
| Stop hit — per-position SL triggered | 2026-03-17 11:45:00 | 595.10 | 593.67 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:45:00 | 513.25 | 509.14 | 0.00 | ORB-long ORB[504.00,511.15] vol=2.3x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 511.39 | 509.54 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 545.35 | 549.00 | 0.00 | ORB-short ORB[547.05,554.40] vol=1.6x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:55:00 | 542.71 | 547.67 | 0.00 | T1 1.5R @ 542.71 |
| Target hit | 2026-04-22 15:20:00 | 540.60 | 543.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-05-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:00:00 | 526.70 | 520.72 | 0.00 | ORB-long ORB[516.65,519.75] vol=1.9x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 524.88 | 521.81 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:20:00 | 559.00 | 554.95 | 0.00 | ORB-long ORB[550.35,556.35] vol=1.5x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:35:00 | 562.14 | 556.16 | 0.00 | T1 1.5R @ 562.14 |
| Stop hit — per-position SL triggered | 2026-05-07 11:10:00 | 559.00 | 557.14 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:05:00 | 653.90 | 2026-02-09 11:20:00 | 652.10 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-17 09:45:00 | 641.40 | 2026-02-17 09:50:00 | 639.84 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-20 10:05:00 | 657.05 | 2026-02-20 10:25:00 | 660.41 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-20 10:05:00 | 657.05 | 2026-02-20 11:20:00 | 657.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 11:00:00 | 662.45 | 2026-02-23 11:10:00 | 664.87 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-23 11:00:00 | 662.45 | 2026-02-23 11:25:00 | 662.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 618.30 | 2026-03-06 11:50:00 | 619.88 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-17 10:25:00 | 595.10 | 2026-03-17 11:05:00 | 597.82 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-17 10:25:00 | 595.10 | 2026-03-17 11:45:00 | 595.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 10:45:00 | 513.25 | 2026-04-07 10:50:00 | 511.39 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-22 09:40:00 | 545.35 | 2026-04-22 09:55:00 | 542.71 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-22 09:40:00 | 545.35 | 2026-04-22 15:20:00 | 540.60 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2026-05-04 10:00:00 | 526.70 | 2026-05-04 10:15:00 | 524.88 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-07 10:20:00 | 559.00 | 2026-05-07 10:35:00 | 562.14 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-05-07 10:20:00 | 559.00 | 2026-05-07 11:10:00 | 559.00 | STOP_HIT | 0.50 | 0.00% |
