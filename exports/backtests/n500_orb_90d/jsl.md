# Jindal Stainless Ltd. (JSL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 753.00
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 2
- **Avg / median % per leg:** 0.19% / -0.19%
- **Sum % (uncompounded):** 2.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.50% | 3.5% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.50% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.24% | -1.2% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.24% | -1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 3 | 25.0% | 1 | 9 | 2 | 0.19% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 757.35 | 749.26 | 0.00 | ORB-long ORB[738.60,748.70] vol=1.8x ATR=2.48 |
| Stop hit — per-position SL triggered | 2026-02-20 10:55:00 | 754.87 | 750.75 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 768.60 | 765.56 | 0.00 | ORB-long ORB[757.65,765.95] vol=1.7x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:30:00 | 771.62 | 765.92 | 0.00 | T1 1.5R @ 771.62 |
| Target hit | 2026-02-23 15:20:00 | 797.00 | 787.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 793.65 | 797.82 | 0.00 | ORB-short ORB[797.25,807.00] vol=2.0x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 795.15 | 797.48 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 757.70 | 763.47 | 0.00 | ORB-short ORB[762.20,767.95] vol=2.2x ATR=2.16 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 759.86 | 762.51 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:40:00 | 729.25 | 722.13 | 0.00 | ORB-long ORB[713.35,723.15] vol=3.2x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-04-06 09:45:00 | 725.94 | 722.66 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 800.35 | 795.61 | 0.00 | ORB-long ORB[788.65,793.00] vol=2.3x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-04-21 09:45:00 | 797.50 | 796.31 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 783.00 | 777.20 | 0.00 | ORB-long ORB[771.05,779.65] vol=1.8x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:30:00 | 787.23 | 779.80 | 0.00 | T1 1.5R @ 787.23 |
| Stop hit — per-position SL triggered | 2026-04-28 10:40:00 | 783.00 | 780.27 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 761.25 | 769.46 | 0.00 | ORB-short ORB[776.00,787.40] vol=1.9x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-05-06 11:25:00 | 763.69 | 769.04 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 765.35 | 768.71 | 0.00 | ORB-short ORB[767.15,773.75] vol=2.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-05-07 11:10:00 | 767.17 | 768.59 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 763.60 | 765.66 | 0.00 | ORB-short ORB[766.40,773.80] vol=2.4x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-05-08 11:40:00 | 765.08 | 765.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-20 10:40:00 | 757.35 | 2026-02-20 10:55:00 | 754.87 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-23 11:15:00 | 768.60 | 2026-02-23 11:30:00 | 771.62 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-23 11:15:00 | 768.60 | 2026-02-23 15:20:00 | 797.00 | TARGET_HIT | 0.50 | 3.70% |
| SELL | retest1 | 2026-02-26 11:00:00 | 793.65 | 2026-02-26 11:25:00 | 795.15 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-03-06 10:45:00 | 757.70 | 2026-03-06 11:35:00 | 759.86 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-06 09:40:00 | 729.25 | 2026-04-06 09:45:00 | 725.94 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-21 09:35:00 | 800.35 | 2026-04-21 09:45:00 | 797.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-28 10:10:00 | 783.00 | 2026-04-28 10:30:00 | 787.23 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-28 10:10:00 | 783.00 | 2026-04-28 10:40:00 | 783.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:10:00 | 761.25 | 2026-05-06 11:25:00 | 763.69 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-07 11:05:00 | 765.35 | 2026-05-07 11:10:00 | 767.17 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-08 11:00:00 | 763.60 | 2026-05-08 11:40:00 | 765.08 | STOP_HIT | 1.00 | -0.19% |
