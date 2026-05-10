# Tata Investment Corporation Ltd. (TATAINVEST)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 719.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 5
- **Target hits / Stop hits / Partials:** 2 / 5 / 5
- **Avg / median % per leg:** 0.32% / 0.55%
- **Sum % (uncompounded):** 3.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.10% | 0.8% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 0 | 5 | 3 | 0.10% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 0.76% | 3.0% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 0.76% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.32% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 634.95 | 629.68 | 0.00 | ORB-long ORB[622.65,631.15] vol=3.0x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 632.16 | 630.75 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 634.20 | 630.86 | 0.00 | ORB-long ORB[628.00,633.00] vol=1.7x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:05:00 | 637.16 | 632.49 | 0.00 | T1 1.5R @ 637.16 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 634.20 | 633.19 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:05:00 | 646.20 | 652.48 | 0.00 | ORB-short ORB[650.05,658.00] vol=3.6x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 15:05:00 | 642.41 | 649.95 | 0.00 | T1 1.5R @ 642.41 |
| Target hit | 2026-03-06 15:20:00 | 640.55 | 649.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 713.80 | 709.83 | 0.00 | ORB-long ORB[705.45,713.25] vol=1.8x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 717.96 | 711.34 | 0.00 | T1 1.5R @ 717.96 |
| Stop hit — per-position SL triggered | 2026-04-27 10:00:00 | 713.80 | 711.74 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:50:00 | 724.50 | 717.82 | 0.00 | ORB-long ORB[711.05,717.65] vol=9.0x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:55:00 | 729.09 | 720.04 | 0.00 | T1 1.5R @ 729.09 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 724.50 | 722.52 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 727.75 | 719.31 | 0.00 | ORB-long ORB[710.05,716.70] vol=7.4x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 724.45 | 721.28 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 724.55 | 728.60 | 0.00 | ORB-short ORB[728.40,735.25] vol=2.2x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:10:00 | 720.55 | 726.49 | 0.00 | T1 1.5R @ 720.55 |
| Target hit | 2026-05-08 15:20:00 | 717.15 | 722.59 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 09:30:00 | 634.95 | 2026-02-16 09:40:00 | 632.16 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-17 09:40:00 | 634.20 | 2026-02-17 10:05:00 | 637.16 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-17 09:40:00 | 634.20 | 2026-02-17 10:40:00 | 634.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 11:05:00 | 646.20 | 2026-03-06 15:05:00 | 642.41 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-06 11:05:00 | 646.20 | 2026-03-06 15:20:00 | 640.55 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2026-04-27 09:40:00 | 713.80 | 2026-04-27 09:50:00 | 717.96 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-27 09:40:00 | 713.80 | 2026-04-27 10:00:00 | 713.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 10:50:00 | 724.50 | 2026-04-28 10:55:00 | 729.09 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-28 10:50:00 | 724.50 | 2026-04-28 11:00:00 | 724.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:40:00 | 727.75 | 2026-05-05 09:45:00 | 724.45 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-05-08 09:45:00 | 724.55 | 2026-05-08 10:10:00 | 720.55 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-08 09:45:00 | 724.55 | 2026-05-08 15:20:00 | 717.15 | TARGET_HIT | 0.50 | 1.02% |
