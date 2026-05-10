# Gujarat Mineral Development Corporation Ltd. (GMDCLTD)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 685.00
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
- **Avg / median % per leg:** 0.59% / 0.55%
- **Sum % (uncompounded):** 7.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.65% | 6.5% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.65% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.28% | 0.6% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.28% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.59% | 7.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 623.60 | 615.81 | 0.00 | ORB-long ORB[603.00,612.00] vol=2.0x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:10:00 | 629.67 | 620.96 | 0.00 | T1 1.5R @ 629.67 |
| Stop hit — per-position SL triggered | 2026-02-09 13:05:00 | 623.60 | 628.40 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 636.50 | 631.63 | 0.00 | ORB-long ORB[625.50,634.75] vol=2.5x ATR=3.49 |
| Stop hit — per-position SL triggered | 2026-02-10 09:35:00 | 633.01 | 632.09 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:55:00 | 574.30 | 566.17 | 0.00 | ORB-long ORB[556.00,564.55] vol=1.6x ATR=3.37 |
| Stop hit — per-position SL triggered | 2026-02-20 10:05:00 | 570.93 | 567.62 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 569.00 | 566.35 | 0.00 | ORB-long ORB[562.70,568.60] vol=2.1x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:45:00 | 572.10 | 568.07 | 0.00 | T1 1.5R @ 572.10 |
| Stop hit — per-position SL triggered | 2026-02-26 11:40:00 | 569.00 | 570.38 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:15:00 | 570.25 | 578.01 | 0.00 | ORB-short ORB[576.80,582.70] vol=3.0x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:35:00 | 567.01 | 577.07 | 0.00 | T1 1.5R @ 567.01 |
| Stop hit — per-position SL triggered | 2026-02-27 11:55:00 | 570.25 | 576.37 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:45:00 | 599.50 | 594.95 | 0.00 | ORB-long ORB[590.25,597.00] vol=1.8x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:55:00 | 602.46 | 595.65 | 0.00 | T1 1.5R @ 602.46 |
| Target hit | 2026-04-13 15:20:00 | 614.70 | 612.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:35:00 | 683.80 | 681.52 | 0.00 | ORB-long ORB[673.00,682.50] vol=4.4x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:15:00 | 689.95 | 683.43 | 0.00 | T1 1.5R @ 689.95 |
| Target hit | 2026-04-27 15:20:00 | 698.90 | 691.17 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:00:00 | 623.60 | 2026-02-09 11:10:00 | 629.67 | PARTIAL | 0.50 | 0.97% |
| BUY | retest1 | 2026-02-09 11:00:00 | 623.60 | 2026-02-09 13:05:00 | 623.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 09:30:00 | 636.50 | 2026-02-10 09:35:00 | 633.01 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-02-20 09:55:00 | 574.30 | 2026-02-20 10:05:00 | 570.93 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2026-02-26 09:35:00 | 569.00 | 2026-02-26 09:45:00 | 572.10 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-26 09:35:00 | 569.00 | 2026-02-26 11:40:00 | 569.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 11:15:00 | 570.25 | 2026-02-27 11:35:00 | 567.01 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-27 11:15:00 | 570.25 | 2026-02-27 11:55:00 | 570.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:45:00 | 599.50 | 2026-04-13 10:55:00 | 602.46 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-13 10:45:00 | 599.50 | 2026-04-13 15:20:00 | 614.70 | TARGET_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2026-04-27 09:35:00 | 683.80 | 2026-04-27 10:15:00 | 689.95 | PARTIAL | 0.50 | 0.90% |
| BUY | retest1 | 2026-04-27 09:35:00 | 683.80 | 2026-04-27 15:20:00 | 698.90 | TARGET_HIT | 0.50 | 2.21% |
