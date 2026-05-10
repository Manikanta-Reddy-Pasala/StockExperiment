# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 882.20
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 6
- **Target hits / Stop hits / Partials:** 5 / 6 / 6
- **Avg / median % per leg:** 0.45% / 0.36%
- **Sum % (uncompounded):** 7.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.46% | 3.6% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.46% | 3.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 6 | 66.7% | 3 | 3 | 3 | 0.45% | 4.0% |
| SELL @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 3 | 3 | 3 | 0.45% | 4.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 11 | 64.7% | 5 | 6 | 6 | 0.45% | 7.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 774.10 | 767.02 | 0.00 | ORB-long ORB[759.95,767.90] vol=2.0x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:45:00 | 778.07 | 768.19 | 0.00 | T1 1.5R @ 778.07 |
| Stop hit — per-position SL triggered | 2026-02-20 11:00:00 | 774.10 | 769.32 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 751.85 | 754.00 | 0.00 | ORB-short ORB[752.50,763.00] vol=3.1x ATR=4.00 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 755.85 | 754.06 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:35:00 | 724.85 | 733.23 | 0.00 | ORB-short ORB[733.10,740.60] vol=2.0x ATR=2.51 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 727.36 | 731.47 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 687.00 | 690.68 | 0.00 | ORB-short ORB[691.00,697.25] vol=2.2x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:10:00 | 683.08 | 688.18 | 0.00 | T1 1.5R @ 683.08 |
| Target hit | 2026-03-13 15:20:00 | 678.70 | 683.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:05:00 | 642.35 | 647.58 | 0.00 | ORB-short ORB[646.60,654.80] vol=1.5x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 637.91 | 645.48 | 0.00 | T1 1.5R @ 637.91 |
| Target hit | 2026-03-23 13:50:00 | 632.35 | 632.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-04-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 11:00:00 | 667.40 | 659.69 | 0.00 | ORB-long ORB[656.05,665.15] vol=2.0x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 11:35:00 | 671.53 | 662.13 | 0.00 | T1 1.5R @ 671.53 |
| Target hit | 2026-04-02 15:20:00 | 684.65 | 671.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:55:00 | 779.30 | 773.00 | 0.00 | ORB-long ORB[769.00,778.15] vol=1.6x ATR=2.41 |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 776.89 | 773.69 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 818.60 | 815.50 | 0.00 | ORB-long ORB[809.35,815.60] vol=1.9x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:45:00 | 821.43 | 816.25 | 0.00 | T1 1.5R @ 821.43 |
| Target hit | 2026-04-21 15:00:00 | 820.75 | 823.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:10:00 | 815.95 | 820.30 | 0.00 | ORB-short ORB[816.35,827.35] vol=1.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-04-22 11:50:00 | 818.08 | 819.82 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:05:00 | 799.15 | 803.89 | 0.00 | ORB-short ORB[802.20,809.40] vol=2.1x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:20:00 | 796.24 | 802.42 | 0.00 | T1 1.5R @ 796.24 |
| Target hit | 2026-04-23 15:20:00 | 792.80 | 794.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-05-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:10:00 | 901.90 | 893.74 | 0.00 | ORB-long ORB[884.50,894.45] vol=2.1x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-05-08 11:25:00 | 898.60 | 894.50 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-20 10:35:00 | 774.10 | 2026-02-20 10:45:00 | 778.07 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-20 10:35:00 | 774.10 | 2026-02-20 11:00:00 | 774.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 751.85 | 2026-02-24 09:45:00 | 755.85 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-02-26 10:35:00 | 724.85 | 2026-02-26 10:55:00 | 727.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-13 09:50:00 | 687.00 | 2026-03-13 12:10:00 | 683.08 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-13 09:50:00 | 687.00 | 2026-03-13 15:20:00 | 678.70 | TARGET_HIT | 0.50 | 1.21% |
| SELL | retest1 | 2026-03-23 10:05:00 | 642.35 | 2026-03-23 10:15:00 | 637.91 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-23 10:05:00 | 642.35 | 2026-03-23 13:50:00 | 632.35 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2026-04-02 11:00:00 | 667.40 | 2026-04-02 11:35:00 | 671.53 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-02 11:00:00 | 667.40 | 2026-04-02 15:20:00 | 684.65 | TARGET_HIT | 0.50 | 2.58% |
| BUY | retest1 | 2026-04-10 10:55:00 | 779.30 | 2026-04-10 11:15:00 | 776.89 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-21 11:00:00 | 818.60 | 2026-04-21 11:45:00 | 821.43 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-21 11:00:00 | 818.60 | 2026-04-21 15:00:00 | 820.75 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-22 11:10:00 | 815.95 | 2026-04-22 11:50:00 | 818.08 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-23 11:05:00 | 799.15 | 2026-04-23 11:20:00 | 796.24 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-23 11:05:00 | 799.15 | 2026-04-23 15:20:00 | 792.80 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2026-05-08 11:10:00 | 901.90 | 2026-05-08 11:25:00 | 898.60 | STOP_HIT | 1.00 | -0.37% |
