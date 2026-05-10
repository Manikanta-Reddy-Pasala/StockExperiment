# Brigade Enterprises Ltd. (BRIGADE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 760.25
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 7
- **Avg / median % per leg:** 0.37% / 0.37%
- **Sum % (uncompounded):** 5.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 1 | 3 | 4 | 0.42% | 3.4% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 1 | 3 | 4 | 0.42% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.31% | 2.2% |
| SELL @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.31% | 2.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 10 | 66.7% | 3 | 5 | 7 | 0.37% | 5.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 735.70 | 739.20 | 0.00 | ORB-short ORB[738.10,744.40] vol=2.2x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 12:45:00 | 732.96 | 737.61 | 0.00 | T1 1.5R @ 732.96 |
| Stop hit — per-position SL triggered | 2026-02-18 13:20:00 | 735.70 | 736.89 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 659.55 | 661.91 | 0.00 | ORB-short ORB[662.70,671.20] vol=2.4x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:50:00 | 653.58 | 660.50 | 0.00 | T1 1.5R @ 653.58 |
| Target hit | 2026-03-04 14:50:00 | 657.45 | 657.01 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 719.30 | 711.68 | 0.00 | ORB-long ORB[704.60,711.70] vol=1.9x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:45:00 | 724.38 | 715.30 | 0.00 | T1 1.5R @ 724.38 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 719.30 | 716.43 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 754.85 | 752.28 | 0.00 | ORB-long ORB[738.25,746.40] vol=2.3x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:10:00 | 759.64 | 752.87 | 0.00 | T1 1.5R @ 759.64 |
| Target hit | 2026-04-17 15:20:00 | 762.15 | 758.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 788.00 | 783.28 | 0.00 | ORB-long ORB[776.95,784.90] vol=1.5x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:35:00 | 791.80 | 786.43 | 0.00 | T1 1.5R @ 791.80 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 788.00 | 787.11 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:05:00 | 802.15 | 796.69 | 0.00 | ORB-long ORB[788.35,797.40] vol=4.1x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:10:00 | 806.78 | 798.24 | 0.00 | T1 1.5R @ 806.78 |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 802.15 | 799.27 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 794.85 | 798.03 | 0.00 | ORB-short ORB[795.00,805.55] vol=2.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 14:20:00 | 790.11 | 795.00 | 0.00 | T1 1.5R @ 790.11 |
| Target hit | 2026-05-04 15:20:00 | 791.95 | 794.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-05-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:30:00 | 779.05 | 782.85 | 0.00 | ORB-short ORB[785.10,795.00] vol=5.3x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-05-05 10:40:00 | 781.80 | 782.77 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 11:00:00 | 735.70 | 2026-02-18 12:45:00 | 732.96 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-18 11:00:00 | 735.70 | 2026-02-18 13:20:00 | 735.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:45:00 | 659.55 | 2026-03-04 10:50:00 | 653.58 | PARTIAL | 0.50 | 0.90% |
| SELL | retest1 | 2026-03-04 09:45:00 | 659.55 | 2026-03-04 14:50:00 | 657.45 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-04-10 09:35:00 | 719.30 | 2026-04-10 09:45:00 | 724.38 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-04-10 09:35:00 | 719.30 | 2026-04-10 10:00:00 | 719.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 11:00:00 | 754.85 | 2026-04-17 11:10:00 | 759.64 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-17 11:00:00 | 754.85 | 2026-04-17 15:20:00 | 762.15 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2026-04-21 09:30:00 | 788.00 | 2026-04-21 09:35:00 | 791.80 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-21 09:30:00 | 788.00 | 2026-04-21 09:50:00 | 788.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 10:05:00 | 802.15 | 2026-04-28 10:10:00 | 806.78 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-28 10:05:00 | 802.15 | 2026-04-28 10:15:00 | 802.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 10:30:00 | 794.85 | 2026-05-04 14:20:00 | 790.11 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-05-04 10:30:00 | 794.85 | 2026-05-04 15:20:00 | 791.95 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2026-05-05 10:30:00 | 779.05 | 2026-05-05 10:40:00 | 781.80 | STOP_HIT | 1.00 | -0.35% |
