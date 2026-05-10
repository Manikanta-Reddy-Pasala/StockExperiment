# Sai Life Sciences Ltd. (SAILIFE)

## Backtest Summary

- **Window:** 2024-12-18 05:30:00 → 2026-05-08 05:30:00 (343 bars)
- **Last close:** 1115.60
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 3
- **Avg / median % per leg:** 5.52% / 6.65%
- **Sum % (uncompounded):** 27.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 1 | 3 | 5.52% | 27.6% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 1 | 3 | 5.52% | 27.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 1 | 3 | 5.52% | 27.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 917.60 | 829.51 | 887.77 | Stage2 pullback-breakout RSI=60 vol=2.2x ATR=24.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 05:30:00 | 967.55 | 838.52 | 910.42 | T1 booked 50% @ 967.55 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 917.60 | 841.80 | 920.22 | SL hit (bars_held=14) |

### Cycle 2 — BUY (started 2026-02-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 05:30:00 | 893.65 | 842.57 | 854.42 | Stage2 pullback-breakout RSI=56 vol=2.1x ATR=37.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 05:30:00 | 969.06 | 851.41 | 907.28 | T1 booked 50% @ 969.06 |
| Target hit | 2026-03-13 05:30:00 | 953.05 | 869.03 | 974.40 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-04-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 05:30:00 | 1041.60 | 894.43 | 983.81 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=36.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 05:30:00 | 1115.00 | 911.31 | 1042.90 | T1 booked 50% @ 1115.00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-19 05:30:00 | 917.60 | 2026-01-06 05:30:00 | 967.55 | PARTIAL | 0.50 | 5.44% |
| BUY | retest1 | 2025-12-19 05:30:00 | 917.60 | 2026-01-09 05:30:00 | 917.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-09 05:30:00 | 893.65 | 2026-02-23 05:30:00 | 969.06 | PARTIAL | 0.50 | 8.44% |
| BUY | retest1 | 2026-02-09 05:30:00 | 893.65 | 2026-03-13 05:30:00 | 953.05 | TARGET_HIT | 0.50 | 6.65% |
| BUY | retest1 | 2026-04-23 05:30:00 | 1041.60 | 2026-05-08 05:30:00 | 1115.00 | PARTIAL | 0.50 | 7.05% |
