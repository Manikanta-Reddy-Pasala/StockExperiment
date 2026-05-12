# Brigade Enterprises Ltd. (BRIGADE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 758.25
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 1.01% / -1.80%
- **Sum % (uncompounded):** 3.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.01% | 3.0% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.01% | 3.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.01% | 3.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 00:00:00 | 580.90 | 518.68 | 567.31 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=16.97 |
| Stop hit — per-position SL triggered | 2023-07-26 00:00:00 | 570.45 | 524.26 | 572.72 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 630.35 | 539.72 | 592.52 | Stage2 pullback-breakout RSI=68 vol=4.2x ATR=19.46 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 601.15 | 544.87 | 608.24 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-03-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 00:00:00 | 932.00 | 780.85 | 917.37 | Stage2 pullback-breakout RSI=51 vol=2.7x ATR=44.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 00:00:00 | 1020.24 | 806.79 | 964.51 | T1 booked 50% @ 1020.24 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-12 00:00:00 | 580.90 | 2023-07-26 00:00:00 | 570.45 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest1 | 2023-09-04 00:00:00 | 630.35 | 2023-09-12 00:00:00 | 601.15 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest1 | 2024-03-22 00:00:00 | 932.00 | 2024-04-18 00:00:00 | 1020.24 | PARTIAL | 0.50 | 9.47% |
