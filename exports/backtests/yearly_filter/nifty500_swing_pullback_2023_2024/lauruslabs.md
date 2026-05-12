# Laurus Labs Ltd. (LAURUSLABS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1243.00
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 3.23% / 5.70%
- **Sum % (uncompounded):** 12.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 2 | 2 | 3.23% | 12.9% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 2 | 2 | 3.23% | 12.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 2 | 2 | 3.23% | 12.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 00:00:00 | 414.55 | 381.57 | 388.34 | Stage2 pullback-breakout RSI=68 vol=4.3x ATR=11.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 00:00:00 | 438.18 | 382.61 | 396.59 | T1 booked 50% @ 438.18 |
| Stop hit — per-position SL triggered | 2024-01-08 00:00:00 | 418.70 | 386.17 | 414.09 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 439.85 | 393.06 | 403.88 | Stage2 pullback-breakout RSI=69 vol=4.0x ATR=13.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 00:00:00 | 467.24 | 395.23 | 418.83 | T1 booked 50% @ 467.24 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 439.85 | 396.66 | 425.05 | SL hit (bars_held=7) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-22 00:00:00 | 414.55 | 2023-12-27 00:00:00 | 438.18 | PARTIAL | 0.50 | 5.70% |
| BUY | retest1 | 2023-12-22 00:00:00 | 414.55 | 2024-01-08 00:00:00 | 418.70 | STOP_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2024-04-03 00:00:00 | 439.85 | 2024-04-09 00:00:00 | 467.24 | PARTIAL | 0.50 | 6.23% |
| BUY | retest1 | 2024-04-03 00:00:00 | 439.85 | 2024-04-15 00:00:00 | 439.85 | STOP_HIT | 0.50 | 0.00% |
