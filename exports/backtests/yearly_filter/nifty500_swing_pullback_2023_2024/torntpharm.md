# Torrent Pharmaceuticals Ltd. (TORNTPHARM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 4409.30
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 7.33% / 4.42%
- **Sum % (uncompounded):** 21.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 7.33% | 22.0% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 7.33% | 22.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 7.33% | 22.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 00:00:00 | 2144.20 | 1869.29 | 2084.20 | Stage2 pullback-breakout RSI=62 vol=1.9x ATR=47.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 00:00:00 | 2238.91 | 1885.62 | 2133.58 | T1 booked 50% @ 2238.91 |
| Target hit | 2024-03-15 00:00:00 | 2594.45 | 2176.58 | 2644.74 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-05-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 00:00:00 | 2734.70 | 2287.25 | 2635.52 | Stage2 pullback-breakout RSI=64 vol=3.7x ATR=62.50 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 2640.95 | 2294.79 | 2641.02 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-18 00:00:00 | 2144.20 | 2023-12-26 00:00:00 | 2238.91 | PARTIAL | 0.50 | 4.42% |
| BUY | retest1 | 2023-12-18 00:00:00 | 2144.20 | 2024-03-15 00:00:00 | 2594.45 | TARGET_HIT | 0.50 | 21.00% |
| BUY | retest1 | 2024-05-03 00:00:00 | 2734.70 | 2024-05-07 00:00:00 | 2640.95 | STOP_HIT | 1.00 | -3.43% |
