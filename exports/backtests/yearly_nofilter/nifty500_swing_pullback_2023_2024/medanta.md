# Global Health Ltd. (MEDANTA)

## Backtest Summary

- **Window:** 2022-11-16 00:00:00 → 2026-05-11 00:00:00 (863 bars)
- **Last close:** 1194.70
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
- **Winners / losers:** 3 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 4.29% / 4.87%
- **Sum % (uncompounded):** 12.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 1 | 1 | 1 | 4.29% | 12.9% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 1 | 1 | 4.29% | 12.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 1 | 1 | 1 | 4.29% | 12.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 00:00:00 | 715.10 | 593.46 | 696.43 | Stage2 pullback-breakout RSI=60 vol=3.4x ATR=17.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-03 00:00:00 | 750.76 | 596.20 | 702.96 | T1 booked 50% @ 750.76 |
| Target hit | 2023-10-23 00:00:00 | 749.95 | 617.95 | 749.97 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 1340.15 | 951.21 | 1257.62 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=63.92 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 1380.30 | 994.50 | 1335.58 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-28 00:00:00 | 715.10 | 2023-10-03 00:00:00 | 750.76 | PARTIAL | 0.50 | 4.99% |
| BUY | retest1 | 2023-09-28 00:00:00 | 715.10 | 2023-10-23 00:00:00 | 749.95 | TARGET_HIT | 0.50 | 4.87% |
| BUY | retest1 | 2024-03-27 00:00:00 | 1340.15 | 2024-04-15 00:00:00 | 1380.30 | STOP_HIT | 1.00 | 3.00% |
