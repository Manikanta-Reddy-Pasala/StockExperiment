# Intellect Design Arena Ltd. (INTELLECT)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 806.80
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
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 5.28% / 7.66%
- **Sum % (uncompounded):** 21.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.28% | 21.1% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.28% | 21.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.28% | 21.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 05:30:00 | 1015.65 | 958.44 | 968.26 | Stage2 pullback-breakout RSI=56 vol=2.1x ATR=40.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 05:30:00 | 1095.93 | 964.56 | 1008.75 | T1 booked 50% @ 1095.93 |
| Stop hit — per-position SL triggered | 2025-09-24 05:30:00 | 1029.05 | 966.17 | 1015.32 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 1044.75 | 969.54 | 988.46 | Stage2 pullback-breakout RSI=60 vol=2.7x ATR=40.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 05:30:00 | 1124.79 | 973.03 | 1012.96 | T1 booked 50% @ 1124.79 |
| Target hit | 2025-11-25 05:30:00 | 1088.80 | 994.59 | 1090.41 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-10 05:30:00 | 1015.65 | 2025-09-22 05:30:00 | 1095.93 | PARTIAL | 0.50 | 7.90% |
| BUY | retest1 | 2025-09-10 05:30:00 | 1015.65 | 2025-09-24 05:30:00 | 1029.05 | STOP_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-10-27 05:30:00 | 1044.75 | 2025-10-31 05:30:00 | 1124.79 | PARTIAL | 0.50 | 7.66% |
| BUY | retest1 | 2025-10-27 05:30:00 | 1044.75 | 2025-11-25 05:30:00 | 1088.80 | TARGET_HIT | 0.50 | 4.22% |
