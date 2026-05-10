# One 97 Communications Ltd. (PAYTM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1187.10
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 7.23% / 5.32%
- **Sum % (uncompounded):** 36.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.23% | 36.2% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.23% | 36.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.23% | 36.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 05:30:00 | 940.55 | 818.32 | 913.13 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=25.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 05:30:00 | 990.60 | 822.85 | 928.57 | T1 booked 50% @ 990.60 |
| Target hit | 2025-09-19 05:30:00 | 1177.20 | 949.75 | 1214.66 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 05:30:00 | 1224.20 | 969.65 | 1180.19 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=35.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 05:30:00 | 1295.58 | 991.14 | 1219.90 | T1 booked 50% @ 1295.58 |
| Target hit | 2025-11-04 05:30:00 | 1268.00 | 1025.59 | 1271.85 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-11-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 05:30:00 | 1346.50 | 1031.69 | 1283.16 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=33.50 |
| Stop hit — per-position SL triggered | 2025-11-14 05:30:00 | 1296.25 | 1045.51 | 1295.10 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-10 05:30:00 | 940.55 | 2025-07-15 05:30:00 | 990.60 | PARTIAL | 0.50 | 5.32% |
| BUY | retest1 | 2025-07-10 05:30:00 | 940.55 | 2025-09-19 05:30:00 | 1177.20 | TARGET_HIT | 0.50 | 25.16% |
| BUY | retest1 | 2025-10-06 05:30:00 | 1224.20 | 2025-10-16 05:30:00 | 1295.58 | PARTIAL | 0.50 | 5.83% |
| BUY | retest1 | 2025-10-06 05:30:00 | 1224.20 | 2025-11-04 05:30:00 | 1268.00 | TARGET_HIT | 0.50 | 3.58% |
| BUY | retest1 | 2025-11-07 05:30:00 | 1346.50 | 2025-11-14 05:30:00 | 1296.25 | STOP_HIT | 1.00 | -3.73% |
