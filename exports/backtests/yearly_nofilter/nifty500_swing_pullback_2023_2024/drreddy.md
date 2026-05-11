# Dr. Reddy's Laboratories Ltd. (DRREDDY)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1293.90
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 0.70% / 4.03%
- **Sum % (uncompounded):** 2.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.70% | 2.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.70% | 2.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.70% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 00:00:00 | 1171.71 | 1053.57 | 1129.34 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=23.11 |
| Stop hit — per-position SL triggered | 2024-01-11 00:00:00 | 1137.04 | 1063.95 | 1148.82 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-01-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 00:00:00 | 1180.42 | 1070.71 | 1145.62 | Stage2 pullback-breakout RSI=62 vol=2.7x ATR=23.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 00:00:00 | 1227.97 | 1076.35 | 1162.46 | T1 booked 50% @ 1227.97 |
| Target hit | 2024-03-01 00:00:00 | 1238.71 | 1111.14 | 1255.10 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 1243.75 | 1146.84 | 1218.56 | Stage2 pullback-breakout RSI=56 vol=2.6x ATR=26.50 |
| Stop hit — per-position SL triggered | 2024-05-08 00:00:00 | 1204.00 | 1154.75 | 1234.87 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-12-28 00:00:00 | 1171.71 | 2024-01-11 00:00:00 | 1137.04 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest1 | 2024-01-24 00:00:00 | 1180.42 | 2024-02-01 00:00:00 | 1227.97 | PARTIAL | 0.50 | 4.03% |
| BUY | retest1 | 2024-01-24 00:00:00 | 1180.42 | 2024-03-01 00:00:00 | 1238.71 | TARGET_HIT | 0.50 | 4.94% |
| BUY | retest1 | 2024-04-25 00:00:00 | 1243.75 | 2024-05-08 00:00:00 | 1204.00 | STOP_HIT | 1.00 | -3.20% |
