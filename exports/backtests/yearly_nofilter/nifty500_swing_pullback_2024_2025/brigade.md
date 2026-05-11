# Brigade Enterprises Ltd. (BRIGADE)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 2 / 0 / 2
- **Avg / median % per leg:** 6.39% / 8.26%
- **Sum % (uncompounded):** 25.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 6.39% | 25.6% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 6.39% | 25.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 6.39% | 25.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 00:00:00 | 1241.10 | 1065.49 | 1185.47 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=46.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 00:00:00 | 1334.98 | 1086.11 | 1259.56 | T1 booked 50% @ 1334.98 |
| Target hit | 2024-10-08 00:00:00 | 1343.65 | 1126.99 | 1345.81 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 1227.90 | 1147.55 | 1179.54 | Stage2 pullback-breakout RSI=55 vol=2.2x ATR=55.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 00:00:00 | 1339.36 | 1157.29 | 1232.66 | T1 booked 50% @ 1339.36 |
| Target hit | 2024-12-11 00:00:00 | 1235.95 | 1160.47 | 1240.68 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-02 00:00:00 | 1241.10 | 2024-09-13 00:00:00 | 1334.98 | PARTIAL | 0.50 | 7.56% |
| BUY | retest1 | 2024-09-02 00:00:00 | 1241.10 | 2024-10-08 00:00:00 | 1343.65 | TARGET_HIT | 0.50 | 8.26% |
| BUY | retest1 | 2024-11-25 00:00:00 | 1227.90 | 2024-12-06 00:00:00 | 1339.36 | PARTIAL | 0.50 | 9.08% |
| BUY | retest1 | 2024-11-25 00:00:00 | 1227.90 | 2024-12-11 00:00:00 | 1235.95 | TARGET_HIT | 0.50 | 0.66% |
