# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1373.60
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
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 0
- **Target hits / Stop hits / Partials:** 1 / 0 / 2
- **Avg / median % per leg:** 7.05% / 6.73%
- **Sum % (uncompounded):** 21.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 1 | 0 | 2 | 7.05% | 21.1% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 1 | 0 | 2 | 7.05% | 21.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 1 | 0 | 2 | 7.05% | 21.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 00:00:00 | 1251.09 | 1139.33 | 1208.28 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=33.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 00:00:00 | 1317.39 | 1149.03 | 1247.69 | T1 booked 50% @ 1317.39 |
| Target hit | 2024-10-21 00:00:00 | 1365.05 | 1217.63 | 1423.12 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 00:00:00 | 1511.56 | 1221.95 | 1426.11 | Stage2 pullback-breakout RSI=63 vol=5.8x ATR=50.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 00:00:00 | 1613.35 | 1261.75 | 1519.61 | T1 booked 50% @ 1613.35 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-28 00:00:00 | 1251.09 | 2024-09-06 00:00:00 | 1317.39 | PARTIAL | 0.50 | 5.30% |
| BUY | retest1 | 2024-08-28 00:00:00 | 1251.09 | 2024-10-21 00:00:00 | 1365.05 | TARGET_HIT | 0.50 | 9.11% |
| BUY | retest1 | 2024-10-23 00:00:00 | 1511.56 | 2024-11-11 00:00:00 | 1613.35 | PARTIAL | 0.50 | 6.73% |
