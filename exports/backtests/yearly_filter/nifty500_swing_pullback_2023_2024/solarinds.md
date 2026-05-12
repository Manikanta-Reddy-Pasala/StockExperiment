# Solar Industries India Ltd. (SOLARINDS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 15995.00
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
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 4.96% / 5.87%
- **Sum % (uncompounded):** 19.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.96% | 19.8% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.96% | 19.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.96% | 19.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-03-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 00:00:00 | 7288.40 | 5654.66 | 6830.40 | Stage2 pullback-breakout RSI=66 vol=4.6x ATR=213.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 00:00:00 | 7716.05 | 5675.61 | 6918.95 | T1 booked 50% @ 7716.05 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 7288.40 | 5690.96 | 6947.46 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-03-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 00:00:00 | 8144.85 | 5788.05 | 7243.14 | Stage2 pullback-breakout RSI=67 vol=2.0x ATR=410.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 00:00:00 | 8965.71 | 5818.57 | 7396.70 | T1 booked 50% @ 8965.71 |
| Target hit | 2024-04-16 00:00:00 | 8462.70 | 6324.17 | 8497.89 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-03-04 00:00:00 | 7288.40 | 2024-03-05 00:00:00 | 7716.05 | PARTIAL | 0.50 | 5.87% |
| BUY | retest1 | 2024-03-04 00:00:00 | 7288.40 | 2024-03-06 00:00:00 | 7288.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-14 00:00:00 | 8144.85 | 2024-03-15 00:00:00 | 8965.71 | PARTIAL | 0.50 | 10.08% |
| BUY | retest1 | 2024-03-14 00:00:00 | 8144.85 | 2024-04-16 00:00:00 | 8462.70 | TARGET_HIT | 0.50 | 3.90% |
