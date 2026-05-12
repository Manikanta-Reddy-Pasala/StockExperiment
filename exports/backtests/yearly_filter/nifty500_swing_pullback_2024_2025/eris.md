# Eris Lifesciences Ltd. (ERIS)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 1373.30
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
- **Avg / median % per leg:** 10.61% / 7.53%
- **Sum % (uncompounded):** 42.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 10.61% | 42.5% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 10.61% | 42.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 10.61% | 42.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 1099.75 | 919.82 | 1039.16 | Stage2 pullback-breakout RSI=70 vol=3.3x ATR=41.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 00:00:00 | 1182.52 | 937.94 | 1095.91 | T1 booked 50% @ 1182.52 |
| Target hit | 2024-09-20 00:00:00 | 1366.70 | 1041.48 | 1374.88 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-11-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 00:00:00 | 1369.45 | 1127.31 | 1322.54 | Stage2 pullback-breakout RSI=60 vol=3.5x ATR=46.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 00:00:00 | 1463.12 | 1144.04 | 1352.15 | T1 booked 50% @ 1463.12 |
| Target hit | 2024-12-16 00:00:00 | 1421.75 | 1189.42 | 1445.61 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-29 00:00:00 | 1099.75 | 2024-08-09 00:00:00 | 1182.52 | PARTIAL | 0.50 | 7.53% |
| BUY | retest1 | 2024-07-29 00:00:00 | 1099.75 | 2024-09-20 00:00:00 | 1366.70 | TARGET_HIT | 0.50 | 24.27% |
| BUY | retest1 | 2024-11-12 00:00:00 | 1369.45 | 2024-11-25 00:00:00 | 1463.12 | PARTIAL | 0.50 | 6.84% |
| BUY | retest1 | 2024-11-12 00:00:00 | 1369.45 | 2024-12-16 00:00:00 | 1421.75 | TARGET_HIT | 0.50 | 3.82% |
