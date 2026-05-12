# Suzlon Energy Ltd. (SUZLON)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (659 bars)
- **Last close:** 54.97
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
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 0.87% / 0.00%
- **Sum % (uncompounded):** 2.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.87% | 2.6% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.87% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.87% | 2.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 00:00:00 | 78.05 | 51.29 | 75.06 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 00:00:00 | 84.40 | 51.89 | 76.28 | T1 booked 50% @ 84.40 |
| Stop hit — per-position SL triggered | 2024-09-19 00:00:00 | 78.05 | 53.38 | 78.60 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2024-12-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 00:00:00 | 69.56 | 60.27 | 66.17 | Stage2 pullback-breakout RSI=60 vol=2.7x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-12-20 00:00:00 | 65.71 | 60.45 | 66.19 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-10 00:00:00 | 78.05 | 2024-09-12 00:00:00 | 84.40 | PARTIAL | 0.50 | 8.14% |
| BUY | retest1 | 2024-09-10 00:00:00 | 78.05 | 2024-09-19 00:00:00 | 78.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 00:00:00 | 69.56 | 2024-12-20 00:00:00 | 65.71 | STOP_HIT | 1.00 | -5.53% |
