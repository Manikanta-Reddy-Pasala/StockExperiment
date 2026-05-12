# Timken India Ltd. (TIMKEN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3518.10
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
- **Avg / median % per leg:** 0.53% / 0.00%
- **Sum % (uncompounded):** 1.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.53% | 1.6% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.53% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.53% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 3381.00 | 3182.35 | 3329.90 | Stage2 pullback-breakout RSI=55 vol=3.1x ATR=82.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 00:00:00 | 3545.44 | 3189.16 | 3365.54 | T1 booked 50% @ 3545.44 |
| Stop hit — per-position SL triggered | 2023-08-04 00:00:00 | 3381.00 | 3196.55 | 3384.03 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 3362.55 | 3138.74 | 3204.45 | Stage2 pullback-breakout RSI=69 vol=4.1x ATR=73.36 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 3252.50 | 3158.24 | 3299.64 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-28 00:00:00 | 3381.00 | 2023-08-01 00:00:00 | 3545.44 | PARTIAL | 0.50 | 4.86% |
| BUY | retest1 | 2023-07-28 00:00:00 | 3381.00 | 2023-08-04 00:00:00 | 3381.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-11 00:00:00 | 3362.55 | 2024-01-24 00:00:00 | 3252.50 | STOP_HIT | 1.00 | -3.27% |
