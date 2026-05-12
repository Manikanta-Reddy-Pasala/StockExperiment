# RITES Ltd. (RITES)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 227.15
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -5.65% / -6.44%
- **Sum % (uncompounded):** -16.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.65% | -16.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.65% | -16.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.65% | -16.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 00:00:00 | 373.85 | 315.61 | 349.00 | Stage2 pullback-breakout RSI=65 vol=5.6x ATR=13.72 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 362.78 | 321.08 | 363.55 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 379.58 | 322.86 | 358.29 | Stage2 pullback-breakout RSI=58 vol=5.0x ATR=19.10 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 350.92 | 324.67 | 358.53 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-09-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 00:00:00 | 373.20 | 328.08 | 342.87 | Stage2 pullback-breakout RSI=69 vol=4.0x ATR=16.02 |
| Stop hit — per-position SL triggered | 2024-09-26 00:00:00 | 349.17 | 329.36 | 348.81 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-04 00:00:00 | 373.85 | 2024-07-19 00:00:00 | 362.78 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest1 | 2024-07-29 00:00:00 | 379.58 | 2024-08-05 00:00:00 | 350.92 | STOP_HIT | 1.00 | -7.55% |
| BUY | retest1 | 2024-09-20 00:00:00 | 373.20 | 2024-09-26 00:00:00 | 349.17 | STOP_HIT | 1.00 | -6.44% |
