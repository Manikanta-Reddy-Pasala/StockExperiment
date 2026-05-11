# InterGlobe Aviation Ltd. (INDIGO)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 4342.00
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
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 2.64% / 4.65%
- **Sum % (uncompounded):** 10.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.64% | 10.6% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.64% | 10.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 2.64% | 10.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 00:00:00 | 4385.85 | 3494.23 | 4287.86 | Stage2 pullback-breakout RSI=59 vol=1.5x ATR=97.43 |
| Stop hit — per-position SL triggered | 2024-07-23 00:00:00 | 4239.70 | 3536.24 | 4312.44 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 00:00:00 | 4483.15 | 3686.98 | 4314.73 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=104.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 00:00:00 | 4691.68 | 3697.16 | 4352.42 | T1 booked 50% @ 4691.68 |
| Target hit | 2024-09-25 00:00:00 | 4782.20 | 3934.59 | 4817.45 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 00:00:00 | 4352.65 | 4068.81 | 4165.51 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=119.34 |
| Stop hit — per-position SL triggered | 2024-12-12 00:00:00 | 4464.35 | 4103.26 | 4338.81 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-15 00:00:00 | 4385.85 | 2024-07-23 00:00:00 | 4239.70 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest1 | 2024-08-22 00:00:00 | 4483.15 | 2024-08-23 00:00:00 | 4691.68 | PARTIAL | 0.50 | 4.65% |
| BUY | retest1 | 2024-08-22 00:00:00 | 4483.15 | 2024-09-25 00:00:00 | 4782.20 | TARGET_HIT | 0.50 | 6.67% |
| BUY | retest1 | 2024-11-28 00:00:00 | 4352.65 | 2024-12-12 00:00:00 | 4464.35 | STOP_HIT | 1.00 | 2.57% |
