# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1112.20
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
- **Avg / median % per leg:** 5.79% / 6.95%
- **Sum % (uncompounded):** 23.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 5.79% | 23.2% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 5.79% | 23.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 5.79% | 23.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 00:00:00 | 597.35 | 403.74 | 541.89 | Stage2 pullback-breakout RSI=68 vol=3.4x ATR=25.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 00:00:00 | 648.30 | 422.86 | 593.07 | T1 booked 50% @ 648.30 |
| Target hit | 2023-10-23 00:00:00 | 612.70 | 437.50 | 616.22 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 00:00:00 | 607.95 | 469.25 | 565.23 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=21.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 00:00:00 | 650.19 | 475.76 | 588.76 | T1 booked 50% @ 650.19 |
| Target hit | 2024-01-04 00:00:00 | 639.05 | 502.03 | 645.63 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-28 00:00:00 | 597.35 | 2023-10-12 00:00:00 | 648.30 | PARTIAL | 0.50 | 8.53% |
| BUY | retest1 | 2023-09-28 00:00:00 | 597.35 | 2023-10-23 00:00:00 | 612.70 | TARGET_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2023-12-07 00:00:00 | 607.95 | 2023-12-13 00:00:00 | 650.19 | PARTIAL | 0.50 | 6.95% |
| BUY | retest1 | 2023-12-07 00:00:00 | 607.95 | 2024-01-04 00:00:00 | 639.05 | TARGET_HIT | 0.50 | 5.12% |
