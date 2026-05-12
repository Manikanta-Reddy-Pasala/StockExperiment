# Kalyan Jewellers India Ltd. (KALYANKJIL)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 424.55
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
- **Avg / median % per leg:** 3.70% / 9.27%
- **Sum % (uncompounded):** 14.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.70% | 14.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.70% | 14.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.70% | 14.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 00:00:00 | 596.40 | 418.71 | 551.55 | Stage2 pullback-breakout RSI=68 vol=7.6x ATR=27.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 00:00:00 | 651.66 | 434.10 | 590.38 | T1 booked 50% @ 651.66 |
| Target hit | 2024-10-07 00:00:00 | 702.20 | 489.94 | 705.41 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 700.00 | 530.85 | 687.07 | Stage2 pullback-breakout RSI=53 vol=1.8x ATR=30.92 |
| Stop hit — per-position SL triggered | 2024-11-14 00:00:00 | 653.62 | 540.00 | 686.29 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-12-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 00:00:00 | 775.45 | 564.33 | 714.05 | Stage2 pullback-breakout RSI=70 vol=1.5x ATR=28.93 |
| Stop hit — per-position SL triggered | 2024-12-20 00:00:00 | 732.06 | 580.43 | 734.79 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-22 00:00:00 | 596.40 | 2024-09-03 00:00:00 | 651.66 | PARTIAL | 0.50 | 9.27% |
| BUY | retest1 | 2024-08-22 00:00:00 | 596.40 | 2024-10-07 00:00:00 | 702.20 | TARGET_HIT | 0.50 | 17.74% |
| BUY | retest1 | 2024-11-06 00:00:00 | 700.00 | 2024-11-14 00:00:00 | 653.62 | STOP_HIT | 1.00 | -6.63% |
| BUY | retest1 | 2024-12-09 00:00:00 | 775.45 | 2024-12-20 00:00:00 | 732.06 | STOP_HIT | 1.00 | -5.60% |
