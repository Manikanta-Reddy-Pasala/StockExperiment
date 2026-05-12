# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 4465.30
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 2.63% / 1.15%
- **Sum % (uncompounded):** 13.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 2.63% | 13.2% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 2.63% | 13.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 2.63% | 13.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 00:00:00 | 4702.00 | 3164.00 | 4273.59 | Stage2 pullback-breakout RSI=69 vol=3.4x ATR=200.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 00:00:00 | 5103.89 | 3199.88 | 4401.89 | T1 booked 50% @ 5103.89 |
| Target hit | 2024-08-29 00:00:00 | 4749.95 | 3357.32 | 4755.05 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-09-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 00:00:00 | 5165.85 | 3491.87 | 4772.05 | Stage2 pullback-breakout RSI=66 vol=6.7x ATR=243.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 00:00:00 | 5653.81 | 3567.50 | 4988.48 | T1 booked 50% @ 5653.81 |
| Target hit | 2024-10-03 00:00:00 | 5225.20 | 3749.77 | 5290.85 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 00:00:00 | 5872.95 | 4112.78 | 5452.36 | Stage2 pullback-breakout RSI=63 vol=4.1x ATR=273.78 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 5462.27 | 4186.49 | 5516.39 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-13 00:00:00 | 4702.00 | 2024-08-16 00:00:00 | 5103.89 | PARTIAL | 0.50 | 8.55% |
| BUY | retest1 | 2024-08-13 00:00:00 | 4702.00 | 2024-08-29 00:00:00 | 4749.95 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2024-09-12 00:00:00 | 5165.85 | 2024-09-18 00:00:00 | 5653.81 | PARTIAL | 0.50 | 9.45% |
| BUY | retest1 | 2024-09-12 00:00:00 | 5165.85 | 2024-10-03 00:00:00 | 5225.20 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2024-11-06 00:00:00 | 5872.95 | 2024-11-13 00:00:00 | 5462.27 | STOP_HIT | 1.00 | -6.99% |
