# BEL (BEL)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 439.70
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
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 1
- **Avg / median % per leg:** 2.76% / 3.92%
- **Sum % (uncompounded):** 8.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 2.76% | 8.3% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 2.76% | 8.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 2.76% | 8.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 00:00:00 | 288.55 | 258.75 | 280.67 | Stage2 pullback-breakout RSI=56 vol=2.1x ATR=9.28 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 281.55 | 261.88 | 287.71 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 292.35 | 263.00 | 284.98 | Stage2 pullback-breakout RSI=56 vol=2.1x ATR=9.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 00:00:00 | 312.19 | 265.53 | 294.89 | T1 booked 50% @ 312.19 |
| Target hit | 2024-12-18 00:00:00 | 303.80 | 270.47 | 306.65 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-30 00:00:00 | 288.55 | 2024-11-13 00:00:00 | 281.55 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest1 | 2024-11-25 00:00:00 | 292.35 | 2024-12-03 00:00:00 | 312.19 | PARTIAL | 0.50 | 6.79% |
| BUY | retest1 | 2024-11-25 00:00:00 | 292.35 | 2024-12-18 00:00:00 | 303.80 | TARGET_HIT | 0.50 | 3.92% |
