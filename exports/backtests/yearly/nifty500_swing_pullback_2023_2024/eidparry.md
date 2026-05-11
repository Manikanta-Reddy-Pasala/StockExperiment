# E.I.D. Parry (India) Ltd. (EIDPARRY)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 834.30
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
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.72% / 0.00%
- **Sum % (uncompounded):** -2.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.72% | -2.9% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.72% | -2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.72% | -2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 05:30:00 | 591.70 | 523.01 | 571.42 | Stage2 pullback-breakout RSI=64 vol=2.8x ATR=16.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 05:30:00 | 624.10 | 523.94 | 575.74 | T1 booked 50% @ 624.10 |
| Stop hit — per-position SL triggered | 2024-02-07 05:30:00 | 591.70 | 532.26 | 606.42 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2024-02-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 05:30:00 | 637.95 | 537.71 | 601.57 | Stage2 pullback-breakout RSI=64 vol=3.9x ATR=23.23 |
| Stop hit — per-position SL triggered | 2024-03-04 05:30:00 | 617.15 | 546.34 | 617.62 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-04-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 05:30:00 | 627.70 | 553.03 | 585.85 | Stage2 pullback-breakout RSI=66 vol=4.6x ATR=21.36 |
| Stop hit — per-position SL triggered | 2024-04-19 05:30:00 | 595.66 | 555.77 | 594.87 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-24 05:30:00 | 591.70 | 2024-01-25 05:30:00 | 624.10 | PARTIAL | 0.50 | 5.48% |
| BUY | retest1 | 2024-01-24 05:30:00 | 591.70 | 2024-02-07 05:30:00 | 591.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-20 05:30:00 | 637.95 | 2024-03-04 05:30:00 | 617.15 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest1 | 2024-04-10 05:30:00 | 627.70 | 2024-04-19 05:30:00 | 595.66 | STOP_HIT | 1.00 | -5.10% |
