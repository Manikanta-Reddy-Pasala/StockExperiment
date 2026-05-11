# EIH Ltd. (EIHOTEL)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 331.85
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
- **Avg / median % per leg:** 1.02% / 1.94%
- **Sum % (uncompounded):** 3.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 1.02% | 3.1% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 1.02% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 1.02% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 435.55 | 381.34 | 421.80 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=14.65 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 413.58 | 383.62 | 423.71 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-10-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 00:00:00 | 391.15 | 382.29 | 380.07 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=12.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 00:00:00 | 415.27 | 383.20 | 389.01 | T1 booked 50% @ 415.27 |
| Target hit | 2024-10-18 00:00:00 | 398.75 | 385.29 | 401.79 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-29 00:00:00 | 435.55 | 2024-08-05 00:00:00 | 413.58 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest1 | 2024-10-04 00:00:00 | 391.15 | 2024-10-10 00:00:00 | 415.27 | PARTIAL | 0.50 | 6.17% |
| BUY | retest1 | 2024-10-04 00:00:00 | 391.15 | 2024-10-18 00:00:00 | 398.75 | TARGET_HIT | 0.50 | 1.94% |
