# Welspun Living Ltd. (WELSPUNLIV)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (654 bars)
- **Last close:** 133.80
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
- **Avg / median % per leg:** 5.97% / 6.89%
- **Sum % (uncompounded):** 23.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 5.97% | 23.9% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 5.97% | 23.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 5.97% | 23.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 00:00:00 | 160.62 | 143.55 | 149.45 | Stage2 pullback-breakout RSI=68 vol=5.5x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 00:00:00 | 170.94 | 144.07 | 153.20 | T1 booked 50% @ 170.94 |
| Target hit | 2024-08-05 00:00:00 | 174.30 | 149.20 | 174.84 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-12-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 00:00:00 | 160.08 | 158.67 | 153.75 | Stage2 pullback-breakout RSI=57 vol=2.2x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 00:00:00 | 171.12 | 158.98 | 158.14 | T1 booked 50% @ 171.12 |
| Target hit | 2024-12-20 00:00:00 | 163.34 | 160.09 | 165.97 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-09 00:00:00 | 160.62 | 2024-07-11 00:00:00 | 170.94 | PARTIAL | 0.50 | 6.42% |
| BUY | retest1 | 2024-07-09 00:00:00 | 160.62 | 2024-08-05 00:00:00 | 174.30 | TARGET_HIT | 0.50 | 8.52% |
| BUY | retest1 | 2024-12-03 00:00:00 | 160.08 | 2024-12-09 00:00:00 | 171.12 | PARTIAL | 0.50 | 6.89% |
| BUY | retest1 | 2024-12-03 00:00:00 | 160.08 | 2024-12-20 00:00:00 | 163.34 | TARGET_HIT | 0.50 | 2.04% |
