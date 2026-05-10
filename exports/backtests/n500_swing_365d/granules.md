# Granules India Ltd. (GRANULES)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 752.95
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
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 2
- **Avg / median % per leg:** 1.77% / 4.92%
- **Sum % (uncompounded):** 7.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 1.77% | 7.1% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 1.77% | 7.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 2 | 2 | 1.77% | 7.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 583.40 | 536.88 | 566.68 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=14.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 05:30:00 | 612.10 | 538.28 | 574.20 | T1 booked 50% @ 612.10 |
| Stop hit — per-position SL triggered | 2026-01-12 05:30:00 | 583.40 | 546.71 | 597.68 | SL hit (bars_held=15) |

### Cycle 2 — BUY (started 2026-02-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 05:30:00 | 599.05 | 552.94 | 575.83 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=18.26 |
| Stop hit — per-position SL triggered | 2026-03-02 05:30:00 | 571.65 | 555.15 | 582.00 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2026-03-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 05:30:00 | 606.35 | 558.35 | 581.53 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=20.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 05:30:00 | 647.12 | 562.33 | 602.09 | T1 booked 50% @ 647.12 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-19 05:30:00 | 583.40 | 2025-12-23 05:30:00 | 612.10 | PARTIAL | 0.50 | 4.92% |
| BUY | retest1 | 2025-12-19 05:30:00 | 583.40 | 2026-01-12 05:30:00 | 583.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 05:30:00 | 599.05 | 2026-03-02 05:30:00 | 571.65 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest1 | 2026-03-24 05:30:00 | 606.35 | 2026-04-06 05:30:00 | 647.12 | PARTIAL | 0.50 | 6.72% |
