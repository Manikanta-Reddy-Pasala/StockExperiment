# Hindustan Zinc Ltd. (HINDZINC)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 635.10
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 2.68% / 4.63%
- **Sum % (uncompounded):** 16.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.68% | 16.1% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.68% | 16.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.68% | 16.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 05:30:00 | 495.20 | 460.65 | 482.51 | Stage2 pullback-breakout RSI=58 vol=2.8x ATR=12.33 |
| Stop hit — per-position SL triggered | 2025-11-18 05:30:00 | 476.71 | 461.29 | 482.36 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2025-11-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 05:30:00 | 485.20 | 461.90 | 475.61 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=11.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 05:30:00 | 507.67 | 463.03 | 481.97 | T1 booked 50% @ 507.67 |
| Stop hit — per-position SL triggered | 2025-12-09 05:30:00 | 485.20 | 464.25 | 485.90 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2025-12-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 05:30:00 | 512.65 | 464.73 | 488.45 | Stage2 pullback-breakout RSI=63 vol=3.9x ATR=13.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 05:30:00 | 538.72 | 466.26 | 498.37 | T1 booked 50% @ 538.72 |
| Target hit | 2026-01-08 05:30:00 | 590.75 | 489.91 | 596.16 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-04-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 05:30:00 | 628.00 | 539.55 | 576.23 | Stage2 pullback-breakout RSI=67 vol=3.2x ATR=21.59 |
| Stop hit — per-position SL triggered | 2026-04-30 05:30:00 | 595.61 | 541.58 | 584.39 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-13 05:30:00 | 495.20 | 2025-11-18 05:30:00 | 476.71 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest1 | 2025-11-28 05:30:00 | 485.20 | 2025-12-03 05:30:00 | 507.67 | PARTIAL | 0.50 | 4.63% |
| BUY | retest1 | 2025-11-28 05:30:00 | 485.20 | 2025-12-09 05:30:00 | 485.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 05:30:00 | 512.65 | 2025-12-12 05:30:00 | 538.72 | PARTIAL | 0.50 | 5.09% |
| BUY | retest1 | 2025-12-10 05:30:00 | 512.65 | 2026-01-08 05:30:00 | 590.75 | TARGET_HIT | 0.50 | 15.23% |
| BUY | retest1 | 2026-04-27 05:30:00 | 628.00 | 2026-04-30 05:30:00 | 595.61 | STOP_HIT | 1.00 | -5.16% |
