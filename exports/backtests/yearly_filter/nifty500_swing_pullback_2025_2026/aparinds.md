# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2024-09-04 00:00:00 → 2026-05-08 00:00:00 (415 bars)
- **Last close:** 12810.00
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
| TARGET_HIT | 0 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 1
- **Target hits / Stop hits / Partials:** 0 / 1 / 1
- **Avg / median % per leg:** 1.38% / 9.85%
- **Sum % (uncompounded):** 2.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.38% | 2.8% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.38% | 2.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 1 | 1 | 1.38% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 00:00:00 | 10608.00 | 8803.27 | 9841.07 | Stage2 pullback-breakout RSI=63 vol=2.3x ATR=501.23 |
| Stop hit — per-position SL triggered | 2026-03-30 00:00:00 | 9856.16 | 8813.99 | 9844.87 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2026-04-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 00:00:00 | 10767.00 | 8892.62 | 10013.61 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=530.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 00:00:00 | 11827.11 | 9038.75 | 10655.98 | T1 booked 50% @ 11827.11 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-27 00:00:00 | 10608.00 | 2026-03-30 00:00:00 | 9856.16 | STOP_HIT | 1.00 | -7.09% |
| BUY | retest1 | 2026-04-09 00:00:00 | 10767.00 | 2026-04-20 00:00:00 | 11827.11 | PARTIAL | 0.50 | 9.85% |
