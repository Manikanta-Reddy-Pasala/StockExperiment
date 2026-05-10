# Ather Energy Ltd. (ATHERENERG)

## Backtest Summary

- **Window:** 2025-05-06 05:30:00 → 2026-05-08 05:30:00 (251 bars)
- **Last close:** 915.05
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
- **Winners / losers:** 2 / 0
- **Target hits / Stop hits / Partials:** 0 / 1 / 1
- **Avg / median % per leg:** 5.49% / 9.53%
- **Sum % (uncompounded):** 10.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.49% | 11.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.49% | 11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.49% | 11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 05:30:00 | 750.20 | 584.80 | 706.67 | Stage2 pullback-breakout RSI=65 vol=3.0x ATR=30.28 |
| Stop hit — per-position SL triggered | 2026-04-07 05:30:00 | 761.00 | 603.88 | 748.02 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2026-04-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 05:30:00 | 820.25 | 607.68 | 756.85 | Stage2 pullback-breakout RSI=65 vol=2.9x ATR=39.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 05:30:00 | 898.45 | 613.18 | 780.38 | T1 booked 50% @ 898.45 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-18 05:30:00 | 750.20 | 2026-04-07 05:30:00 | 761.00 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest1 | 2026-04-09 05:30:00 | 820.25 | 2026-04-13 05:30:00 | 898.45 | PARTIAL | 0.50 | 9.53% |
