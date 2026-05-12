# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-08 00:00:00 (663 bars)
- **Last close:** 1198.40
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
- **Avg / median % per leg:** 2.65% / 4.39%
- **Sum % (uncompounded):** 7.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 2.65% | 8.0% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 2.65% | 8.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 2.65% | 8.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 00:00:00 | 1626.15 | 1462.05 | 1585.69 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=35.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 00:00:00 | 1697.52 | 1476.87 | 1634.59 | T1 booked 50% @ 1697.52 |
| Target hit | 2024-09-19 00:00:00 | 1736.50 | 1525.71 | 1753.89 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 00:00:00 | 1870.10 | 1568.62 | 1801.07 | Stage2 pullback-breakout RSI=70 vol=1.9x ATR=40.15 |
| Stop hit — per-position SL triggered | 2024-10-23 00:00:00 | 1809.87 | 1585.02 | 1822.47 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-14 00:00:00 | 1626.15 | 2024-08-26 00:00:00 | 1697.52 | PARTIAL | 0.50 | 4.39% |
| BUY | retest1 | 2024-08-14 00:00:00 | 1626.15 | 2024-09-19 00:00:00 | 1736.50 | TARGET_HIT | 0.50 | 6.79% |
| BUY | retest1 | 2024-10-15 00:00:00 | 1870.10 | 2024-10-23 00:00:00 | 1809.87 | STOP_HIT | 1.00 | -3.22% |
