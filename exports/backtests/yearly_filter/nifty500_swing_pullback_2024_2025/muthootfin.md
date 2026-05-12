# Muthoot Finance Ltd. (MUTHOOTFIN)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 3528.90
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.82% / 2.22%
- **Sum % (uncompounded):** 3.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 3 | 1 | 0.82% | 3.3% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 0.82% | 3.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 0.82% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 00:00:00 | 1829.30 | 1509.50 | 1749.67 | Stage2 pullback-breakout RSI=65 vol=2.3x ATR=52.17 |
| Stop hit — per-position SL triggered | 2024-07-11 00:00:00 | 1831.90 | 1537.50 | 1785.70 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 00:00:00 | 1915.55 | 1607.09 | 1842.87 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=60.11 |
| Stop hit — per-position SL triggered | 2024-09-04 00:00:00 | 1958.00 | 1640.26 | 1916.43 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 00:00:00 | 2013.05 | 1663.26 | 1949.15 | Stage2 pullback-breakout RSI=63 vol=3.7x ATR=50.43 |
| Stop hit — per-position SL triggered | 2024-09-20 00:00:00 | 1937.40 | 1680.45 | 1975.12 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2024-12-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 00:00:00 | 1991.30 | 1781.09 | 1923.89 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=46.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 00:00:00 | 2084.88 | 1789.79 | 1963.91 | T1 booked 50% @ 2084.88 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-27 00:00:00 | 1829.30 | 2024-07-11 00:00:00 | 1831.90 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2024-08-21 00:00:00 | 1915.55 | 2024-09-04 00:00:00 | 1958.00 | STOP_HIT | 1.00 | 2.22% |
| BUY | retest1 | 2024-09-13 00:00:00 | 2013.05 | 2024-09-20 00:00:00 | 1937.40 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest1 | 2024-12-09 00:00:00 | 1991.30 | 2024-12-12 00:00:00 | 2084.88 | PARTIAL | 0.50 | 4.70% |
