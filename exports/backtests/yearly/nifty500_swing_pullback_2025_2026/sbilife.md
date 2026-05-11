# SBI Life Insurance Company Ltd. (SBILIFE)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 1872.10
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 0.39% / 3.31%
- **Sum % (uncompounded):** 1.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.39% | 1.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.39% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.39% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 05:30:00 | 1832.10 | 1697.90 | 1814.41 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=32.41 |
| Stop hit — per-position SL triggered | 2025-08-04 05:30:00 | 1783.49 | 1705.69 | 1821.56 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2025-09-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 05:30:00 | 1841.70 | 1739.53 | 1821.60 | Stage2 pullback-breakout RSI=55 vol=1.9x ATR=33.61 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 1791.29 | 1743.47 | 1820.21 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 1903.10 | 1756.64 | 1831.58 | Stage2 pullback-breakout RSI=69 vol=2.9x ATR=34.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 05:30:00 | 1972.73 | 1760.54 | 1853.87 | T1 booked 50% @ 1972.73 |
| Target hit | 2025-11-28 05:30:00 | 1966.00 | 1805.13 | 1983.83 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-25 05:30:00 | 1832.10 | 2025-08-04 05:30:00 | 1783.49 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2025-09-19 05:30:00 | 1841.70 | 2025-09-26 05:30:00 | 1791.29 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest1 | 2025-10-27 05:30:00 | 1903.10 | 2025-10-29 05:30:00 | 1972.73 | PARTIAL | 0.50 | 3.66% |
| BUY | retest1 | 2025-10-27 05:30:00 | 1903.10 | 2025-11-28 05:30:00 | 1966.00 | TARGET_HIT | 0.50 | 3.31% |
