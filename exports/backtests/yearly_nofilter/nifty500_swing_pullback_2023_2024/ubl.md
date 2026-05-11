# United Breweries Ltd. (UBL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1414.40
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
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 3.82% / 3.65%
- **Sum % (uncompounded):** 15.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 2 | 1 | 3.82% | 15.3% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 2 | 1 | 3.82% | 15.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 2 | 1 | 3.82% | 15.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 00:00:00 | 1610.65 | 1548.54 | 1576.42 | Stage2 pullback-breakout RSI=60 vol=7.0x ATR=33.73 |
| Stop hit — per-position SL triggered | 2023-10-26 00:00:00 | 1560.05 | 1549.99 | 1581.53 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-11-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 00:00:00 | 1597.60 | 1557.09 | 1585.30 | Stage2 pullback-breakout RSI=53 vol=2.2x ATR=29.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 00:00:00 | 1655.87 | 1560.13 | 1599.81 | T1 booked 50% @ 1655.87 |
| Target hit | 2024-01-29 00:00:00 | 1801.70 | 1632.61 | 1816.98 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 1804.75 | 1668.54 | 1731.53 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=43.88 |
| Stop hit — per-position SL triggered | 2024-04-18 00:00:00 | 1840.75 | 1683.81 | 1796.18 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-20 00:00:00 | 1610.65 | 2023-10-26 00:00:00 | 1560.05 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest1 | 2023-11-22 00:00:00 | 1597.60 | 2023-11-30 00:00:00 | 1655.87 | PARTIAL | 0.50 | 3.65% |
| BUY | retest1 | 2023-11-22 00:00:00 | 1597.60 | 2024-01-29 00:00:00 | 1801.70 | TARGET_HIT | 0.50 | 12.78% |
| BUY | retest1 | 2024-04-02 00:00:00 | 1804.75 | 2024-04-18 00:00:00 | 1840.75 | STOP_HIT | 1.00 | 1.99% |
