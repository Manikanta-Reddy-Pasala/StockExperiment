# C.E. Info Systems Ltd. (MAPMYINDIA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 956.45
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 4.21% / 6.18%
- **Sum % (uncompounded):** 21.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.21% | 21.1% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.21% | 21.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.21% | 21.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 00:00:00 | 1773.80 | 1359.28 | 1707.25 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=57.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 00:00:00 | 1888.56 | 1364.64 | 1725.33 | T1 booked 50% @ 1888.56 |
| Target hit | 2023-10-20 00:00:00 | 1980.50 | 1469.07 | 1986.06 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 1903.00 | 1779.51 | 1804.06 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=64.59 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 1806.11 | 1787.95 | 1843.84 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-04-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 00:00:00 | 1942.45 | 1790.11 | 1853.88 | Stage2 pullback-breakout RSI=63 vol=2.9x ATR=60.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 00:00:00 | 2062.47 | 1801.10 | 1912.29 | T1 booked 50% @ 2062.47 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 1978.20 | 1809.45 | 1945.34 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-26 00:00:00 | 1773.80 | 2023-09-27 00:00:00 | 1888.56 | PARTIAL | 0.50 | 6.47% |
| BUY | retest1 | 2023-09-26 00:00:00 | 1773.80 | 2023-10-20 00:00:00 | 1980.50 | TARGET_HIT | 0.50 | 11.65% |
| BUY | retest1 | 2024-04-01 00:00:00 | 1903.00 | 2024-04-15 00:00:00 | 1806.11 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest1 | 2024-04-18 00:00:00 | 1942.45 | 2024-04-26 00:00:00 | 2062.47 | PARTIAL | 0.50 | 6.18% |
| BUY | retest1 | 2024-04-18 00:00:00 | 1942.45 | 2024-05-03 00:00:00 | 1978.20 | STOP_HIT | 0.50 | 1.84% |
