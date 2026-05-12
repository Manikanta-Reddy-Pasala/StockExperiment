# Colgate Palmolive (India) Ltd. (COLPAL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2182.20
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.79% / -1.28%
- **Sum % (uncompounded):** -7.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.79% | -7.2% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.79% | -7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.79% | -7.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 00:00:00 | 2067.90 | 1745.17 | 1991.53 | Stage2 pullback-breakout RSI=69 vol=1.6x ATR=41.17 |
| Stop hit — per-position SL triggered | 2023-09-28 00:00:00 | 2006.15 | 1750.50 | 1995.43 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 00:00:00 | 2060.75 | 1770.26 | 2004.29 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=43.09 |
| Stop hit — per-position SL triggered | 2023-10-26 00:00:00 | 2034.40 | 1798.75 | 2045.61 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-03-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 00:00:00 | 2687.40 | 2197.77 | 2567.71 | Stage2 pullback-breakout RSI=67 vol=2.8x ATR=66.06 |
| Stop hit — per-position SL triggered | 2024-04-01 00:00:00 | 2716.20 | 2245.17 | 2649.68 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 2782.60 | 2250.52 | 2662.34 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=73.42 |
| Stop hit — per-position SL triggered | 2024-04-09 00:00:00 | 2672.47 | 2272.95 | 2678.68 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-26 00:00:00 | 2067.90 | 2023-09-28 00:00:00 | 2006.15 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2023-10-11 00:00:00 | 2060.75 | 2023-10-26 00:00:00 | 2034.40 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest1 | 2024-03-14 00:00:00 | 2687.40 | 2024-04-01 00:00:00 | 2716.20 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest1 | 2024-04-02 00:00:00 | 2782.60 | 2024-04-09 00:00:00 | 2672.47 | STOP_HIT | 1.00 | -3.96% |
