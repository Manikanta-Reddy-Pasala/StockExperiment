# C.E. Info Systems Ltd. (MAPMYINDIA)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -5.87% / -5.63%
- **Sum % (uncompounded):** -17.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.87% | -17.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.87% | -17.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.87% | -17.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 00:00:00 | 2317.90 | 2049.27 | 2291.07 | Stage2 pullback-breakout RSI=52 vol=1.8x ATR=110.88 |
| Stop hit — per-position SL triggered | 2024-08-14 00:00:00 | 2151.59 | 2051.35 | 2265.91 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-09-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 00:00:00 | 2162.50 | 2062.78 | 2092.29 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=69.18 |
| Stop hit — per-position SL triggered | 2024-10-07 00:00:00 | 2058.73 | 2066.93 | 2112.38 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 00:00:00 | 2070.00 | 2060.43 | 2009.88 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=77.69 |
| Stop hit — per-position SL triggered | 2024-11-11 00:00:00 | 1953.47 | 2058.71 | 2002.60 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-12 00:00:00 | 2317.90 | 2024-08-14 00:00:00 | 2151.59 | STOP_HIT | 1.00 | -7.18% |
| BUY | retest1 | 2024-09-27 00:00:00 | 2162.50 | 2024-10-07 00:00:00 | 2058.73 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest1 | 2024-11-07 00:00:00 | 2070.00 | 2024-11-11 00:00:00 | 1953.47 | STOP_HIT | 1.00 | -5.63% |
