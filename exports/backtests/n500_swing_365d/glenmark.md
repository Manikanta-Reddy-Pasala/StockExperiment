# Glenmark Pharmaceuticals Ltd. (GLENMARK)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 2366.40
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 2.07% / 1.90%
- **Sum % (uncompounded):** 14.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 3 | 3 | 2.07% | 14.5% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 3 | 3 | 2.07% | 14.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 3 | 3 | 2.07% | 14.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 05:30:00 | 2004.40 | 1711.57 | 1978.74 | Stage2 pullback-breakout RSI=54 vol=1.9x ATR=49.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 05:30:00 | 2103.03 | 1724.97 | 2004.70 | T1 booked 50% @ 2103.03 |
| Target hit | 2025-09-17 05:30:00 | 2042.40 | 1746.88 | 2046.06 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-12-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 05:30:00 | 2047.00 | 1831.48 | 1963.02 | Stage2 pullback-breakout RSI=68 vol=2.1x ATR=43.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 05:30:00 | 2133.08 | 1851.62 | 2018.16 | T1 booked 50% @ 2133.08 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 2079.80 | 1853.89 | 2024.03 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-03-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 05:30:00 | 2225.80 | 1912.11 | 2079.99 | Stage2 pullback-breakout RSI=69 vol=3.2x ATR=60.60 |
| Stop hit — per-position SL triggered | 2026-03-19 05:30:00 | 2134.90 | 1930.59 | 2128.73 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2026-04-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 05:30:00 | 2335.00 | 1978.10 | 2200.77 | Stage2 pullback-breakout RSI=69 vol=2.6x ATR=69.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 05:30:00 | 2473.89 | 1993.12 | 2255.21 | T1 booked 50% @ 2473.89 |
| Stop hit — per-position SL triggered | 2026-05-08 05:30:00 | 2335.00 | 2016.15 | 2314.41 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-03 05:30:00 | 2004.40 | 2025-09-09 05:30:00 | 2103.03 | PARTIAL | 0.50 | 4.92% |
| BUY | retest1 | 2025-09-03 05:30:00 | 2004.40 | 2025-09-17 05:30:00 | 2042.40 | TARGET_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2025-12-23 05:30:00 | 2047.00 | 2026-01-07 05:30:00 | 2133.08 | PARTIAL | 0.50 | 4.21% |
| BUY | retest1 | 2025-12-23 05:30:00 | 2047.00 | 2026-01-08 05:30:00 | 2079.80 | STOP_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2026-03-10 05:30:00 | 2225.80 | 2026-03-19 05:30:00 | 2134.90 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest1 | 2026-04-23 05:30:00 | 2335.00 | 2026-04-29 05:30:00 | 2473.89 | PARTIAL | 0.50 | 5.95% |
| BUY | retest1 | 2026-04-23 05:30:00 | 2335.00 | 2026-05-08 05:30:00 | 2335.00 | STOP_HIT | 0.50 | 0.00% |
