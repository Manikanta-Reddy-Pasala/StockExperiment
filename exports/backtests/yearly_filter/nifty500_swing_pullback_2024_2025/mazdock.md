# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 2576.10
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
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 4.08% / 8.33%
- **Sum % (uncompounded):** 20.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.08% | 20.4% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.08% | 20.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 4.08% | 20.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 00:00:00 | 2387.40 | 1673.05 | 2277.36 | Stage2 pullback-breakout RSI=55 vol=2.5x ATR=129.19 |
| Stop hit — per-position SL triggered | 2024-09-06 00:00:00 | 2193.61 | 1684.37 | 2271.06 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 00:00:00 | 2214.18 | 1771.39 | 2110.37 | Stage2 pullback-breakout RSI=56 vol=4.5x ATR=95.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 00:00:00 | 2404.52 | 1800.53 | 2160.88 | T1 booked 50% @ 2404.52 |
| Stop hit — per-position SL triggered | 2024-10-22 00:00:00 | 2214.18 | 1803.30 | 2153.10 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2024-11-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 00:00:00 | 2117.07 | 1853.00 | 2053.32 | Stage2 pullback-breakout RSI=55 vol=2.1x ATR=88.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 00:00:00 | 2293.47 | 1860.82 | 2088.72 | T1 booked 50% @ 2293.47 |
| Target hit | 2024-12-20 00:00:00 | 2362.05 | 1947.24 | 2389.82 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-04 00:00:00 | 2387.40 | 2024-09-06 00:00:00 | 2193.61 | STOP_HIT | 1.00 | -8.12% |
| BUY | retest1 | 2024-10-10 00:00:00 | 2214.18 | 2024-10-21 00:00:00 | 2404.52 | PARTIAL | 0.50 | 8.60% |
| BUY | retest1 | 2024-10-10 00:00:00 | 2214.18 | 2024-10-22 00:00:00 | 2214.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 00:00:00 | 2117.07 | 2024-11-28 00:00:00 | 2293.47 | PARTIAL | 0.50 | 8.33% |
| BUY | retest1 | 2024-11-26 00:00:00 | 2117.07 | 2024-12-20 00:00:00 | 2362.05 | TARGET_HIT | 0.50 | 11.57% |
