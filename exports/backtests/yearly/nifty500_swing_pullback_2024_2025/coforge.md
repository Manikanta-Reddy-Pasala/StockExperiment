# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1368.10
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
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 7.68% / 6.73%
- **Sum % (uncompounded):** 38.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.68% | 38.4% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.68% | 38.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 7.68% | 38.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 05:30:00 | 1251.09 | 1137.18 | 1208.28 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=33.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 05:30:00 | 1317.39 | 1147.02 | 1247.69 | T1 booked 50% @ 1317.39 |
| Target hit | 2024-10-21 05:30:00 | 1365.05 | 1216.14 | 1423.12 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-10-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 05:30:00 | 1511.56 | 1220.49 | 1426.11 | Stage2 pullback-breakout RSI=63 vol=5.8x ATR=50.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 05:30:00 | 1613.35 | 1260.47 | 1519.61 | T1 booked 50% @ 1613.35 |
| Target hit | 2025-01-09 05:30:00 | 1856.60 | 1443.80 | 1881.08 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-01-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 05:30:00 | 1839.29 | 1471.94 | 1788.68 | Stage2 pullback-breakout RSI=55 vol=4.1x ATR=68.39 |
| Stop hit — per-position SL triggered | 2025-01-28 05:30:00 | 1736.71 | 1480.70 | 1782.31 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-28 05:30:00 | 1251.09 | 2024-09-06 05:30:00 | 1317.39 | PARTIAL | 0.50 | 5.30% |
| BUY | retest1 | 2024-08-28 05:30:00 | 1251.09 | 2024-10-21 05:30:00 | 1365.05 | TARGET_HIT | 0.50 | 9.11% |
| BUY | retest1 | 2024-10-23 05:30:00 | 1511.56 | 2024-11-11 05:30:00 | 1613.35 | PARTIAL | 0.50 | 6.73% |
| BUY | retest1 | 2024-10-23 05:30:00 | 1511.56 | 2025-01-09 05:30:00 | 1856.60 | TARGET_HIT | 0.50 | 22.83% |
| BUY | retest1 | 2025-01-23 05:30:00 | 1839.29 | 2025-01-28 05:30:00 | 1736.71 | STOP_HIT | 1.00 | -5.58% |
