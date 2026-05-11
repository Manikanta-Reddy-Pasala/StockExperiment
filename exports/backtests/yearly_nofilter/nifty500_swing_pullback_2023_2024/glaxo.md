# Glaxosmithkline Pharmaceuticals Ltd. (GLAXO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2437.60
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
- **Avg / median % per leg:** 9.85% / 5.24%
- **Sum % (uncompounded):** 39.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 9.85% | 39.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 9.85% | 39.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 9.85% | 39.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 00:00:00 | 1416.25 | 1344.51 | 1403.02 | Stage2 pullback-breakout RSI=57 vol=2.9x ATR=27.28 |
| Stop hit — per-position SL triggered | 2023-07-31 00:00:00 | 1396.10 | 1349.73 | 1400.37 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 00:00:00 | 1552.70 | 1414.99 | 1470.16 | Stage2 pullback-breakout RSI=66 vol=2.8x ATR=40.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 00:00:00 | 1634.01 | 1416.72 | 1481.38 | T1 booked 50% @ 1634.01 |
| Target hit | 2024-02-13 00:00:00 | 2179.65 | 1691.91 | 2273.64 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-04-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 00:00:00 | 2077.25 | 1814.33 | 1952.16 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=66.62 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 1977.31 | 1838.62 | 2020.37 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-17 00:00:00 | 1416.25 | 2023-07-31 00:00:00 | 1396.10 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest1 | 2023-11-15 00:00:00 | 1552.70 | 2023-11-16 00:00:00 | 1634.01 | PARTIAL | 0.50 | 5.24% |
| BUY | retest1 | 2023-11-15 00:00:00 | 1552.70 | 2024-02-13 00:00:00 | 2179.65 | TARGET_HIT | 0.50 | 40.38% |
| BUY | retest1 | 2024-04-24 00:00:00 | 2077.25 | 2024-05-09 00:00:00 | 1977.31 | STOP_HIT | 1.00 | -4.81% |
