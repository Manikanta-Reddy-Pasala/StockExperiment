# Data Patterns (India) Ltd. (DATAPATTNS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 4103.40
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 1
- **Target hits / Stop hits / Partials:** 2 / 3 / 4
- **Avg / median % per leg:** 6.32% / 6.41%
- **Sum % (uncompounded):** 56.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 8 | 88.9% | 2 | 3 | 4 | 6.32% | 56.9% |
| BUY @ 2nd Alert (retest1) | 9 | 8 | 88.9% | 2 | 3 | 4 | 6.32% | 56.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 8 | 88.9% | 2 | 3 | 4 | 6.32% | 56.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 00:00:00 | 2112.30 | 1583.32 | 2005.73 | Stage2 pullback-breakout RSI=66 vol=2.8x ATR=65.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 00:00:00 | 2243.21 | 1610.55 | 2061.37 | T1 booked 50% @ 2243.21 |
| Target hit | 2023-09-06 00:00:00 | 2250.35 | 1706.51 | 2278.41 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 00:00:00 | 1979.65 | 1821.51 | 1896.49 | Stage2 pullback-breakout RSI=57 vol=4.9x ATR=63.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 2106.55 | 1830.88 | 1945.25 | T1 booked 50% @ 2106.55 |
| Stop hit — per-position SL triggered | 2023-12-11 00:00:00 | 2030.10 | 1840.55 | 1977.72 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 00:00:00 | 1978.55 | 1856.63 | 1925.73 | Stage2 pullback-breakout RSI=56 vol=2.8x ATR=59.96 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 1888.61 | 1863.95 | 1944.89 | SL hit (bars_held=7) |

### Cycle 4 — BUY (started 2024-02-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 00:00:00 | 2019.15 | 1869.31 | 1898.21 | Stage2 pullback-breakout RSI=61 vol=7.6x ATR=74.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-22 00:00:00 | 2168.99 | 1878.34 | 1966.78 | T1 booked 50% @ 2168.99 |
| Target hit | 2024-03-12 00:00:00 | 2423.85 | 1966.43 | 2433.10 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 2713.30 | 2017.38 | 2424.60 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=154.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 00:00:00 | 3022.13 | 2059.87 | 2574.99 | T1 booked 50% @ 3022.13 |
| Stop hit — per-position SL triggered | 2024-04-18 00:00:00 | 2737.35 | 2090.03 | 2656.40 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-09 00:00:00 | 2112.30 | 2023-08-17 00:00:00 | 2243.21 | PARTIAL | 0.50 | 6.20% |
| BUY | retest1 | 2023-08-09 00:00:00 | 2112.30 | 2023-09-06 00:00:00 | 2250.35 | TARGET_HIT | 0.50 | 6.54% |
| BUY | retest1 | 2023-11-24 00:00:00 | 1979.65 | 2023-12-04 00:00:00 | 2106.55 | PARTIAL | 0.50 | 6.41% |
| BUY | retest1 | 2023-11-24 00:00:00 | 1979.65 | 2023-12-11 00:00:00 | 2030.10 | STOP_HIT | 0.50 | 2.55% |
| BUY | retest1 | 2024-01-08 00:00:00 | 1978.55 | 2024-01-17 00:00:00 | 1888.61 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest1 | 2024-02-16 00:00:00 | 2019.15 | 2024-02-22 00:00:00 | 2168.99 | PARTIAL | 0.50 | 7.42% |
| BUY | retest1 | 2024-02-16 00:00:00 | 2019.15 | 2024-03-12 00:00:00 | 2423.85 | TARGET_HIT | 0.50 | 20.04% |
| BUY | retest1 | 2024-04-02 00:00:00 | 2713.30 | 2024-04-10 00:00:00 | 3022.13 | PARTIAL | 0.50 | 11.38% |
| BUY | retest1 | 2024-04-02 00:00:00 | 2713.30 | 2024-04-18 00:00:00 | 2737.35 | STOP_HIT | 0.50 | 0.89% |
