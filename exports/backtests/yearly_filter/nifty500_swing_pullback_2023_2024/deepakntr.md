# Deepak Nitrite Ltd. (DEEPAKNTR)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1859.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 0.56% / 2.00%
- **Sum % (uncompounded):** 4.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.56% | 4.5% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.56% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.56% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 00:00:00 | 2170.20 | 2006.67 | 2039.51 | Stage2 pullback-breakout RSI=68 vol=5.4x ATR=51.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 00:00:00 | 2274.09 | 2017.94 | 2118.67 | T1 booked 50% @ 2274.09 |
| Target hit | 2023-09-20 00:00:00 | 2213.70 | 2042.72 | 2214.08 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 2138.35 | 2050.00 | 2049.76 | Stage2 pullback-breakout RSI=60 vol=2.3x ATR=52.24 |
| Stop hit — per-position SL triggered | 2023-11-10 00:00:00 | 2059.99 | 2052.25 | 2068.64 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 00:00:00 | 2206.20 | 2059.54 | 2111.72 | Stage2 pullback-breakout RSI=64 vol=3.9x ATR=52.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 00:00:00 | 2311.14 | 2071.21 | 2168.90 | T1 booked 50% @ 2311.14 |
| Stop hit — per-position SL triggered | 2023-12-08 00:00:00 | 2206.20 | 2074.50 | 2181.15 | SL hit (bars_held=10) |

### Cycle 4 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 2306.35 | 2083.30 | 2211.67 | Stage2 pullback-breakout RSI=70 vol=4.9x ATR=56.48 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 2221.63 | 2088.88 | 2226.63 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-02-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 00:00:00 | 2282.70 | 2167.42 | 2265.29 | Stage2 pullback-breakout RSI=51 vol=1.8x ATR=69.02 |
| Stop hit — per-position SL triggered | 2024-02-29 00:00:00 | 2179.17 | 2178.74 | 2271.99 | SL hit (bars_held=10) |

### Cycle 6 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 2214.45 | 2173.93 | 2168.61 | Stage2 pullback-breakout RSI=55 vol=2.7x ATR=53.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 00:00:00 | 2321.19 | 2177.49 | 2201.58 | T1 booked 50% @ 2321.19 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-29 00:00:00 | 2170.20 | 2023-09-05 00:00:00 | 2274.09 | PARTIAL | 0.50 | 4.79% |
| BUY | retest1 | 2023-08-29 00:00:00 | 2170.20 | 2023-09-20 00:00:00 | 2213.70 | TARGET_HIT | 0.50 | 2.00% |
| BUY | retest1 | 2023-11-06 00:00:00 | 2138.35 | 2023-11-10 00:00:00 | 2059.99 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest1 | 2023-11-23 00:00:00 | 2206.20 | 2023-12-06 00:00:00 | 2311.14 | PARTIAL | 0.50 | 4.76% |
| BUY | retest1 | 2023-11-23 00:00:00 | 2206.20 | 2023-12-08 00:00:00 | 2206.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-15 00:00:00 | 2306.35 | 2023-12-20 00:00:00 | 2221.63 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest1 | 2024-02-15 00:00:00 | 2282.70 | 2024-02-29 00:00:00 | 2179.17 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest1 | 2024-04-04 00:00:00 | 2214.45 | 2024-04-12 00:00:00 | 2321.19 | PARTIAL | 0.50 | 4.82% |
