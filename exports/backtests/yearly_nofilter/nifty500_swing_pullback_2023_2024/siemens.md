# Siemens Ltd. (SIEMENS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3621.40
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** -0.30% / -2.99%
- **Sum % (uncompounded):** -2.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | -0.30% | -2.1% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | -0.30% | -2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | -0.30% | -2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 00:00:00 | 1872.52 | 1643.49 | 1843.60 | Stage2 pullback-breakout RSI=57 vol=1.5x ATR=37.32 |
| Stop hit — per-position SL triggered | 2023-07-20 00:00:00 | 1816.54 | 1657.17 | 1842.85 | SL hit (bars_held=7) |

### Cycle 2 — BUY (started 2023-07-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 00:00:00 | 1980.10 | 1671.76 | 1862.90 | Stage2 pullback-breakout RSI=69 vol=3.9x ATR=47.51 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 1908.83 | 1676.60 | 1872.45 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-08-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 00:00:00 | 1890.92 | 1699.03 | 1842.37 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=43.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 00:00:00 | 1978.53 | 1709.71 | 1873.19 | T1 booked 50% @ 1978.53 |
| Target hit | 2023-09-12 00:00:00 | 1908.96 | 1727.83 | 1912.95 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 00:00:00 | 2084.70 | 1795.98 | 1983.99 | Stage2 pullback-breakout RSI=69 vol=1.6x ATR=50.97 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 2008.24 | 1811.37 | 2017.54 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 2140.46 | 1828.45 | 2044.37 | Stage2 pullback-breakout RSI=68 vol=2.0x ATR=52.49 |
| Stop hit — per-position SL triggered | 2024-01-30 00:00:00 | 2061.72 | 1830.75 | 2045.85 | SL hit (bars_held=1) |

### Cycle 6 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 2457.03 | 1962.13 | 2325.39 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=76.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-27 00:00:00 | 2610.72 | 1979.47 | 2384.53 | T1 booked 50% @ 2610.72 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 00:00:00 | 1872.52 | 2023-07-20 00:00:00 | 1816.54 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2023-07-31 00:00:00 | 1980.10 | 2023-08-02 00:00:00 | 1908.83 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest1 | 2023-08-24 00:00:00 | 1890.92 | 2023-08-31 00:00:00 | 1978.53 | PARTIAL | 0.50 | 4.63% |
| BUY | retest1 | 2023-08-24 00:00:00 | 1890.92 | 2023-09-12 00:00:00 | 1908.96 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2024-01-09 00:00:00 | 2084.70 | 2024-01-17 00:00:00 | 2008.24 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest1 | 2024-01-29 00:00:00 | 2140.46 | 2024-01-30 00:00:00 | 2061.72 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest1 | 2024-03-21 00:00:00 | 2457.03 | 2024-03-27 00:00:00 | 2610.72 | PARTIAL | 0.50 | 6.26% |
