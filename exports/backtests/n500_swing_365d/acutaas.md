# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 2734.30
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 3 / 3 / 4
- **Avg / median % per leg:** 6.29% / 7.15%
- **Sum % (uncompounded):** 62.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 3 | 3 | 4 | 6.29% | 62.9% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 3 | 3 | 4 | 6.29% | 62.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 8 | 80.0% | 3 | 3 | 4 | 6.29% | 62.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 05:30:00 | 1220.30 | 1070.43 | 1166.58 | Stage2 pullback-breakout RSI=62 vol=3.4x ATR=41.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 05:30:00 | 1304.26 | 1072.99 | 1181.90 | T1 booked 50% @ 1304.26 |
| Target hit | 2025-09-25 05:30:00 | 1411.10 | 1178.52 | 1435.96 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 05:30:00 | 1498.40 | 1201.61 | 1420.66 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=45.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 05:30:00 | 1590.06 | 1215.74 | 1469.01 | T1 booked 50% @ 1590.06 |
| Target hit | 2025-11-13 05:30:00 | 1682.10 | 1298.16 | 1691.96 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-11-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 05:30:00 | 1872.20 | 1327.90 | 1718.52 | Stage2 pullback-breakout RSI=69 vol=5.4x ATR=70.20 |
| Stop hit — per-position SL triggered | 2025-11-26 05:30:00 | 1766.91 | 1337.04 | 1731.43 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-12-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 05:30:00 | 1704.90 | 1412.04 | 1679.38 | Stage2 pullback-breakout RSI=53 vol=1.5x ATR=56.24 |
| Stop hit — per-position SL triggered | 2026-01-14 05:30:00 | 1710.00 | 1442.07 | 1705.46 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-01-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 05:30:00 | 1862.40 | 1461.00 | 1701.36 | Stage2 pullback-breakout RSI=68 vol=3.6x ATR=66.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 05:30:00 | 1995.56 | 1483.19 | 1786.65 | T1 booked 50% @ 1995.56 |
| Target hit | 2026-03-13 05:30:00 | 2083.80 | 1633.47 | 2128.18 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2026-03-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 05:30:00 | 2302.60 | 1645.37 | 2147.97 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=95.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 05:30:00 | 2494.08 | 1678.33 | 2218.92 | T1 booked 50% @ 2494.08 |
| Stop hit — per-position SL triggered | 2026-04-01 05:30:00 | 2302.60 | 1710.16 | 2306.14 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-30 05:30:00 | 1220.30 | 2025-07-31 05:30:00 | 1304.26 | PARTIAL | 0.50 | 6.88% |
| BUY | retest1 | 2025-07-30 05:30:00 | 1220.30 | 2025-09-25 05:30:00 | 1411.10 | TARGET_HIT | 0.50 | 15.64% |
| BUY | retest1 | 2025-10-13 05:30:00 | 1498.40 | 2025-10-17 05:30:00 | 1590.06 | PARTIAL | 0.50 | 6.12% |
| BUY | retest1 | 2025-10-13 05:30:00 | 1498.40 | 2025-11-13 05:30:00 | 1682.10 | TARGET_HIT | 0.50 | 12.26% |
| BUY | retest1 | 2025-11-24 05:30:00 | 1872.20 | 2025-11-26 05:30:00 | 1766.91 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest1 | 2025-12-31 05:30:00 | 1704.90 | 2026-01-14 05:30:00 | 1710.00 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest1 | 2026-01-28 05:30:00 | 1862.40 | 2026-02-03 05:30:00 | 1995.56 | PARTIAL | 0.50 | 7.15% |
| BUY | retest1 | 2026-01-28 05:30:00 | 1862.40 | 2026-03-13 05:30:00 | 2083.80 | TARGET_HIT | 0.50 | 11.89% |
| BUY | retest1 | 2026-03-17 05:30:00 | 2302.60 | 2026-03-24 05:30:00 | 2494.08 | PARTIAL | 0.50 | 8.32% |
| BUY | retest1 | 2026-03-17 05:30:00 | 2302.60 | 2026-04-01 05:30:00 | 2302.60 | STOP_HIT | 0.50 | 0.00% |
