# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1652.10
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
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 2
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 0.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.06% | 0.5% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.06% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.06% | 0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 05:30:00 | 2116.85 | 1625.35 | 1967.56 | Stage2 pullback-breakout RSI=69 vol=2.6x ATR=78.52 |
| Stop hit — per-position SL triggered | 2024-07-23 05:30:00 | 1999.07 | 1670.85 | 2047.77 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-08-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 05:30:00 | 2126.25 | 1712.05 | 1945.71 | Stage2 pullback-breakout RSI=67 vol=3.6x ATR=76.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 05:30:00 | 2278.62 | 1736.55 | 2053.75 | T1 booked 50% @ 2278.62 |
| Target hit | 2024-09-20 05:30:00 | 2341.60 | 1860.86 | 2402.77 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-10-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 05:30:00 | 2524.25 | 1958.61 | 2405.47 | Stage2 pullback-breakout RSI=64 vol=3.9x ATR=101.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 05:30:00 | 2727.42 | 1996.45 | 2501.37 | T1 booked 50% @ 2727.42 |
| Stop hit — per-position SL triggered | 2024-11-13 05:30:00 | 2524.25 | 2075.14 | 2688.77 | SL hit (bars_held=16) |

### Cycle 4 — BUY (started 2024-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 05:30:00 | 3002.35 | 2110.61 | 2692.06 | Stage2 pullback-breakout RSI=66 vol=5.8x ATR=157.11 |
| Stop hit — per-position SL triggered | 2024-11-26 05:30:00 | 2766.68 | 2117.11 | 2698.90 | SL hit (bars_held=1) |

### Cycle 5 — BUY (started 2024-12-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 05:30:00 | 2875.70 | 2155.34 | 2736.25 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=149.01 |
| Stop hit — per-position SL triggered | 2024-12-18 05:30:00 | 2721.90 | 2222.67 | 2807.87 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2025-01-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 05:30:00 | 2826.20 | 2268.92 | 2702.34 | Stage2 pullback-breakout RSI=58 vol=2.3x ATR=114.31 |
| Stop hit — per-position SL triggered | 2025-01-10 05:30:00 | 2654.74 | 2287.41 | 2712.88 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-08 05:30:00 | 2116.85 | 2024-07-23 05:30:00 | 1999.07 | STOP_HIT | 1.00 | -5.56% |
| BUY | retest1 | 2024-08-19 05:30:00 | 2126.25 | 2024-08-26 05:30:00 | 2278.62 | PARTIAL | 0.50 | 7.17% |
| BUY | retest1 | 2024-08-19 05:30:00 | 2126.25 | 2024-09-20 05:30:00 | 2341.60 | TARGET_HIT | 0.50 | 10.13% |
| BUY | retest1 | 2024-10-22 05:30:00 | 2524.25 | 2024-10-30 05:30:00 | 2727.42 | PARTIAL | 0.50 | 8.05% |
| BUY | retest1 | 2024-10-22 05:30:00 | 2524.25 | 2024-11-13 05:30:00 | 2524.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 05:30:00 | 3002.35 | 2024-11-26 05:30:00 | 2766.68 | STOP_HIT | 1.00 | -7.85% |
| BUY | retest1 | 2024-12-04 05:30:00 | 2875.70 | 2024-12-18 05:30:00 | 2721.90 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest1 | 2025-01-06 05:30:00 | 2826.20 | 2025-01-10 05:30:00 | 2654.74 | STOP_HIT | 1.00 | -6.07% |
