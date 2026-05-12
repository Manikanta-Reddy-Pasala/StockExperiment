# BEL (BEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 439.70
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
| TARGET_HIT | 3 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 3 / 3 / 3
- **Avg / median % per leg:** 4.29% / 2.90%
- **Sum % (uncompounded):** 38.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 3 | 3 | 3 | 4.29% | 38.6% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 3 | 3 | 3 | 4.29% | 38.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 3 | 3 | 3 | 4.29% | 38.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 00:00:00 | 127.40 | 107.46 | 122.01 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=3.37 |
| Stop hit — per-position SL triggered | 2023-07-25 00:00:00 | 127.05 | 109.28 | 124.83 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 130.15 | 109.83 | 125.64 | Stage2 pullback-breakout RSI=65 vol=2.5x ATR=2.85 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 125.88 | 110.39 | 126.42 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2023-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 00:00:00 | 133.30 | 112.72 | 128.91 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 00:00:00 | 139.62 | 114.66 | 132.91 | T1 booked 50% @ 139.62 |
| Target hit | 2023-09-12 00:00:00 | 134.15 | 116.13 | 136.08 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 143.40 | 123.17 | 137.20 | Stage2 pullback-breakout RSI=65 vol=2.2x ATR=3.02 |
| Stop hit — per-position SL triggered | 2023-11-24 00:00:00 | 138.87 | 124.65 | 139.94 | SL hit (bars_held=8) |

### Cycle 5 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 145.90 | 125.18 | 140.71 | Stage2 pullback-breakout RSI=63 vol=2.5x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 152.32 | 125.69 | 142.51 | T1 booked 50% @ 152.32 |
| Target hit | 2024-02-01 00:00:00 | 183.40 | 143.89 | 185.42 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 00:00:00 | 221.00 | 161.64 | 200.89 | Stage2 pullback-breakout RSI=70 vol=2.6x ATR=7.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 00:00:00 | 236.27 | 166.66 | 215.79 | T1 booked 50% @ 236.27 |
| Target hit | 2024-05-07 00:00:00 | 227.40 | 175.46 | 229.23 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 00:00:00 | 127.40 | 2023-07-25 00:00:00 | 127.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-28 00:00:00 | 130.15 | 2023-08-02 00:00:00 | 125.88 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest1 | 2023-08-22 00:00:00 | 133.30 | 2023-09-04 00:00:00 | 139.62 | PARTIAL | 0.50 | 4.74% |
| BUY | retest1 | 2023-08-22 00:00:00 | 133.30 | 2023-09-12 00:00:00 | 134.15 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2023-11-13 00:00:00 | 143.40 | 2023-11-24 00:00:00 | 138.87 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest1 | 2023-11-30 00:00:00 | 145.90 | 2023-12-04 00:00:00 | 152.32 | PARTIAL | 0.50 | 4.40% |
| BUY | retest1 | 2023-11-30 00:00:00 | 145.90 | 2024-02-01 00:00:00 | 183.40 | TARGET_HIT | 0.50 | 25.70% |
| BUY | retest1 | 2024-04-02 00:00:00 | 221.00 | 2024-04-15 00:00:00 | 236.27 | PARTIAL | 0.50 | 6.91% |
| BUY | retest1 | 2024-04-02 00:00:00 | 221.00 | 2024-05-07 00:00:00 | 227.40 | TARGET_HIT | 0.50 | 2.90% |
