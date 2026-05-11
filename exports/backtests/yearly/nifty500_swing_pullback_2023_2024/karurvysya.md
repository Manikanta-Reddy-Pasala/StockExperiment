# Karur Vysya Bank Ltd. (KARURVYSYA)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 304.65
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 2.28% / 0.00%
- **Sum % (uncompounded):** 22.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 2.28% | 22.8% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 2.28% | 22.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 4 | 40.0% | 1 | 6 | 3 | 2.28% | 22.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 05:30:00 | 104.96 | 89.80 | 102.47 | Stage2 pullback-breakout RSI=56 vol=3.0x ATR=3.54 |
| Stop hit — per-position SL triggered | 2023-08-24 05:30:00 | 99.66 | 89.92 | 102.42 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2023-09-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 05:30:00 | 107.71 | 90.85 | 102.35 | Stage2 pullback-breakout RSI=64 vol=3.2x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 05:30:00 | 115.07 | 92.42 | 106.51 | T1 booked 50% @ 115.07 |
| Stop hit — per-position SL triggered | 2023-09-28 05:30:00 | 107.71 | 93.77 | 109.48 | SL hit (bars_held=16) |

### Cycle 3 — BUY (started 2023-10-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 05:30:00 | 114.67 | 95.64 | 111.12 | Stage2 pullback-breakout RSI=63 vol=2.7x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 05:30:00 | 122.14 | 95.88 | 111.97 | T1 booked 50% @ 122.14 |
| Target hit | 2024-01-08 05:30:00 | 136.38 | 111.16 | 137.40 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 05:30:00 | 145.42 | 112.56 | 138.54 | Stage2 pullback-breakout RSI=67 vol=2.0x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-01-17 05:30:00 | 139.45 | 113.15 | 139.27 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2024-01-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 05:30:00 | 152.21 | 114.38 | 141.12 | Stage2 pullback-breakout RSI=69 vol=7.4x ATR=4.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 05:30:00 | 162.07 | 115.98 | 145.84 | T1 booked 50% @ 162.07 |
| Stop hit — per-position SL triggered | 2024-02-07 05:30:00 | 152.21 | 118.52 | 151.75 | SL hit (bars_held=10) |

### Cycle 6 — BUY (started 2024-02-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 05:30:00 | 154.33 | 122.89 | 151.88 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=5.35 |
| Stop hit — per-position SL triggered | 2024-03-11 05:30:00 | 146.30 | 125.42 | 151.76 | SL hit (bars_held=9) |

### Cycle 7 — BUY (started 2024-04-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 05:30:00 | 162.38 | 129.49 | 153.22 | Stage2 pullback-breakout RSI=67 vol=4.1x ATR=5.62 |
| Stop hit — per-position SL triggered | 2024-04-23 05:30:00 | 159.75 | 132.32 | 157.01 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-23 05:30:00 | 104.96 | 2023-08-24 05:30:00 | 99.66 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest1 | 2023-09-05 05:30:00 | 107.71 | 2023-09-18 05:30:00 | 115.07 | PARTIAL | 0.50 | 6.83% |
| BUY | retest1 | 2023-09-05 05:30:00 | 107.71 | 2023-09-28 05:30:00 | 107.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-16 05:30:00 | 114.67 | 2023-10-17 05:30:00 | 122.14 | PARTIAL | 0.50 | 6.52% |
| BUY | retest1 | 2023-10-16 05:30:00 | 114.67 | 2024-01-08 05:30:00 | 136.38 | TARGET_HIT | 0.50 | 18.93% |
| BUY | retest1 | 2024-01-15 05:30:00 | 145.42 | 2024-01-17 05:30:00 | 139.45 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest1 | 2024-01-23 05:30:00 | 152.21 | 2024-01-30 05:30:00 | 162.07 | PARTIAL | 0.50 | 6.48% |
| BUY | retest1 | 2024-01-23 05:30:00 | 152.21 | 2024-02-07 05:30:00 | 152.21 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-27 05:30:00 | 154.33 | 2024-03-11 05:30:00 | 146.30 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest1 | 2024-04-05 05:30:00 | 162.38 | 2024-04-23 05:30:00 | 159.75 | STOP_HIT | 1.00 | -1.62% |
