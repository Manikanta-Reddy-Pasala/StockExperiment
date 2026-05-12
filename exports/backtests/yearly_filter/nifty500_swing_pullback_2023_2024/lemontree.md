# Lemon Tree Hotels Ltd. (LEMONTREE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 117.85
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 1.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.20% | 1.4% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.20% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.20% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 00:00:00 | 96.65 | 85.93 | 92.37 | Stage2 pullback-breakout RSI=66 vol=4.5x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 93.62 | 86.02 | 92.61 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2023-12-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 00:00:00 | 118.65 | 101.35 | 115.03 | Stage2 pullback-breakout RSI=59 vol=2.3x ATR=3.20 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 115.90 | 103.07 | 117.86 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 00:00:00 | 129.85 | 104.43 | 119.73 | Stage2 pullback-breakout RSI=69 vol=9.9x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 00:00:00 | 138.15 | 106.77 | 127.15 | T1 booked 50% @ 138.15 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 129.85 | 107.58 | 128.97 | SL hit (bars_held=12) |

### Cycle 4 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 142.85 | 119.11 | 135.00 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 134.50 | 120.52 | 137.43 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 144.50 | 121.79 | 137.46 | Stage2 pullback-breakout RSI=64 vol=3.9x ATR=4.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 00:00:00 | 153.63 | 122.37 | 139.96 | T1 booked 50% @ 153.63 |
| Stop hit — per-position SL triggered | 2024-05-09 00:00:00 | 144.50 | 124.13 | 145.42 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-01 00:00:00 | 96.65 | 2023-08-02 00:00:00 | 93.62 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest1 | 2023-12-06 00:00:00 | 118.65 | 2023-12-20 00:00:00 | 115.90 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest1 | 2024-01-02 00:00:00 | 129.85 | 2024-01-15 00:00:00 | 138.15 | PARTIAL | 0.50 | 6.39% |
| BUY | retest1 | 2024-01-02 00:00:00 | 129.85 | 2024-01-18 00:00:00 | 129.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-03 00:00:00 | 142.85 | 2024-04-15 00:00:00 | 134.50 | STOP_HIT | 1.00 | -5.85% |
| BUY | retest1 | 2024-04-26 00:00:00 | 144.50 | 2024-04-30 00:00:00 | 153.63 | PARTIAL | 0.50 | 6.32% |
| BUY | retest1 | 2024-04-26 00:00:00 | 144.50 | 2024-05-09 00:00:00 | 144.50 | STOP_HIT | 0.50 | 0.00% |
