# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 136.31
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
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 4
- **Avg / median % per leg:** 4.21% / 6.58%
- **Sum % (uncompounded):** 37.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 2 | 3 | 4 | 4.21% | 37.9% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 3 | 4 | 4.21% | 37.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 2 | 3 | 4 | 4.21% | 37.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 60.65 | 49.21 | 56.96 | Stage2 pullback-breakout RSI=64 vol=3.7x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 64.75 | 49.50 | 58.22 | T1 booked 50% @ 64.75 |
| Target hit | 2023-07-28 00:00:00 | 67.70 | 52.86 | 67.86 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 115.35 | 81.44 | 110.45 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 00:00:00 | 122.94 | 83.19 | 113.23 | T1 booked 50% @ 122.94 |
| Target hit | 2023-12-20 00:00:00 | 119.20 | 86.23 | 120.42 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 00:00:00 | 137.20 | 92.98 | 127.11 | Stage2 pullback-breakout RSI=69 vol=1.5x ATR=5.02 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 129.67 | 93.80 | 128.44 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-02-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 00:00:00 | 139.10 | 97.15 | 131.17 | Stage2 pullback-breakout RSI=63 vol=1.9x ATR=5.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 00:00:00 | 150.92 | 98.04 | 133.16 | T1 booked 50% @ 150.92 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 139.10 | 100.21 | 136.76 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 136.15 | 114.42 | 133.39 | Stage2 pullback-breakout RSI=54 vol=2.0x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 00:00:00 | 145.14 | 115.32 | 134.67 | T1 booked 50% @ 145.14 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 136.15 | 115.46 | 134.17 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 60.65 | 2023-07-04 00:00:00 | 64.75 | PARTIAL | 0.50 | 6.77% |
| BUY | retest1 | 2023-06-30 00:00:00 | 60.65 | 2023-07-28 00:00:00 | 67.70 | TARGET_HIT | 0.50 | 11.62% |
| BUY | retest1 | 2023-12-04 00:00:00 | 115.35 | 2023-12-11 00:00:00 | 122.94 | PARTIAL | 0.50 | 6.58% |
| BUY | retest1 | 2023-12-04 00:00:00 | 115.35 | 2023-12-20 00:00:00 | 119.20 | TARGET_HIT | 0.50 | 3.34% |
| BUY | retest1 | 2024-01-16 00:00:00 | 137.20 | 2024-01-18 00:00:00 | 129.67 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest1 | 2024-02-01 00:00:00 | 139.10 | 2024-02-05 00:00:00 | 150.92 | PARTIAL | 0.50 | 8.50% |
| BUY | retest1 | 2024-02-01 00:00:00 | 139.10 | 2024-02-12 00:00:00 | 139.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-29 00:00:00 | 136.15 | 2024-05-06 00:00:00 | 145.14 | PARTIAL | 0.50 | 6.61% |
| BUY | retest1 | 2024-04-29 00:00:00 | 136.15 | 2024-05-07 00:00:00 | 136.15 | STOP_HIT | 0.50 | 0.00% |
