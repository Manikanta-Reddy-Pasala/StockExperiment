# IDBI Bank Ltd. (IDBI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 73.00
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
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 0 / 6 / 3
- **Avg / median % per leg:** 1.00% / 0.00%
- **Sum % (uncompounded):** 8.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 0 | 6 | 3 | 1.00% | 9.0% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 6 | 3 | 1.00% | 9.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 0 | 6 | 3 | 1.00% | 9.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 56.70 | 50.56 | 54.50 | Stage2 pullback-breakout RSI=66 vol=3.1x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 00:00:00 | 59.25 | 50.91 | 55.84 | T1 booked 50% @ 59.25 |
| Stop hit — per-position SL triggered | 2023-07-17 00:00:00 | 58.15 | 51.27 | 56.74 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 64.90 | 54.05 | 61.20 | Stage2 pullback-breakout RSI=65 vol=4.9x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 00:00:00 | 68.76 | 54.21 | 62.05 | T1 booked 50% @ 68.76 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 64.90 | 55.07 | 65.11 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-12-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 00:00:00 | 67.10 | 59.57 | 63.45 | Stage2 pullback-breakout RSI=65 vol=3.3x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-18 00:00:00 | 70.86 | 60.06 | 65.28 | T1 booked 50% @ 70.86 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 67.10 | 60.21 | 65.61 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 69.30 | 60.85 | 66.62 | Stage2 pullback-breakout RSI=63 vol=1.5x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 69.30 | 61.56 | 67.81 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 87.85 | 71.00 | 83.15 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-04-19 00:00:00 | 84.00 | 72.52 | 85.27 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 91.70 | 73.43 | 86.68 | Stage2 pullback-breakout RSI=64 vol=2.8x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-05-07 00:00:00 | 86.90 | 74.19 | 87.48 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 56.70 | 2023-07-10 00:00:00 | 59.25 | PARTIAL | 0.50 | 4.49% |
| BUY | retest1 | 2023-07-03 00:00:00 | 56.70 | 2023-07-17 00:00:00 | 58.15 | STOP_HIT | 0.50 | 2.56% |
| BUY | retest1 | 2023-09-04 00:00:00 | 64.90 | 2023-09-05 00:00:00 | 68.76 | PARTIAL | 0.50 | 5.95% |
| BUY | retest1 | 2023-09-04 00:00:00 | 64.90 | 2023-09-13 00:00:00 | 64.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-07 00:00:00 | 67.10 | 2023-12-18 00:00:00 | 70.86 | PARTIAL | 0.50 | 5.60% |
| BUY | retest1 | 2023-12-07 00:00:00 | 67.10 | 2023-12-20 00:00:00 | 67.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-04 00:00:00 | 69.30 | 2024-01-18 00:00:00 | 69.30 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest1 | 2024-04-03 00:00:00 | 87.85 | 2024-04-19 00:00:00 | 84.00 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest1 | 2024-04-29 00:00:00 | 91.70 | 2024-05-07 00:00:00 | 86.90 | STOP_HIT | 1.00 | -5.23% |
