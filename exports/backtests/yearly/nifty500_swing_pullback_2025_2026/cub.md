# City Union Bank Ltd. (CUB)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 260.35
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
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 0.33% / 0.00%
- **Sum % (uncompounded):** 2.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.33% | 2.3% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.33% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 5 | 2 | 0.33% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 05:30:00 | 205.13 | 176.21 | 195.81 | Stage2 pullback-breakout RSI=65 vol=1.7x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 05:30:00 | 216.02 | 176.93 | 198.85 | T1 booked 50% @ 216.02 |
| Stop hit — per-position SL triggered | 2025-07-11 05:30:00 | 210.13 | 180.46 | 209.63 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-10-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 05:30:00 | 220.58 | 192.78 | 209.64 | Stage2 pullback-breakout RSI=64 vol=2.1x ATR=5.64 |
| Stop hit — per-position SL triggered | 2025-10-09 05:30:00 | 212.12 | 193.47 | 211.19 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-10-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 05:30:00 | 239.27 | 196.88 | 221.33 | Stage2 pullback-breakout RSI=69 vol=2.4x ATR=7.47 |
| Stop hit — per-position SL triggered | 2025-10-31 05:30:00 | 228.06 | 197.96 | 224.42 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2026-01-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 05:30:00 | 301.75 | 234.06 | 283.68 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=11.27 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 284.84 | 234.55 | 283.68 | SL hit (bars_held=1) |

### Cycle 5 — BUY (started 2026-04-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 05:30:00 | 265.75 | 245.60 | 252.85 | Stage2 pullback-breakout RSI=58 vol=2.6x ATR=11.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 05:30:00 | 288.78 | 246.93 | 260.05 | T1 booked 50% @ 288.78 |
| Stop hit — per-position SL triggered | 2026-05-05 05:30:00 | 265.75 | 247.92 | 263.95 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-26 05:30:00 | 205.13 | 2025-06-30 05:30:00 | 216.02 | PARTIAL | 0.50 | 5.31% |
| BUY | retest1 | 2025-06-26 05:30:00 | 205.13 | 2025-07-11 05:30:00 | 210.13 | STOP_HIT | 0.50 | 2.44% |
| BUY | retest1 | 2025-10-06 05:30:00 | 220.58 | 2025-10-09 05:30:00 | 212.12 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest1 | 2025-10-28 05:30:00 | 239.27 | 2025-10-31 05:30:00 | 228.06 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest1 | 2026-01-30 05:30:00 | 301.75 | 2026-02-01 05:30:00 | 284.84 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest1 | 2026-04-20 05:30:00 | 265.75 | 2026-04-28 05:30:00 | 288.78 | PARTIAL | 0.50 | 8.67% |
| BUY | retest1 | 2026-04-20 05:30:00 | 265.75 | 2026-05-05 05:30:00 | 265.75 | STOP_HIT | 0.50 | 0.00% |
