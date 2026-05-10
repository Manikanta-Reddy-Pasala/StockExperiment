# IDBI Bank Ltd. (IDBI)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 74.72
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -1.99% / -2.29%
- **Sum % (uncompounded):** -9.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 5 | 0 | -1.99% | -9.9% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 5 | 0 | -1.99% | -9.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | -1.99% | -9.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 05:30:00 | 101.32 | 85.33 | 94.48 | Stage2 pullback-breakout RSI=66 vol=3.3x ATR=3.48 |
| Stop hit — per-position SL triggered | 2025-07-11 05:30:00 | 99.00 | 86.85 | 98.55 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-08-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 05:30:00 | 97.58 | 88.34 | 91.75 | Stage2 pullback-breakout RSI=63 vol=7.3x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-08-26 05:30:00 | 93.28 | 88.50 | 92.25 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-09-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 05:30:00 | 92.27 | 88.54 | 90.63 | Stage2 pullback-breakout RSI=52 vol=2.1x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-09-22 05:30:00 | 93.20 | 88.99 | 92.26 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-12-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 05:30:00 | 99.78 | 92.48 | 98.14 | Stage2 pullback-breakout RSI=54 vol=1.7x ATR=2.94 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 101.53 | 93.16 | 99.27 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-02-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 05:30:00 | 109.16 | 95.30 | 101.31 | Stage2 pullback-breakout RSI=63 vol=3.4x ATR=4.37 |
| Stop hit — per-position SL triggered | 2026-02-05 05:30:00 | 102.60 | 95.37 | 101.47 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-27 05:30:00 | 101.32 | 2025-07-11 05:30:00 | 99.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest1 | 2025-08-21 05:30:00 | 97.58 | 2025-08-26 05:30:00 | 93.28 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest1 | 2025-09-08 05:30:00 | 92.27 | 2025-09-22 05:30:00 | 93.20 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest1 | 2025-12-12 05:30:00 | 99.78 | 2025-12-29 05:30:00 | 101.53 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest1 | 2026-02-04 05:30:00 | 109.16 | 2026-02-05 05:30:00 | 102.60 | STOP_HIT | 1.00 | -6.01% |
