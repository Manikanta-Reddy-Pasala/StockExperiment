# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 83.90
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 0.97% / 1.30%
- **Sum % (uncompounded):** 7.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.97% | 7.8% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.97% | 7.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.97% | 7.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 05:30:00 | 54.62 | 53.85 | 53.67 | Stage2 pullback-breakout RSI=54 vol=1.9x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 05:30:00 | 56.89 | 53.90 | 54.25 | T1 booked 50% @ 56.89 |
| Target hit | 2025-09-25 05:30:00 | 55.33 | 54.07 | 55.34 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 05:30:00 | 59.30 | 54.37 | 56.41 | Stage2 pullback-breakout RSI=63 vol=5.8x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-10-17 05:30:00 | 56.81 | 54.43 | 56.58 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-11-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 05:30:00 | 59.91 | 55.18 | 58.25 | Stage2 pullback-breakout RSI=57 vol=1.6x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-12-01 05:30:00 | 57.65 | 55.58 | 58.82 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-12-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 05:30:00 | 60.51 | 55.88 | 57.71 | Stage2 pullback-breakout RSI=65 vol=1.7x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 05:30:00 | 63.19 | 56.02 | 58.65 | T1 booked 50% @ 63.19 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 60.51 | 57.56 | 63.73 | SL hit (bars_held=22) |

### Cycle 5 — BUY (started 2026-02-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 05:30:00 | 68.99 | 58.55 | 65.47 | Stage2 pullback-breakout RSI=66 vol=1.9x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 05:30:00 | 73.04 | 59.13 | 67.56 | T1 booked 50% @ 73.04 |
| Stop hit — per-position SL triggered | 2026-03-04 05:30:00 | 68.99 | 59.66 | 69.15 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-11 05:30:00 | 54.62 | 2025-09-17 05:30:00 | 56.89 | PARTIAL | 0.50 | 4.16% |
| BUY | retest1 | 2025-09-11 05:30:00 | 54.62 | 2025-09-25 05:30:00 | 55.33 | TARGET_HIT | 0.50 | 1.30% |
| BUY | retest1 | 2025-10-15 05:30:00 | 59.30 | 2025-10-17 05:30:00 | 56.81 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest1 | 2025-11-17 05:30:00 | 59.91 | 2025-12-01 05:30:00 | 57.65 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest1 | 2025-12-30 05:30:00 | 60.51 | 2026-01-01 05:30:00 | 63.19 | PARTIAL | 0.50 | 4.43% |
| BUY | retest1 | 2025-12-30 05:30:00 | 60.51 | 2026-02-01 05:30:00 | 60.51 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 05:30:00 | 68.99 | 2026-02-25 05:30:00 | 73.04 | PARTIAL | 0.50 | 5.87% |
| BUY | retest1 | 2026-02-18 05:30:00 | 68.99 | 2026-03-04 05:30:00 | 68.99 | STOP_HIT | 0.50 | 0.00% |
