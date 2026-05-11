# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 956.15
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** 1.80% / 1.62%
- **Sum % (uncompounded):** 10.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.80% | 10.8% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.80% | 10.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.80% | 10.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 05:30:00 | 534.65 | 380.04 | 502.40 | Stage2 pullback-breakout RSI=59 vol=7.3x ATR=30.05 |
| Stop hit — per-position SL triggered | 2024-08-23 05:30:00 | 548.65 | 396.31 | 532.97 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-09-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 05:30:00 | 562.45 | 418.13 | 510.53 | Stage2 pullback-breakout RSI=62 vol=11.1x ATR=25.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 05:30:00 | 613.16 | 419.60 | 515.81 | T1 booked 50% @ 613.16 |
| Stop hit — per-position SL triggered | 2024-09-24 05:30:00 | 562.45 | 421.01 | 520.07 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-12-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 05:30:00 | 530.40 | 475.66 | 519.18 | Stage2 pullback-breakout RSI=56 vol=1.5x ATR=15.02 |
| Stop hit — per-position SL triggered | 2025-01-09 05:30:00 | 525.00 | 481.05 | 526.73 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2025-01-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 05:30:00 | 532.00 | 485.91 | 520.45 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=13.23 |
| Stop hit — per-position SL triggered | 2025-02-11 05:30:00 | 524.35 | 490.53 | 529.04 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-03-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 05:30:00 | 546.40 | 497.26 | 531.07 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=11.02 |
| Stop hit — per-position SL triggered | 2025-03-26 05:30:00 | 555.25 | 502.65 | 545.78 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-08 05:30:00 | 534.65 | 2024-08-23 05:30:00 | 548.65 | STOP_HIT | 1.00 | 2.62% |
| BUY | retest1 | 2024-09-20 05:30:00 | 562.45 | 2024-09-23 05:30:00 | 613.16 | PARTIAL | 0.50 | 9.02% |
| BUY | retest1 | 2024-09-20 05:30:00 | 562.45 | 2024-09-24 05:30:00 | 562.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-26 05:30:00 | 530.40 | 2025-01-09 05:30:00 | 525.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest1 | 2025-01-29 05:30:00 | 532.00 | 2025-02-11 05:30:00 | 524.35 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-03-11 05:30:00 | 546.40 | 2025-03-26 05:30:00 | 555.25 | STOP_HIT | 1.00 | 1.62% |
