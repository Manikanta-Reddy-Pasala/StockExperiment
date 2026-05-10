# State Bank of India (SBIN)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1019.30
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** 1.05% / -1.40%
- **Sum % (uncompounded):** 6.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.05% | 6.3% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.05% | 6.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.05% | 6.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 05:30:00 | 820.35 | 789.98 | 800.65 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=12.52 |
| Stop hit — per-position SL triggered | 2025-07-14 05:30:00 | 808.85 | 791.96 | 806.72 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-07-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 05:30:00 | 831.70 | 792.59 | 809.94 | Stage2 pullback-breakout RSI=66 vol=1.6x ATR=11.14 |
| Stop hit — per-position SL triggered | 2025-07-22 05:30:00 | 814.99 | 793.78 | 814.04 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2025-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 05:30:00 | 831.55 | 799.80 | 816.34 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=9.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 05:30:00 | 849.84 | 800.37 | 820.22 | T1 booked 50% @ 849.84 |
| Target hit | 2025-12-03 05:30:00 | 951.05 | 851.43 | 962.09 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-01-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 05:30:00 | 1066.20 | 898.50 | 1026.61 | Stage2 pullback-breakout RSI=70 vol=1.6x ATR=17.18 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 1040.43 | 901.45 | 1030.16 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2026-04-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 05:30:00 | 1107.85 | 981.44 | 1066.12 | Stage2 pullback-breakout RSI=58 vol=1.6x ATR=32.95 |
| Stop hit — per-position SL triggered | 2026-05-05 05:30:00 | 1058.42 | 991.70 | 1078.14 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-30 05:30:00 | 820.35 | 2025-07-14 05:30:00 | 808.85 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest1 | 2025-07-16 05:30:00 | 831.70 | 2025-07-22 05:30:00 | 814.99 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest1 | 2025-09-16 05:30:00 | 831.55 | 2025-09-17 05:30:00 | 849.84 | PARTIAL | 0.50 | 2.20% |
| BUY | retest1 | 2025-09-16 05:30:00 | 831.55 | 2025-12-03 05:30:00 | 951.05 | TARGET_HIT | 0.50 | 14.37% |
| BUY | retest1 | 2026-01-29 05:30:00 | 1066.20 | 2026-02-01 05:30:00 | 1040.43 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest1 | 2026-04-20 05:30:00 | 1107.85 | 2026-05-05 05:30:00 | 1058.42 | STOP_HIT | 1.00 | -4.46% |
