# Great Eastern Shipping Co. Ltd. (GESHIP)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1588.30
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 0.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.20% | 0.8% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.20% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.20% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 05:30:00 | 1099.60 | 1009.64 | 1047.19 | Stage2 pullback-breakout RSI=69 vol=2.6x ATR=30.71 |
| Stop hit — per-position SL triggered | 2025-11-04 05:30:00 | 1053.54 | 1010.82 | 1051.01 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2025-12-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 05:30:00 | 1109.40 | 1032.26 | 1094.35 | Stage2 pullback-breakout RSI=55 vol=3.9x ATR=30.06 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 1102.60 | 1039.46 | 1102.17 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 05:30:00 | 1173.60 | 1052.44 | 1112.95 | Stage2 pullback-breakout RSI=66 vol=2.4x ATR=32.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 05:30:00 | 1239.39 | 1055.25 | 1127.82 | T1 booked 50% @ 1239.39 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 1173.60 | 1056.62 | 1134.02 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2026-04-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 05:30:00 | 1554.60 | 1201.03 | 1429.92 | Stage2 pullback-breakout RSI=67 vol=4.2x ATR=58.17 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-31 05:30:00 | 1099.60 | 2025-11-04 05:30:00 | 1053.54 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest1 | 2025-12-12 05:30:00 | 1109.40 | 2025-12-29 05:30:00 | 1102.60 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2026-01-28 05:30:00 | 1173.60 | 2026-01-30 05:30:00 | 1239.39 | PARTIAL | 0.50 | 5.61% |
| BUY | retest1 | 2026-01-28 05:30:00 | 1173.60 | 2026-02-01 05:30:00 | 1173.60 | STOP_HIT | 0.50 | 0.00% |
