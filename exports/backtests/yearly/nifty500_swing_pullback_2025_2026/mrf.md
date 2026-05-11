# MRF Ltd. (MRF)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 130425.00
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
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.95% / 1.88%
- **Sum % (uncompounded):** 6.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.95% | 6.6% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.95% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 0.95% | 6.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 05:30:00 | 143215.00 | 128011.97 | 138546.44 | Stage2 pullback-breakout RSI=63 vol=2.3x ATR=2997.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 05:30:00 | 149210.83 | 129299.52 | 142174.94 | T1 booked 50% @ 149210.83 |
| Target hit | 2025-08-01 05:30:00 | 145905.00 | 132391.79 | 147612.25 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-08-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 05:30:00 | 146535.00 | 133508.79 | 144494.37 | Stage2 pullback-breakout RSI=54 vol=2.9x ATR=3413.58 |
| Stop hit — per-position SL triggered | 2025-08-28 05:30:00 | 141414.62 | 134232.02 | 144968.37 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2025-09-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 05:30:00 | 153885.00 | 134597.60 | 145493.19 | Stage2 pullback-breakout RSI=64 vol=4.2x ATR=3829.56 |
| Stop hit — per-position SL triggered | 2025-09-05 05:30:00 | 148140.66 | 135035.87 | 146494.31 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2025-10-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 05:30:00 | 155210.00 | 137996.90 | 150409.60 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=4008.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 05:30:00 | 163226.48 | 139445.41 | 154064.65 | T1 booked 50% @ 163226.48 |
| Stop hit — per-position SL triggered | 2025-10-27 05:30:00 | 159335.00 | 140285.24 | 156273.21 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2026-02-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 05:30:00 | 146455.00 | 144493.40 | 139922.61 | Stage2 pullback-breakout RSI=58 vol=6.8x ATR=4143.04 |
| Stop hit — per-position SL triggered | 2026-02-20 05:30:00 | 146390.00 | 144842.13 | 144980.71 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-27 05:30:00 | 143215.00 | 2025-07-09 05:30:00 | 149210.83 | PARTIAL | 0.50 | 4.19% |
| BUY | retest1 | 2025-06-27 05:30:00 | 143215.00 | 2025-08-01 05:30:00 | 145905.00 | TARGET_HIT | 0.50 | 1.88% |
| BUY | retest1 | 2025-08-19 05:30:00 | 146535.00 | 2025-08-28 05:30:00 | 141414.62 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest1 | 2025-09-02 05:30:00 | 153885.00 | 2025-09-05 05:30:00 | 148140.66 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest1 | 2025-10-08 05:30:00 | 155210.00 | 2025-10-20 05:30:00 | 163226.48 | PARTIAL | 0.50 | 5.16% |
| BUY | retest1 | 2025-10-08 05:30:00 | 155210.00 | 2025-10-27 05:30:00 | 159335.00 | STOP_HIT | 0.50 | 2.66% |
| BUY | retest1 | 2026-02-06 05:30:00 | 146455.00 | 2026-02-20 05:30:00 | 146390.00 | STOP_HIT | 1.00 | -0.04% |
