# Torrent Pharmaceuticals Ltd. (TORNTPHARM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 4380.80
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.85% / -0.95%
- **Sum % (uncompounded):** 4.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.85% | 4.2% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.85% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.85% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 05:30:00 | 3629.20 | 3389.67 | 3586.42 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=62.38 |
| Stop hit — per-position SL triggered | 2025-09-25 05:30:00 | 3535.64 | 3398.43 | 3593.60 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-12-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 05:30:00 | 3795.70 | 3495.69 | 3725.39 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=65.28 |
| Stop hit — per-position SL triggered | 2025-12-18 05:30:00 | 3759.50 | 3522.47 | 3757.40 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2026-01-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 05:30:00 | 4086.60 | 3593.34 | 3927.30 | Stage2 pullback-breakout RSI=70 vol=2.0x ATR=74.50 |
| Stop hit — per-position SL triggered | 2026-01-20 05:30:00 | 3974.84 | 3597.16 | 3932.01 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 4045.60 | 3633.72 | 3965.24 | Stage2 pullback-breakout RSI=59 vol=1.5x ATR=102.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 05:30:00 | 4249.73 | 3670.08 | 4029.14 | T1 booked 50% @ 4249.73 |
| Target hit | 2026-03-16 05:30:00 | 4266.60 | 3786.86 | 4310.66 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-19 05:30:00 | 3629.20 | 2025-09-25 05:30:00 | 3535.64 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest1 | 2025-12-04 05:30:00 | 3795.70 | 2025-12-18 05:30:00 | 3759.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2026-01-19 05:30:00 | 4086.60 | 2026-01-20 05:30:00 | 3974.84 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest1 | 2026-02-03 05:30:00 | 4045.60 | 2026-02-16 05:30:00 | 4249.73 | PARTIAL | 0.50 | 5.05% |
| BUY | retest1 | 2026-02-03 05:30:00 | 4045.60 | 2026-03-16 05:30:00 | 4266.60 | TARGET_HIT | 0.50 | 5.46% |
