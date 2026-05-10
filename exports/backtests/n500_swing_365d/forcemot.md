# Force Motors Ltd. (FORCEMOT)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 20877.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 1.76% / 5.11%
- **Sum % (uncompounded):** 12.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 1.76% | 12.4% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 1.76% | 12.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 2 | 3 | 2 | 1.76% | 12.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 05:30:00 | 19734.00 | 12917.78 | 18769.48 | Stage2 pullback-breakout RSI=57 vol=1.5x ATR=970.69 |
| Stop hit — per-position SL triggered | 2025-09-19 05:30:00 | 18277.96 | 13101.15 | 18851.31 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2025-10-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 05:30:00 | 17565.00 | 13739.39 | 16958.94 | Stage2 pullback-breakout RSI=54 vol=2.6x ATR=750.20 |
| Stop hit — per-position SL triggered | 2025-10-23 05:30:00 | 16439.71 | 13843.32 | 17029.13 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 17801.00 | 15003.75 | 17335.45 | Stage2 pullback-breakout RSI=57 vol=2.5x ATR=581.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 05:30:00 | 18963.82 | 15175.41 | 17811.98 | T1 booked 50% @ 18963.82 |
| Target hit | 2026-01-09 05:30:00 | 18710.00 | 15626.42 | 19305.95 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 05:30:00 | 21090.00 | 16321.58 | 19702.90 | Stage2 pullback-breakout RSI=61 vol=5.3x ATR=1016.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 05:30:00 | 23123.16 | 16503.16 | 20436.81 | T1 booked 50% @ 23123.16 |
| Target hit | 2026-03-02 05:30:00 | 23380.00 | 17564.17 | 23575.63 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 22075.00 | 18282.61 | 21230.23 | Stage2 pullback-breakout RSI=53 vol=3.2x ATR=1338.67 |
| Stop hit — per-position SL triggered | 2026-04-23 05:30:00 | 20750.00 | 18639.80 | 21674.17 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-16 05:30:00 | 19734.00 | 2025-09-19 05:30:00 | 18277.96 | STOP_HIT | 1.00 | -7.38% |
| BUY | retest1 | 2025-10-17 05:30:00 | 17565.00 | 2025-10-23 05:30:00 | 16439.71 | STOP_HIT | 1.00 | -6.41% |
| BUY | retest1 | 2025-12-19 05:30:00 | 17801.00 | 2025-12-29 05:30:00 | 18963.82 | PARTIAL | 0.50 | 6.53% |
| BUY | retest1 | 2025-12-19 05:30:00 | 17801.00 | 2026-01-09 05:30:00 | 18710.00 | TARGET_HIT | 0.50 | 5.11% |
| BUY | retest1 | 2026-02-05 05:30:00 | 21090.00 | 2026-02-10 05:30:00 | 23123.16 | PARTIAL | 0.50 | 9.64% |
| BUY | retest1 | 2026-02-05 05:30:00 | 21090.00 | 2026-03-02 05:30:00 | 23380.00 | TARGET_HIT | 0.50 | 10.86% |
| BUY | retest1 | 2026-04-08 05:30:00 | 22075.00 | 2026-04-23 05:30:00 | 20750.00 | STOP_HIT | 1.00 | -6.00% |
