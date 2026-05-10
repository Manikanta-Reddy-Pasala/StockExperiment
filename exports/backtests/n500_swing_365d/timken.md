# Timken India Ltd. (TIMKEN)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 3595.00
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -4.84% / -5.23%
- **Sum % (uncompounded):** -14.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.84% | -14.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.84% | -14.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.84% | -14.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-23 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 05:30:00 | 3533.70 | 3145.24 | 3398.22 | Stage2 pullback-breakout RSI=68 vol=2.3x ATR=88.81 |
| Stop hit — per-position SL triggered | 2025-07-25 05:30:00 | 3400.49 | 3151.30 | 3407.34 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2026-03-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 05:30:00 | 3410.30 | 3105.93 | 3275.85 | Stage2 pullback-breakout RSI=60 vol=2.3x ATR=118.92 |
| Stop hit — per-position SL triggered | 2026-03-19 05:30:00 | 3231.92 | 3121.96 | 3317.28 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2026-04-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 05:30:00 | 3633.00 | 3170.24 | 3433.70 | Stage2 pullback-breakout RSI=65 vol=2.3x ATR=133.45 |
| Stop hit — per-position SL triggered | 2026-04-29 05:30:00 | 3432.83 | 3187.76 | 3469.18 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2026-05-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 05:30:00 | 3595.00 | 3204.21 | 3472.61 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=121.29 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-23 05:30:00 | 3533.70 | 2025-07-25 05:30:00 | 3400.49 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest1 | 2026-03-11 05:30:00 | 3410.30 | 2026-03-19 05:30:00 | 3231.92 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest1 | 2026-04-22 05:30:00 | 3633.00 | 2026-04-29 05:30:00 | 3432.83 | STOP_HIT | 1.00 | -5.51% |
