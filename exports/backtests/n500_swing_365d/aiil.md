# Authum Investment & Infrastructure Ltd. (AIIL)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 494.90
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 2.22% / 6.97%
- **Sum % (uncompounded):** 8.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.22% | 8.9% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.22% | 8.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.22% | 8.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 05:30:00 | 550.96 | 396.00 | 521.15 | Stage2 pullback-breakout RSI=63 vol=3.4x ATR=23.28 |
| Stop hit — per-position SL triggered | 2025-07-25 05:30:00 | 516.03 | 408.81 | 544.25 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2025-08-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 05:30:00 | 625.14 | 440.88 | 581.41 | Stage2 pullback-breakout RSI=67 vol=2.6x ATR=26.47 |
| Stop hit — per-position SL triggered | 2025-09-10 05:30:00 | 622.68 | 458.77 | 609.95 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2025-12-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 05:30:00 | 561.66 | 516.42 | 536.16 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=19.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 05:30:00 | 600.83 | 521.43 | 560.64 | T1 booked 50% @ 600.83 |
| Target hit | 2026-01-20 05:30:00 | 610.10 | 535.30 | 611.97 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-15 05:30:00 | 550.96 | 2025-07-25 05:30:00 | 516.03 | STOP_HIT | 1.00 | -6.34% |
| BUY | retest1 | 2025-08-26 05:30:00 | 625.14 | 2025-09-10 05:30:00 | 622.68 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-12-16 05:30:00 | 561.66 | 2025-12-30 05:30:00 | 600.83 | PARTIAL | 0.50 | 6.97% |
| BUY | retest1 | 2025-12-16 05:30:00 | 561.66 | 2026-01-20 05:30:00 | 610.10 | TARGET_HIT | 0.50 | 8.62% |
