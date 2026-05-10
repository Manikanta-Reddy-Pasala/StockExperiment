# Data Patterns (India) Ltd. (DATAPATTNS)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 4103.40
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 1.88% / 0.00%
- **Sum % (uncompounded):** 9.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.88% | 9.4% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.88% | 9.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 3 | 2 | 1.88% | 9.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 05:30:00 | 2711.00 | 2501.28 | 2561.16 | Stage2 pullback-breakout RSI=61 vol=2.9x ATR=102.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 05:30:00 | 2916.53 | 2509.63 | 2620.07 | T1 booked 50% @ 2916.53 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 2711.00 | 2527.51 | 2692.24 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2025-11-13 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 05:30:00 | 3003.40 | 2586.93 | 2755.15 | Stage2 pullback-breakout RSI=68 vol=10.9x ATR=102.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 05:30:00 | 3209.01 | 2611.57 | 2887.67 | T1 booked 50% @ 3209.01 |
| Stop hit — per-position SL triggered | 2025-11-24 05:30:00 | 3003.40 | 2618.95 | 2904.90 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2026-01-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 05:30:00 | 2731.30 | 2637.36 | 2660.19 | Stage2 pullback-breakout RSI=54 vol=1.8x ATR=91.22 |
| Stop hit — per-position SL triggered | 2026-01-12 05:30:00 | 2594.46 | 2638.68 | 2660.65 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-12 05:30:00 | 2711.00 | 2025-09-17 05:30:00 | 2916.53 | PARTIAL | 0.50 | 7.58% |
| BUY | retest1 | 2025-09-12 05:30:00 | 2711.00 | 2025-09-26 05:30:00 | 2711.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 05:30:00 | 3003.40 | 2025-11-20 05:30:00 | 3209.01 | PARTIAL | 0.50 | 6.85% |
| BUY | retest1 | 2025-11-13 05:30:00 | 3003.40 | 2025-11-24 05:30:00 | 3003.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-05 05:30:00 | 2731.30 | 2026-01-12 05:30:00 | 2594.46 | STOP_HIT | 1.00 | -5.01% |
