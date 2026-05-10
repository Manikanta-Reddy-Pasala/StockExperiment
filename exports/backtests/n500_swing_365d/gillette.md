# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 8148.00
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
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -4.47% / -4.50%
- **Sum % (uncompounded):** -13.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.47% | -13.4% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.47% | -13.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.47% | -13.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 05:30:00 | 10996.00 | 9160.56 | 10536.90 | Stage2 pullback-breakout RSI=66 vol=4.2x ATR=356.44 |
| Stop hit — per-position SL triggered | 2025-07-29 05:30:00 | 10461.35 | 9327.71 | 10743.56 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2025-09-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 05:30:00 | 10548.00 | 9562.31 | 10333.25 | Stage2 pullback-breakout RSI=55 vol=4.4x ATR=316.56 |
| Stop hit — per-position SL triggered | 2025-09-10 05:30:00 | 10073.16 | 9591.19 | 10317.90 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2025-09-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 05:30:00 | 10352.00 | 9620.87 | 10179.06 | Stage2 pullback-breakout RSI=54 vol=4.3x ATR=278.65 |
| Stop hit — per-position SL triggered | 2025-09-24 05:30:00 | 9934.03 | 9632.04 | 10129.95 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-15 05:30:00 | 10996.00 | 2025-07-29 05:30:00 | 10461.35 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest1 | 2025-09-04 05:30:00 | 10548.00 | 2025-09-10 05:30:00 | 10073.16 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest1 | 2025-09-19 05:30:00 | 10352.00 | 2025-09-24 05:30:00 | 9934.03 | STOP_HIT | 1.00 | -4.04% |
