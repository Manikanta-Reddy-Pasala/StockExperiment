# Neuland Laboratories Ltd. (NEULANDLAB)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 17666.00
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 5.75% / 8.31%
- **Sum % (uncompounded):** 22.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 5.75% | 23.0% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 5.75% | 23.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 5.75% | 23.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 05:30:00 | 13693.00 | 12819.27 | 13263.34 | Stage2 pullback-breakout RSI=57 vol=2.4x ATR=575.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 05:30:00 | 14843.10 | 12919.71 | 13673.95 | T1 booked 50% @ 14843.10 |
| Target hit | 2025-09-25 05:30:00 | 14831.00 | 13236.85 | 14929.94 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 17159.00 | 13812.66 | 15996.99 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=538.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 05:30:00 | 18235.54 | 13968.20 | 16591.78 | T1 booked 50% @ 18235.54 |
| Stop hit — per-position SL triggered | 2025-11-11 05:30:00 | 17159.00 | 14000.29 | 16649.04 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-20 05:30:00 | 13693.00 | 2025-09-05 05:30:00 | 14843.10 | PARTIAL | 0.50 | 8.40% |
| BUY | retest1 | 2025-08-20 05:30:00 | 13693.00 | 2025-09-25 05:30:00 | 14831.00 | TARGET_HIT | 0.50 | 8.31% |
| BUY | retest1 | 2025-11-03 05:30:00 | 17159.00 | 2025-11-10 05:30:00 | 18235.54 | PARTIAL | 0.50 | 6.27% |
| BUY | retest1 | 2025-11-03 05:30:00 | 17159.00 | 2025-11-11 05:30:00 | 17159.00 | STOP_HIT | 0.50 | 0.00% |
