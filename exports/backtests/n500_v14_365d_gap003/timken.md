# Timken India Ltd. (TIMKEN)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 3600.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 0
- **Avg / median % per leg:** -3.22% / -3.34%
- **Sum % (uncompounded):** -19.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.22% | -19.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.64% | -13.9% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.81% | -5.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.64% | -13.9% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.81% | -5.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 3126.90 | 2692.85 | 2691.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 3142.70 | 2697.33 | 2693.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 3365.00 | 3366.19 | 3235.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:15:00 | 3422.50 | 3366.19 | 3235.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:15:00 | 3402.10 | 3370.62 | 3247.15 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 14:00:00 | 3400.00 | 3371.35 | 3249.96 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.81 | SL hit (close<ema400) qty=1.00 sl=3253.81 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.81 | SL hit (close<ema400) qty=1.00 sl=3253.81 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.81 | SL hit (close<ema400) qty=1.00 sl=3253.81 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-01 10:15:00 | 3248.90 | 3367.26 | 3253.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 3250.00 | 3366.09 | 3253.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:15:00 | 3252.10 | 3366.09 | 3253.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:00:00 | 3253.30 | 3363.83 | 3253.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 3218.70 | 3362.38 | 3253.59 | SL hit (close<static) qty=1.00 sl=3240.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 3218.70 | 3362.38 | 3253.59 | SL hit (close<static) qty=1.00 sl=3240.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:45:00 | 3251.60 | 3362.38 | 3253.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 3143.00 | 3360.20 | 3253.04 | SL hit (close<static) qty=1.00 sl=3240.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-13 13:15:00 | 2881.70 | 3173.78 | 3174.09 | min_gap filter: gap=0.011% < 0.030% |
| TREND_RESET | 2025-08-13 13:15:00 | 2881.70 | 3173.78 | 3174.09 | EMA inversion without crossover edge (EMA200=3173.78 EMA400=3174.09) — end cycle |
| CROSSOVER_SKIP | 2025-11-18 12:15:00 | 3100.00 | 3029.03 | 3028.94 | min_gap filter: gap=0.003% < 0.030% |
| CROSSOVER_SKIP | 2025-12-30 12:15:00 | 2977.30 | 3050.30 | 3050.37 | min_gap filter: gap=0.002% < 0.030% |
| CROSSOVER_SKIP | 2026-02-04 13:15:00 | 3252.90 | 3037.02 | 3036.78 | min_gap filter: gap=0.007% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-28 09:15:00 | 3422.50 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest1 | 2025-07-30 10:15:00 | 3402.10 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest1 | 2025-07-30 14:00:00 | 3400.00 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-08-01 11:15:00 | 3252.10 | 2025-08-01 13:15:00 | 3218.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-01 13:00:00 | 3253.30 | 2025-08-01 13:15:00 | 3218.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-08-01 13:45:00 | 3251.60 | 2025-08-01 14:15:00 | 3143.00 | STOP_HIT | 1.00 | -3.34% |
