# Waaree Energies Ltd. (WAAREEENER)

## Backtest Summary

- **Window:** 2024-10-28 09:15:00 → 2026-05-11 15:15:00 (2643 bars)
- **Last close:** 3212.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 0
- **Avg / median % per leg:** 2.21% / -1.08%
- **Sum % (uncompounded):** 6.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 1 | 2 | 0 | 2.21% | 6.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 1 | 33.3% | 1 | 2 | 0 | 2.21% | 6.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 1 | 33.3% | 1 | 2 | 0 | 2.21% | 6.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 3164.40 | 3322.93 | 3323.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 11:15:00 | 3154.20 | 3321.26 | 3322.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 3076.00 | 3074.49 | 3168.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 3076.00 | 3074.49 | 3168.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 3104.00 | 2749.33 | 2881.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 3078.10 | 2749.33 | 2881.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2945.90 | 2942.77 | 2954.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 2961.60 | 2942.77 | 2954.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2934.90 | 2942.69 | 2954.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:45:00 | 2948.00 | 2942.69 | 2954.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2964.10 | 2942.81 | 2954.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 2964.10 | 2942.81 | 2954.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 2964.00 | 2943.02 | 2954.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 2948.70 | 2943.02 | 2954.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 2952.80 | 2935.50 | 2949.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:15:00 | 2945.60 | 2935.50 | 2949.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 2964.00 | 2935.78 | 2949.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:45:00 | 2958.80 | 2935.78 | 2949.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 2986.90 | 2936.29 | 2949.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:45:00 | 2978.40 | 2936.29 | 2949.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2918.50 | 2783.12 | 2849.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 2918.50 | 2783.12 | 2849.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 2920.00 | 2784.48 | 2850.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 2960.00 | 2784.48 | 2850.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 3178.00 | 2901.51 | 2901.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 3195.00 | 2978.09 | 2944.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 3143.00 | 3248.50 | 3123.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 3143.00 | 3248.50 | 3123.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 3133.40 | 3247.35 | 3123.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 3140.00 | 3247.35 | 3123.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 3157.80 | 3246.46 | 3123.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 3190.90 | 3232.58 | 3124.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-18 12:15:00 | 2722.40 | 2025-06-18 15:15:00 | 2692.90 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-19 09:15:00 | 2757.00 | 2025-06-19 10:15:00 | 2694.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-06-20 09:30:00 | 2764.50 | 2025-06-24 10:15:00 | 3040.95 | TARGET_HIT | 1.00 | 10.00% |
