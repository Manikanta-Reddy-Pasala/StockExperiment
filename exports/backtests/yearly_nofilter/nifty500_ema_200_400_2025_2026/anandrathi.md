# Anand Rathi Wealth Ltd. (ANANDRATHI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3602.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 1 / 14 / 0
- **Avg / median % per leg:** -1.16% / -1.08%
- **Sum % (uncompounded):** -17.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 9 | 0 | -1.34% | -13.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 4 | 40.0% | 1 | 9 | 0 | -1.34% | -13.4% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 5 | 0 | -0.79% | -3.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 5 | 0 | -0.79% | -3.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 6 | 40.0% | 1 | 14 | 0 | -1.16% | -17.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 1880.00 | 1813.78 | 1813.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1892.90 | 1814.57 | 1814.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 13:15:00 | 2876.80 | 2893.88 | 2709.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:45:00 | 2865.60 | 2893.88 | 2709.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 2945.00 | 3052.15 | 2955.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 2945.00 | 3052.15 | 2955.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2948.90 | 3051.12 | 2955.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 2960.90 | 3051.12 | 2955.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 2928.90 | 3047.60 | 2955.00 | SL hit (close<static) qty=1.00 sl=2932.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 2956.40 | 2986.92 | 2987.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 15:15:00 | 2942.00 | 2986.48 | 2986.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 2977.80 | 2977.71 | 2982.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 2977.80 | 2977.71 | 2982.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 2977.80 | 2977.71 | 2982.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 2977.80 | 2977.71 | 2982.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 2994.00 | 2977.87 | 2982.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 2994.00 | 2977.87 | 2982.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 3033.00 | 2978.42 | 2982.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 3033.00 | 2978.42 | 2982.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 2991.00 | 2979.45 | 2982.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:45:00 | 2990.20 | 2979.45 | 2982.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 2984.20 | 2979.50 | 2982.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 2969.50 | 2979.59 | 2982.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 15:15:00 | 2991.60 | 2979.71 | 2982.80 | SL hit (close>static) qty=1.00 sl=2987.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 3033.90 | 2985.76 | 2985.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 3055.00 | 2990.67 | 2988.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 14:15:00 | 3064.00 | 3066.80 | 3034.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 15:00:00 | 3064.00 | 3066.80 | 3034.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3011.40 | 3066.13 | 3034.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 3078.90 | 3063.61 | 3035.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:30:00 | 3081.60 | 3064.14 | 3036.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 2914.00 | 3060.29 | 3036.06 | SL hit (close<static) qty=1.00 sl=2992.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 09:30:00 | 1892.80 | 2025-05-27 09:15:00 | 1945.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-05-28 13:45:00 | 1896.70 | 2025-05-29 15:15:00 | 1880.00 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-05-29 09:15:00 | 1893.40 | 2025-05-29 15:15:00 | 1880.00 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-11-19 09:15:00 | 2960.90 | 2025-11-19 11:15:00 | 2928.90 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-15 09:15:00 | 2962.40 | 2026-01-22 15:15:00 | 3005.00 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-12-15 10:45:00 | 2961.40 | 2026-01-22 15:15:00 | 3005.00 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-12-15 14:00:00 | 2955.70 | 2026-01-22 15:15:00 | 3005.00 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2026-01-21 13:15:00 | 3038.20 | 2026-01-27 09:15:00 | 2876.50 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest2 | 2026-01-21 14:30:00 | 3035.20 | 2026-01-27 09:15:00 | 2876.50 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest2 | 2026-01-21 15:00:00 | 3045.10 | 2026-01-27 09:15:00 | 2876.50 | STOP_HIT | 1.00 | -5.54% |
| SELL | retest2 | 2026-02-12 14:30:00 | 2969.50 | 2026-02-12 15:15:00 | 2991.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-02-13 09:15:00 | 2930.10 | 2026-02-13 12:15:00 | 2989.70 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-03-17 15:15:00 | 3078.90 | 2026-03-20 09:15:00 | 2914.00 | STOP_HIT | 1.00 | -5.36% |
| BUY | retest2 | 2026-03-18 10:30:00 | 3081.60 | 2026-03-20 09:15:00 | 2914.00 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2026-04-01 09:15:00 | 3101.20 | 2026-04-09 09:15:00 | 3411.32 | TARGET_HIT | 1.00 | 10.00% |
