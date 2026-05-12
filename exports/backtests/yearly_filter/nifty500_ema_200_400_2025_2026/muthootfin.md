# Muthoot Finance Ltd. (MUTHOOTFIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3535.00
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
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 0
- **Avg / median % per leg:** -0.92% / -1.26%
- **Sum % (uncompounded):** -6.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 1 | 2 | 0 | -0.48% | -1.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 1 | 33.3% | 1 | 2 | 0 | -0.48% | -1.5% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.25% | -5.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.25% | -5.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 1 | 14.3% | 1 | 6 | 0 | -0.92% | -6.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 2090.40 | 2203.03 | 2203.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 15:15:00 | 2084.00 | 2196.36 | 2199.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 2150.20 | 2145.97 | 2170.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 2150.20 | 2145.97 | 2170.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 2204.80 | 2146.55 | 2170.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 2204.80 | 2146.55 | 2170.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 2221.00 | 2147.29 | 2170.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 2221.00 | 2147.29 | 2170.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 2454.80 | 2189.47 | 2188.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 2508.50 | 2192.64 | 2190.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 10:15:00 | 2597.40 | 2606.95 | 2508.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 2597.40 | 2606.95 | 2508.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 2546.30 | 2611.49 | 2540.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 2545.70 | 2611.49 | 2540.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 2539.50 | 2610.77 | 2540.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 2539.50 | 2610.77 | 2540.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 2559.70 | 2610.26 | 2540.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 2760.80 | 2601.24 | 2539.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 14:15:00 | 3036.88 | 2820.89 | 2717.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 3473.80 | 3671.04 | 3671.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 3444.40 | 3647.05 | 3659.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3447.20 | 3329.04 | 3433.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3447.20 | 3329.04 | 3433.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3447.20 | 3329.04 | 3433.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 3447.20 | 3329.04 | 3433.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 3461.60 | 3330.36 | 3433.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:30:00 | 3465.00 | 3330.36 | 3433.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 3554.70 | 3365.34 | 3441.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:30:00 | 3558.80 | 3365.34 | 3441.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 3466.60 | 3455.71 | 3474.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:30:00 | 3465.20 | 3455.71 | 3474.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 3471.90 | 3455.87 | 3474.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 3483.10 | 3455.87 | 3474.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 3492.00 | 3456.23 | 3474.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 3492.00 | 3456.23 | 3474.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 3495.00 | 3456.62 | 3474.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 3499.80 | 3456.62 | 3474.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 3497.00 | 3460.84 | 3475.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 3490.20 | 3460.84 | 3475.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 3488.50 | 3462.75 | 3476.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 3489.70 | 3463.02 | 3476.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 09:45:00 | 3492.10 | 3460.95 | 3474.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 3478.60 | 3461.13 | 3474.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 3485.00 | 3461.13 | 3474.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 3497.10 | 3461.49 | 3474.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 3500.00 | 3461.49 | 3474.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 3480.60 | 3461.79 | 3474.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:00:00 | 3480.60 | 3461.79 | 3474.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 3491.00 | 3462.08 | 3474.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:15:00 | 3496.00 | 3462.08 | 3474.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3467.50 | 3460.80 | 3473.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 3533.60 | 3462.35 | 3473.89 | SL hit (close>static) qty=1.00 sl=3529.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 2265.00 | 2025-05-15 10:15:00 | 2131.00 | STOP_HIT | 1.00 | -5.92% |
| BUY | retest2 | 2025-05-14 09:15:00 | 2255.90 | 2025-05-15 10:15:00 | 2131.00 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2025-08-14 09:15:00 | 2760.80 | 2025-09-22 14:15:00 | 3036.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 11:15:00 | 3490.20 | 2026-05-06 14:15:00 | 3533.60 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-04-29 09:30:00 | 3488.50 | 2026-05-06 14:15:00 | 3533.60 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-04-29 10:30:00 | 3489.70 | 2026-05-06 14:15:00 | 3533.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-04 09:45:00 | 3492.10 | 2026-05-06 14:15:00 | 3533.60 | STOP_HIT | 1.00 | -1.19% |
