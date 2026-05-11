# Godrej Industries Ltd. (GODREJIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1202.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 226 |
| ALERT1 | 126 |
| ALERT2 | 124 |
| ALERT2_SKIP | 63 |
| ALERT3 | 331 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 214 |
| PARTIAL | 20 |
| TARGET_HIT | 10 |
| STOP_HIT | 215 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 245 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 77 / 168
- **Target hits / Stop hits / Partials:** 10 / 215 / 20
- **Avg / median % per leg:** 0.42% / -0.76%
- **Sum % (uncompounded):** 102.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 108 | 33 | 30.6% | 7 | 101 | 0 | 0.16% | 17.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.32% | -9.3% |
| BUY @ 3rd Alert (retest2) | 104 | 33 | 31.7% | 7 | 97 | 0 | 0.26% | 27.1% |
| SELL (all) | 137 | 44 | 32.1% | 3 | 114 | 20 | 0.62% | 84.6% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 7 | 2 | 0.66% | 5.9% |
| SELL @ 3rd Alert (retest2) | 128 | 40 | 31.2% | 3 | 107 | 18 | 0.61% | 78.7% |
| retest1 (combined) | 13 | 4 | 30.8% | 0 | 11 | 2 | -0.26% | -3.4% |
| retest2 (combined) | 232 | 73 | 31.5% | 10 | 204 | 18 | 0.46% | 105.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 10:15:00 | 467.35 | 471.37 | 471.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 463.05 | 469.71 | 470.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 14:15:00 | 467.80 | 467.75 | 469.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-17 15:00:00 | 467.80 | 467.75 | 469.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 468.35 | 467.87 | 469.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:15:00 | 469.50 | 467.87 | 469.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 468.05 | 467.91 | 469.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 12:00:00 | 462.25 | 467.03 | 468.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 13:30:00 | 466.55 | 463.34 | 465.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 09:15:00 | 472.20 | 466.82 | 466.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 09:15:00 | 472.20 | 466.82 | 466.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 10:15:00 | 475.10 | 468.48 | 467.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 10:15:00 | 470.30 | 472.13 | 470.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-23 11:00:00 | 470.30 | 472.13 | 470.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 470.50 | 471.80 | 470.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 12:00:00 | 470.50 | 471.80 | 470.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 470.60 | 471.56 | 470.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 09:45:00 | 471.45 | 471.71 | 470.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 11:00:00 | 472.10 | 471.79 | 470.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 12:15:00 | 466.95 | 470.87 | 470.57 | SL hit (close<static) qty=1.00 sl=469.35 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 13:15:00 | 467.75 | 470.25 | 470.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 15:15:00 | 463.00 | 468.27 | 469.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 09:15:00 | 469.10 | 468.44 | 469.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 469.10 | 468.44 | 469.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 469.10 | 468.44 | 469.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:45:00 | 469.10 | 468.44 | 469.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 468.90 | 468.53 | 469.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:45:00 | 469.65 | 468.53 | 469.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 466.05 | 468.03 | 469.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 12:15:00 | 465.75 | 468.03 | 469.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-25 13:15:00 | 469.50 | 468.18 | 468.90 | SL hit (close>static) qty=1.00 sl=469.45 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 11:15:00 | 480.00 | 471.04 | 469.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 15:15:00 | 485.00 | 478.53 | 474.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 12:15:00 | 479.80 | 480.55 | 476.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 13:00:00 | 479.80 | 480.55 | 476.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 476.30 | 479.24 | 476.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 15:15:00 | 475.45 | 479.24 | 476.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 475.45 | 478.48 | 476.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 09:15:00 | 480.70 | 478.48 | 476.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 479.90 | 479.25 | 477.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:30:00 | 479.35 | 479.25 | 477.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 478.65 | 479.13 | 477.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 09:15:00 | 482.30 | 477.99 | 477.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 11:15:00 | 475.50 | 477.99 | 477.65 | SL hit (close<static) qty=1.00 sl=476.35 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 13:15:00 | 475.10 | 477.13 | 477.31 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 479.80 | 477.67 | 477.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 15:15:00 | 481.75 | 478.48 | 477.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 09:15:00 | 476.95 | 478.18 | 477.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 476.95 | 478.18 | 477.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 476.95 | 478.18 | 477.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:30:00 | 477.50 | 478.18 | 477.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 477.15 | 477.97 | 477.77 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 12:15:00 | 476.15 | 477.48 | 477.57 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 15:15:00 | 480.10 | 478.08 | 477.82 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 12:15:00 | 476.10 | 478.49 | 478.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 13:15:00 | 475.35 | 477.86 | 478.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 09:15:00 | 477.65 | 477.16 | 477.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 09:15:00 | 477.65 | 477.16 | 477.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 477.65 | 477.16 | 477.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:45:00 | 478.50 | 477.16 | 477.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 473.20 | 476.36 | 477.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-06 14:00:00 | 472.40 | 475.24 | 476.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 12:30:00 | 471.95 | 473.40 | 474.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 13:15:00 | 472.35 | 473.40 | 474.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 13:00:00 | 472.55 | 474.07 | 474.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 473.55 | 473.96 | 474.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:45:00 | 475.25 | 473.96 | 474.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 470.90 | 473.35 | 474.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 10:15:00 | 470.45 | 472.79 | 473.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 12:00:00 | 470.50 | 471.66 | 473.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 12:30:00 | 470.40 | 470.48 | 472.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 10:30:00 | 470.00 | 468.96 | 470.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 472.85 | 469.74 | 470.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:45:00 | 473.40 | 469.74 | 470.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 472.70 | 470.33 | 471.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 12:30:00 | 473.80 | 470.33 | 471.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-12 13:15:00 | 475.75 | 471.41 | 471.44 | SL hit (close>static) qty=1.00 sl=475.35 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 14:15:00 | 473.85 | 471.90 | 471.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 14:15:00 | 481.50 | 475.85 | 473.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 13:15:00 | 499.05 | 501.48 | 494.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 14:00:00 | 499.05 | 501.48 | 494.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 499.10 | 500.18 | 495.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 10:00:00 | 501.40 | 500.42 | 496.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 09:45:00 | 502.80 | 503.88 | 500.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 09:15:00 | 498.65 | 501.30 | 501.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 09:15:00 | 498.65 | 501.30 | 501.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 493.40 | 498.89 | 500.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 15:15:00 | 480.50 | 480.04 | 484.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-27 09:15:00 | 486.95 | 480.04 | 484.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 497.55 | 483.54 | 486.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:45:00 | 498.85 | 483.54 | 486.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 496.15 | 486.06 | 486.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:30:00 | 499.45 | 486.06 | 486.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 11:15:00 | 497.85 | 488.42 | 487.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 499.80 | 493.13 | 490.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 12:15:00 | 512.00 | 512.32 | 507.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 13:00:00 | 512.00 | 512.32 | 507.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 507.00 | 511.02 | 507.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 14:45:00 | 506.50 | 511.02 | 507.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 15:15:00 | 505.90 | 509.99 | 507.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 09:15:00 | 508.55 | 509.99 | 507.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 15:00:00 | 508.95 | 509.41 | 508.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 14:15:00 | 506.80 | 509.83 | 510.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 14:15:00 | 506.80 | 509.83 | 510.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 15:15:00 | 506.20 | 509.10 | 509.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 09:15:00 | 502.40 | 502.10 | 504.93 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 11:00:00 | 498.00 | 501.28 | 504.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 499.80 | 495.79 | 499.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-11 09:15:00 | 499.80 | 495.79 | 499.71 | SL hit (close>ema400) qty=1.00 sl=499.71 alert=retest1 |

### Cycle 14 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 485.55 | 482.09 | 481.70 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 11:15:00 | 477.60 | 480.79 | 481.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 12:15:00 | 476.10 | 479.85 | 480.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 481.10 | 477.99 | 479.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 481.10 | 477.99 | 479.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 481.10 | 477.99 | 479.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:00:00 | 481.10 | 477.99 | 479.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 476.45 | 477.68 | 479.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 12:15:00 | 475.55 | 477.35 | 478.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 12:45:00 | 475.20 | 476.89 | 478.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 12:00:00 | 475.90 | 476.82 | 477.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 13:15:00 | 475.25 | 476.67 | 477.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 475.50 | 476.06 | 477.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:15:00 | 477.80 | 476.06 | 477.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 478.00 | 476.45 | 477.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 10:45:00 | 475.70 | 476.36 | 477.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 11:15:00 | 475.00 | 476.36 | 477.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-31 09:15:00 | 482.40 | 476.74 | 476.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 482.40 | 476.74 | 476.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-02 09:15:00 | 483.95 | 480.24 | 478.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 480.00 | 480.19 | 479.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 11:00:00 | 480.00 | 480.19 | 479.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 481.80 | 480.52 | 479.29 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 15:15:00 | 476.00 | 478.49 | 478.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 474.65 | 477.72 | 478.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 476.00 | 475.16 | 476.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 476.00 | 475.16 | 476.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 476.00 | 475.16 | 476.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 10:15:00 | 477.10 | 475.16 | 476.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 476.00 | 475.32 | 476.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 10:30:00 | 476.65 | 475.32 | 476.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 475.50 | 475.36 | 476.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 14:00:00 | 472.35 | 474.77 | 475.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 14:45:00 | 474.05 | 474.96 | 475.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:45:00 | 472.70 | 474.70 | 475.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 09:15:00 | 477.65 | 473.34 | 474.21 | SL hit (close>static) qty=1.00 sl=476.50 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 11:15:00 | 477.90 | 474.78 | 474.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 13:15:00 | 478.95 | 475.94 | 475.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 14:15:00 | 493.85 | 495.18 | 490.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 15:00:00 | 493.85 | 495.18 | 490.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 502.40 | 498.76 | 494.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 508.05 | 495.94 | 494.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 11:30:00 | 508.15 | 501.40 | 497.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 13:00:00 | 508.45 | 502.81 | 498.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 13:45:00 | 508.05 | 503.80 | 499.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 501.90 | 505.73 | 503.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:00:00 | 501.90 | 505.73 | 503.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 506.45 | 505.87 | 503.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:30:00 | 502.85 | 505.87 | 503.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 505.65 | 505.83 | 504.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 12:00:00 | 505.65 | 505.83 | 504.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 508.10 | 506.28 | 504.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-18 15:15:00 | 493.50 | 502.01 | 502.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 15:15:00 | 493.50 | 502.01 | 502.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 15:15:00 | 490.55 | 497.69 | 500.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 10:15:00 | 498.65 | 497.58 | 499.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-22 10:30:00 | 498.35 | 497.58 | 499.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 11:15:00 | 498.80 | 497.82 | 499.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 11:45:00 | 498.70 | 497.82 | 499.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 503.30 | 498.32 | 499.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:00:00 | 503.30 | 498.32 | 499.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 10:15:00 | 506.30 | 499.92 | 499.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 11:15:00 | 512.00 | 502.33 | 500.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 15:15:00 | 533.00 | 536.14 | 525.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-25 09:15:00 | 542.05 | 536.14 | 525.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 527.55 | 533.77 | 531.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-28 14:15:00 | 527.55 | 533.77 | 531.92 | SL hit (close<ema400) qty=1.00 sl=531.92 alert=retest1 |

### Cycle 21 — SELL (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 10:15:00 | 526.75 | 530.85 | 530.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 13:15:00 | 521.80 | 527.76 | 529.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 526.70 | 526.40 | 528.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 526.70 | 526.40 | 528.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 526.70 | 526.40 | 528.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 09:30:00 | 530.00 | 526.40 | 528.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 522.10 | 524.38 | 526.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 14:15:00 | 521.40 | 524.38 | 526.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 09:15:00 | 529.20 | 525.25 | 526.39 | SL hit (close>static) qty=1.00 sl=526.70 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 12:15:00 | 528.70 | 527.33 | 527.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 14:15:00 | 535.25 | 528.89 | 527.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 09:15:00 | 534.20 | 537.48 | 534.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 09:15:00 | 534.20 | 537.48 | 534.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 534.20 | 537.48 | 534.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:45:00 | 535.00 | 537.48 | 534.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 532.30 | 536.45 | 534.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:00:00 | 532.30 | 536.45 | 534.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 11:15:00 | 534.35 | 536.03 | 534.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 12:45:00 | 543.05 | 536.42 | 534.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-11 09:15:00 | 597.36 | 571.25 | 560.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 553.80 | 562.60 | 563.25 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 13:15:00 | 565.60 | 562.06 | 561.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 14:15:00 | 571.90 | 564.03 | 562.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 09:15:00 | 576.25 | 576.77 | 571.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 09:15:00 | 576.25 | 576.77 | 571.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 576.25 | 576.77 | 571.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 10:00:00 | 576.25 | 576.77 | 571.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 573.20 | 579.95 | 575.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 15:00:00 | 573.20 | 579.95 | 575.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 573.50 | 578.66 | 575.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:15:00 | 577.50 | 578.66 | 575.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 575.50 | 578.03 | 575.43 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 13:15:00 | 569.70 | 574.03 | 574.19 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 15:15:00 | 575.00 | 574.34 | 574.31 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 565.00 | 572.47 | 573.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 14:15:00 | 560.20 | 565.76 | 569.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 12:15:00 | 565.00 | 562.38 | 565.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 565.00 | 562.38 | 565.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 565.00 | 562.38 | 565.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:45:00 | 566.55 | 562.38 | 565.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 560.60 | 562.02 | 565.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:30:00 | 565.40 | 562.02 | 565.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 562.10 | 561.95 | 564.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 562.20 | 561.95 | 564.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 564.90 | 562.54 | 564.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 11:00:00 | 564.90 | 562.54 | 564.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 11:15:00 | 560.55 | 562.14 | 564.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 13:00:00 | 560.00 | 561.71 | 563.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 558.90 | 557.49 | 557.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 14:15:00 | 560.00 | 556.96 | 556.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 14:15:00 | 560.00 | 556.96 | 556.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 15:15:00 | 562.00 | 557.97 | 557.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 12:15:00 | 580.85 | 582.81 | 577.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-04 12:45:00 | 580.15 | 582.81 | 577.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 665.35 | 675.51 | 669.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 09:30:00 | 663.20 | 672.15 | 668.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 669.15 | 671.55 | 668.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 14:30:00 | 678.00 | 671.56 | 669.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 15:15:00 | 667.00 | 674.02 | 674.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 15:15:00 | 667.00 | 674.02 | 674.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 10:15:00 | 665.00 | 671.57 | 673.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 672.15 | 666.03 | 668.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 672.15 | 666.03 | 668.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 672.15 | 666.03 | 668.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 10:30:00 | 667.90 | 665.42 | 668.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:30:00 | 665.15 | 665.28 | 666.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 09:15:00 | 634.50 | 654.49 | 660.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 12:15:00 | 631.89 | 644.81 | 653.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-25 09:15:00 | 636.75 | 634.20 | 645.23 | SL hit (close>ema200) qty=0.50 sl=634.20 alert=retest2 |

### Cycle 30 — BUY (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 12:15:00 | 628.10 | 626.93 | 626.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 13:15:00 | 628.40 | 627.22 | 627.06 | Break + close above crossover candle high |

### Cycle 31 — SELL (started 2023-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 14:15:00 | 623.90 | 626.56 | 626.78 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 635.65 | 627.48 | 627.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 10:15:00 | 643.30 | 630.64 | 628.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 627.40 | 630.00 | 628.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 11:15:00 | 627.40 | 630.00 | 628.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 627.40 | 630.00 | 628.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 12:00:00 | 627.40 | 630.00 | 628.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 633.05 | 630.61 | 628.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 10:45:00 | 635.30 | 631.80 | 630.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 12:45:00 | 635.65 | 632.83 | 630.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 637.10 | 631.72 | 630.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 10:30:00 | 636.35 | 633.11 | 631.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 633.00 | 633.09 | 631.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 630.85 | 633.09 | 631.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 632.75 | 633.01 | 632.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 15:15:00 | 632.45 | 633.01 | 632.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 632.45 | 632.90 | 632.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 637.25 | 632.90 | 632.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 11:15:00 | 635.00 | 633.19 | 632.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 11:45:00 | 636.00 | 634.99 | 633.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 14:15:00 | 666.25 | 668.32 | 668.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-11-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 14:15:00 | 666.25 | 668.32 | 668.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 10:15:00 | 658.35 | 665.77 | 667.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 664.70 | 660.77 | 663.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 664.70 | 660.77 | 663.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 664.70 | 660.77 | 663.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:00:00 | 664.70 | 660.77 | 663.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 670.00 | 662.62 | 664.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:45:00 | 670.45 | 662.62 | 664.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 665.95 | 663.28 | 664.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:45:00 | 669.10 | 663.28 | 664.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 13:15:00 | 662.00 | 663.24 | 664.09 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 15:15:00 | 667.00 | 664.79 | 664.69 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 14:15:00 | 657.00 | 663.29 | 664.12 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 11:15:00 | 679.70 | 666.92 | 665.50 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-11-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 12:15:00 | 663.65 | 665.35 | 665.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 09:15:00 | 661.85 | 664.30 | 664.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 663.60 | 661.48 | 662.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 663.60 | 661.48 | 662.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 663.60 | 661.48 | 662.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:00:00 | 663.60 | 661.48 | 662.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 661.65 | 661.52 | 662.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 661.65 | 661.52 | 662.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 661.00 | 661.41 | 662.57 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 14:15:00 | 670.00 | 663.92 | 663.48 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 12:15:00 | 659.80 | 663.54 | 663.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 13:15:00 | 657.80 | 662.40 | 663.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 10:15:00 | 653.00 | 651.50 | 655.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 10:15:00 | 653.00 | 651.50 | 655.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 653.00 | 651.50 | 655.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 10:30:00 | 652.40 | 651.50 | 655.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 11:15:00 | 653.25 | 651.85 | 655.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 13:00:00 | 648.60 | 651.20 | 654.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 10:45:00 | 651.00 | 649.94 | 652.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-05 11:00:00 | 650.55 | 649.24 | 650.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 14:15:00 | 655.85 | 649.76 | 649.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 655.85 | 649.76 | 649.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 15:15:00 | 656.60 | 651.13 | 649.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 14:15:00 | 667.85 | 668.43 | 663.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 15:00:00 | 667.85 | 668.43 | 663.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 662.80 | 667.08 | 663.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:00:00 | 662.80 | 667.08 | 663.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 660.60 | 665.78 | 663.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:30:00 | 660.75 | 665.78 | 663.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 11:15:00 | 663.20 | 665.27 | 663.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 12:45:00 | 666.25 | 665.61 | 663.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 15:00:00 | 668.30 | 665.91 | 664.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 10:30:00 | 665.00 | 665.30 | 664.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 11:30:00 | 664.25 | 665.30 | 664.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 12:15:00 | 663.55 | 664.95 | 664.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 12:45:00 | 664.45 | 664.95 | 664.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 13:15:00 | 664.20 | 664.80 | 664.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:15:00 | 665.00 | 664.80 | 664.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 665.00 | 664.84 | 664.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:45:00 | 661.55 | 664.84 | 664.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 662.90 | 664.45 | 664.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-13 09:15:00 | 659.65 | 663.49 | 663.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 659.65 | 663.49 | 663.91 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 667.25 | 663.38 | 662.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 677.20 | 667.12 | 664.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 15:15:00 | 673.10 | 678.32 | 673.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 15:15:00 | 673.10 | 678.32 | 673.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 673.10 | 678.32 | 673.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 09:15:00 | 686.20 | 678.32 | 673.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 10:30:00 | 685.45 | 680.32 | 675.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 12:15:00 | 683.05 | 680.77 | 676.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 14:00:00 | 683.30 | 681.59 | 677.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 678.00 | 681.02 | 677.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 09:15:00 | 691.85 | 681.02 | 677.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 672.50 | 687.74 | 685.89 | SL hit (close<static) qty=1.00 sl=673.10 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 13:15:00 | 680.50 | 684.99 | 685.15 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-12-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 14:15:00 | 686.80 | 685.36 | 685.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 691.50 | 686.85 | 686.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 705.05 | 706.07 | 702.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 11:15:00 | 703.60 | 705.09 | 702.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 703.60 | 705.09 | 702.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 13:00:00 | 707.30 | 705.53 | 702.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 727.90 | 705.27 | 703.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-01 14:15:00 | 778.03 | 759.78 | 742.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 833.15 | 848.49 | 849.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 12:15:00 | 830.65 | 844.92 | 847.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 843.70 | 840.85 | 844.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 843.70 | 840.85 | 844.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 843.70 | 840.85 | 844.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 14:30:00 | 832.50 | 837.93 | 840.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 09:45:00 | 828.05 | 836.17 | 839.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 14:15:00 | 832.40 | 834.08 | 837.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 11:15:00 | 842.30 | 837.95 | 837.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 11:15:00 | 842.30 | 837.95 | 837.95 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 837.90 | 837.94 | 837.95 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 13:15:00 | 843.25 | 839.00 | 838.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 14:15:00 | 845.45 | 840.29 | 839.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 830.65 | 839.33 | 838.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 830.65 | 839.33 | 838.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 830.65 | 839.33 | 838.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:45:00 | 827.10 | 839.33 | 838.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 832.10 | 837.88 | 838.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 819.60 | 834.22 | 836.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 825.05 | 824.57 | 829.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 10:15:00 | 825.05 | 824.57 | 829.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 825.05 | 824.57 | 829.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 11:00:00 | 825.05 | 824.57 | 829.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 823.10 | 820.14 | 824.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:15:00 | 828.10 | 820.14 | 824.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 826.00 | 821.31 | 824.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:30:00 | 831.30 | 821.31 | 824.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 11:15:00 | 825.95 | 822.24 | 824.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 13:15:00 | 825.00 | 823.17 | 825.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 14:00:00 | 825.00 | 823.54 | 825.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 14:30:00 | 820.00 | 822.64 | 824.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 10:15:00 | 830.00 | 823.68 | 824.49 | SL hit (close>static) qty=1.00 sl=828.70 alert=retest2 |

### Cycle 50 — BUY (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 12:15:00 | 831.20 | 825.56 | 825.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 13:15:00 | 835.95 | 827.63 | 826.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 825.10 | 831.52 | 829.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 14:15:00 | 825.10 | 831.52 | 829.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 825.10 | 831.52 | 829.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:45:00 | 825.65 | 831.52 | 829.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 820.00 | 829.22 | 828.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:15:00 | 825.80 | 829.22 | 828.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 821.20 | 827.61 | 828.20 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 840.95 | 829.87 | 828.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 10:15:00 | 847.30 | 833.36 | 830.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 876.25 | 876.84 | 864.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 15:00:00 | 876.25 | 876.84 | 864.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 871.95 | 875.57 | 865.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:30:00 | 866.15 | 875.57 | 865.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 884.50 | 888.52 | 878.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:45:00 | 884.50 | 888.52 | 878.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 870.65 | 883.86 | 877.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 870.65 | 883.86 | 877.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 876.25 | 882.34 | 877.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 13:15:00 | 878.30 | 882.34 | 877.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-02 15:15:00 | 869.00 | 877.80 | 878.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 869.00 | 877.80 | 878.25 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 09:15:00 | 893.50 | 878.89 | 877.57 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 11:15:00 | 858.70 | 876.16 | 878.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 848.50 | 862.52 | 869.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 11:15:00 | 800.00 | 799.14 | 814.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 11:45:00 | 806.85 | 799.14 | 814.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 814.40 | 803.70 | 813.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 814.40 | 803.70 | 813.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 820.00 | 806.96 | 813.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 809.45 | 806.96 | 813.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 12:30:00 | 811.45 | 810.43 | 813.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 13:00:00 | 811.55 | 810.43 | 813.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 15:15:00 | 822.00 | 811.11 | 811.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 15:15:00 | 822.00 | 811.11 | 811.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 835.35 | 815.96 | 813.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 12:15:00 | 813.50 | 817.33 | 814.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 12:15:00 | 813.50 | 817.33 | 814.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 813.50 | 817.33 | 814.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:00:00 | 813.50 | 817.33 | 814.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 808.00 | 815.46 | 814.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 14:00:00 | 808.00 | 815.46 | 814.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 805.95 | 813.56 | 813.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 15:00:00 | 805.95 | 813.56 | 813.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 15:15:00 | 809.65 | 812.78 | 813.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 13:15:00 | 802.50 | 807.35 | 810.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 09:15:00 | 809.20 | 807.04 | 809.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 809.20 | 807.04 | 809.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 809.20 | 807.04 | 809.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:30:00 | 805.15 | 807.04 | 809.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 806.60 | 806.95 | 808.92 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 14:15:00 | 811.30 | 810.04 | 809.97 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 15:15:00 | 805.00 | 809.03 | 809.52 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 09:15:00 | 816.15 | 810.46 | 810.12 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 10:15:00 | 807.30 | 809.83 | 809.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 11:15:00 | 804.85 | 808.83 | 809.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 12:15:00 | 809.55 | 808.97 | 809.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 12:15:00 | 809.55 | 808.97 | 809.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 12:15:00 | 809.55 | 808.97 | 809.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 13:00:00 | 809.55 | 808.97 | 809.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 804.65 | 808.11 | 808.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 13:45:00 | 807.90 | 808.11 | 808.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 799.95 | 805.36 | 807.41 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 811.05 | 806.83 | 806.60 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 14:15:00 | 802.45 | 805.89 | 806.31 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 09:15:00 | 809.05 | 806.57 | 806.55 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 798.10 | 807.09 | 808.07 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 815.50 | 806.27 | 805.66 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 802.85 | 807.86 | 807.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 13:15:00 | 801.55 | 805.75 | 806.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 15:15:00 | 806.80 | 805.45 | 806.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 15:15:00 | 806.80 | 805.45 | 806.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 806.80 | 805.45 | 806.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:30:00 | 812.00 | 805.56 | 806.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 799.95 | 804.44 | 805.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 11:45:00 | 796.60 | 802.76 | 805.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 15:15:00 | 809.00 | 803.58 | 804.61 | SL hit (close>static) qty=1.00 sl=806.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 12:15:00 | 810.10 | 802.10 | 801.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 13:15:00 | 814.20 | 804.52 | 802.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 801.70 | 807.89 | 804.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 801.70 | 807.89 | 804.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 801.70 | 807.89 | 804.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:00:00 | 801.70 | 807.89 | 804.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 810.95 | 808.50 | 805.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 11:30:00 | 812.10 | 809.22 | 805.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 13:00:00 | 812.10 | 809.80 | 806.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 13:45:00 | 812.10 | 810.26 | 806.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 09:15:00 | 793.00 | 807.31 | 806.51 | SL hit (close<static) qty=1.00 sl=799.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 788.60 | 803.57 | 804.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 786.00 | 800.05 | 803.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 13:15:00 | 779.00 | 778.97 | 788.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-13 14:00:00 | 779.00 | 778.97 | 788.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 756.90 | 755.50 | 764.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:45:00 | 759.05 | 755.50 | 764.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 760.35 | 756.47 | 764.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:00:00 | 760.35 | 756.47 | 764.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 752.55 | 755.68 | 763.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:15:00 | 759.75 | 755.68 | 763.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 755.60 | 755.67 | 762.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 13:00:00 | 746.35 | 752.58 | 759.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 14:45:00 | 743.30 | 749.74 | 756.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:30:00 | 745.55 | 745.59 | 753.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 11:15:00 | 752.20 | 744.96 | 744.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 752.20 | 744.96 | 744.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 764.60 | 756.94 | 752.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 760.75 | 761.40 | 756.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 14:15:00 | 760.75 | 761.40 | 756.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 760.75 | 761.40 | 756.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:30:00 | 755.50 | 761.40 | 756.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 762.80 | 761.68 | 757.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 772.90 | 761.68 | 757.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 10:45:00 | 766.95 | 768.76 | 765.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-08 11:15:00 | 843.65 | 821.86 | 811.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 835.10 | 843.41 | 844.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 829.00 | 839.98 | 842.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 11:15:00 | 841.40 | 839.20 | 841.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 11:15:00 | 841.40 | 839.20 | 841.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 841.40 | 839.20 | 841.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 12:00:00 | 841.40 | 839.20 | 841.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 855.95 | 842.55 | 843.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:00:00 | 855.95 | 842.55 | 843.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 13:15:00 | 858.00 | 845.64 | 844.48 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 10:15:00 | 841.30 | 843.45 | 843.69 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 12:15:00 | 847.40 | 843.83 | 843.49 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 838.65 | 842.79 | 843.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 837.10 | 841.65 | 842.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 837.75 | 834.96 | 837.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 837.75 | 834.96 | 837.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 837.75 | 834.96 | 837.90 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 843.95 | 840.10 | 839.67 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 15:15:00 | 831.00 | 838.75 | 839.16 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 861.00 | 843.20 | 841.15 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 13:15:00 | 846.25 | 849.88 | 849.89 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 14:15:00 | 856.95 | 851.29 | 850.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 858.85 | 853.26 | 851.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 09:15:00 | 925.75 | 939.21 | 914.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 09:15:00 | 925.75 | 939.21 | 914.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 925.75 | 939.21 | 914.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:45:00 | 926.50 | 939.21 | 914.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 877.00 | 926.77 | 911.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:00:00 | 877.00 | 926.77 | 911.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 884.40 | 918.29 | 908.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 12:30:00 | 895.85 | 914.01 | 907.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 15:15:00 | 892.00 | 904.60 | 904.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 15:15:00 | 892.00 | 904.60 | 904.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 876.00 | 898.88 | 902.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 881.10 | 876.00 | 885.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 881.10 | 876.00 | 885.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 881.10 | 876.00 | 885.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 881.10 | 876.00 | 885.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 886.80 | 878.16 | 885.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 15:15:00 | 878.00 | 883.22 | 886.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 11:00:00 | 876.65 | 881.90 | 884.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 13:15:00 | 834.10 | 849.71 | 861.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 13:15:00 | 832.82 | 849.71 | 861.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 12:15:00 | 828.40 | 824.61 | 835.23 | SL hit (close>ema200) qty=0.50 sl=824.61 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 13:15:00 | 820.10 | 802.04 | 800.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 838.35 | 814.85 | 806.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 12:15:00 | 831.95 | 833.29 | 824.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 13:00:00 | 831.95 | 833.29 | 824.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 832.70 | 835.75 | 831.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:15:00 | 828.05 | 835.75 | 831.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 824.60 | 833.52 | 830.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 824.60 | 833.52 | 830.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 825.00 | 831.82 | 830.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 845.90 | 831.82 | 830.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 818.25 | 837.16 | 839.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 818.25 | 837.16 | 839.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 805.00 | 819.43 | 828.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 10:15:00 | 791.95 | 789.63 | 802.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-03 11:00:00 | 791.95 | 789.63 | 802.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 797.40 | 790.81 | 800.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:45:00 | 799.80 | 790.81 | 800.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 805.75 | 793.80 | 801.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:00:00 | 805.75 | 793.80 | 801.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 791.55 | 793.35 | 800.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 764.60 | 793.28 | 799.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 783.75 | 783.86 | 794.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 726.37 | 774.99 | 789.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 744.56 | 774.99 | 789.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 12:15:00 | 778.40 | 775.68 | 788.11 | SL hit (close>ema200) qty=0.50 sl=775.68 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 801.60 | 790.45 | 790.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 823.45 | 804.48 | 799.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 804.75 | 811.86 | 806.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 09:15:00 | 804.75 | 811.86 | 806.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 804.75 | 811.86 | 806.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:30:00 | 807.00 | 811.86 | 806.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 804.90 | 810.47 | 806.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:45:00 | 804.45 | 810.47 | 806.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 807.90 | 809.50 | 806.83 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 10:15:00 | 802.90 | 805.75 | 805.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 11:15:00 | 802.05 | 805.01 | 805.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 14:15:00 | 800.10 | 799.65 | 801.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-12 15:00:00 | 800.10 | 799.65 | 801.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 86 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 836.90 | 807.17 | 804.74 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 12:15:00 | 814.45 | 820.01 | 820.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 10:15:00 | 812.85 | 817.55 | 819.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 829.00 | 817.86 | 818.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 829.00 | 817.86 | 818.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 829.00 | 817.86 | 818.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 829.00 | 817.86 | 818.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 823.50 | 818.99 | 818.70 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 813.15 | 819.06 | 819.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 11:15:00 | 811.05 | 817.46 | 818.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 823.95 | 811.73 | 814.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 823.95 | 811.73 | 814.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 823.95 | 811.73 | 814.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 823.95 | 811.73 | 814.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 846.40 | 818.66 | 817.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 851.50 | 832.76 | 825.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 14:15:00 | 844.00 | 845.85 | 836.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 14:45:00 | 845.80 | 845.85 | 836.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 853.35 | 847.33 | 838.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:15:00 | 857.00 | 847.33 | 838.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 857.35 | 845.26 | 843.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 861.95 | 849.14 | 846.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 12:15:00 | 900.60 | 904.46 | 904.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 900.60 | 904.46 | 904.68 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 906.40 | 904.69 | 904.56 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 14:15:00 | 901.60 | 904.16 | 904.36 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 906.80 | 904.90 | 904.65 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 903.25 | 904.43 | 904.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 15:15:00 | 900.00 | 902.99 | 903.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 890.00 | 883.24 | 888.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 890.00 | 883.24 | 888.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 890.00 | 883.24 | 888.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 890.00 | 883.24 | 888.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 889.70 | 884.53 | 888.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 890.05 | 884.53 | 888.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 890.75 | 888.09 | 888.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 885.50 | 888.09 | 888.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 09:15:00 | 900.90 | 890.65 | 890.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 900.90 | 890.65 | 890.04 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 10:15:00 | 883.90 | 890.03 | 890.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 11:15:00 | 882.00 | 888.42 | 889.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 13:15:00 | 888.00 | 887.32 | 889.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 13:15:00 | 888.00 | 887.32 | 889.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 888.00 | 887.32 | 889.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 13:30:00 | 888.95 | 887.32 | 889.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 889.50 | 887.75 | 889.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 15:00:00 | 889.50 | 887.75 | 889.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 890.00 | 888.20 | 889.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:45:00 | 880.10 | 887.12 | 888.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 895.30 | 889.47 | 889.49 | SL hit (close>static) qty=1.00 sl=891.75 alert=retest2 |

### Cycle 98 — BUY (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 12:15:00 | 891.85 | 889.94 | 889.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 901.85 | 892.13 | 890.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 898.55 | 901.99 | 897.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 898.55 | 901.99 | 897.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 905.00 | 902.59 | 898.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 15:15:00 | 901.00 | 902.59 | 898.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 901.00 | 902.27 | 898.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 907.75 | 902.27 | 898.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:30:00 | 910.00 | 905.59 | 902.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:45:00 | 906.75 | 906.95 | 904.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 13:45:00 | 910.65 | 907.08 | 904.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 906.85 | 907.03 | 905.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 906.85 | 907.03 | 905.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 908.00 | 907.23 | 905.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 914.45 | 907.23 | 905.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 894.70 | 924.69 | 925.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 894.70 | 924.69 | 925.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 873.15 | 914.38 | 920.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 10:15:00 | 889.45 | 888.85 | 901.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 11:15:00 | 893.95 | 888.85 | 901.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 874.45 | 880.12 | 891.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 864.50 | 878.06 | 885.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 10:15:00 | 896.65 | 884.34 | 882.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 896.65 | 884.34 | 882.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 897.90 | 890.43 | 886.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 895.65 | 896.20 | 891.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 12:00:00 | 895.65 | 896.20 | 891.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 892.75 | 895.51 | 891.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 892.75 | 895.51 | 891.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 901.25 | 896.66 | 892.23 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 883.00 | 889.72 | 890.05 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 899.60 | 891.73 | 890.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 907.00 | 896.27 | 893.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 11:15:00 | 935.00 | 935.56 | 928.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 12:00:00 | 935.00 | 935.56 | 928.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 935.45 | 937.76 | 934.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 938.20 | 937.76 | 934.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 938.95 | 938.04 | 935.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 938.95 | 938.04 | 935.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1011.50 | 1016.94 | 1007.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 1011.50 | 1016.94 | 1007.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 1016.00 | 1016.75 | 1008.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 1071.85 | 1016.75 | 1008.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-03 14:15:00 | 1179.04 | 1118.16 | 1077.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 1202.50 | 1210.15 | 1210.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 09:15:00 | 1196.90 | 1206.67 | 1208.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 14:15:00 | 1204.60 | 1204.01 | 1206.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 1204.60 | 1204.01 | 1206.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 1204.60 | 1204.01 | 1206.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 1204.60 | 1204.01 | 1206.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 1205.00 | 1204.21 | 1206.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 1209.30 | 1204.21 | 1206.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1209.45 | 1205.26 | 1206.52 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 1211.50 | 1207.87 | 1207.57 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 1200.60 | 1206.47 | 1207.04 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 1230.00 | 1210.70 | 1208.83 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 1205.00 | 1215.15 | 1215.80 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 1238.35 | 1211.54 | 1209.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 1245.25 | 1232.94 | 1223.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 11:15:00 | 1232.85 | 1233.40 | 1225.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 12:30:00 | 1238.15 | 1234.63 | 1226.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 13:00:00 | 1239.55 | 1234.63 | 1226.39 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 1235.00 | 1234.78 | 1228.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-25 12:15:00 | 1228.00 | 1232.89 | 1229.58 | SL hit (close<ema400) qty=1.00 sl=1229.58 alert=retest1 |

### Cycle 109 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1231.00 | 1233.58 | 1233.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 1217.75 | 1229.71 | 1231.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 1085.45 | 1085.31 | 1104.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 1085.45 | 1085.31 | 1104.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1096.95 | 1089.27 | 1101.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1109.60 | 1089.27 | 1101.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1122.45 | 1095.91 | 1103.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 1120.50 | 1095.91 | 1103.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1116.85 | 1100.10 | 1104.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 1128.30 | 1100.10 | 1104.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 1111.10 | 1107.18 | 1107.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 15:15:00 | 1127.40 | 1112.22 | 1109.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 1116.75 | 1117.24 | 1113.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 1116.75 | 1117.24 | 1113.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 1101.70 | 1114.13 | 1112.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 1101.70 | 1114.13 | 1112.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1102.25 | 1111.75 | 1111.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 1102.25 | 1111.75 | 1111.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 15:15:00 | 1095.00 | 1108.40 | 1109.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 1090.95 | 1104.91 | 1108.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 14:15:00 | 1093.75 | 1093.36 | 1100.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-11 15:00:00 | 1093.75 | 1093.36 | 1100.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1085.85 | 1090.54 | 1097.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 1081.10 | 1090.37 | 1094.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:15:00 | 1027.04 | 1052.34 | 1070.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-16 13:15:00 | 1054.85 | 1046.25 | 1061.08 | SL hit (close>ema200) qty=0.50 sl=1046.25 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 12:15:00 | 1027.80 | 1016.96 | 1015.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 15:15:00 | 1042.00 | 1026.05 | 1020.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 10:15:00 | 1009.00 | 1022.77 | 1020.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 10:15:00 | 1009.00 | 1022.77 | 1020.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 1009.00 | 1022.77 | 1020.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:00:00 | 1009.00 | 1022.77 | 1020.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 1005.55 | 1019.33 | 1018.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:45:00 | 1005.25 | 1019.33 | 1018.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 1015.00 | 1018.46 | 1018.40 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 1015.00 | 1017.77 | 1018.09 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 14:15:00 | 1025.95 | 1019.41 | 1018.80 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 1002.80 | 1017.85 | 1019.17 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 1023.75 | 1020.16 | 1020.07 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 1017.25 | 1019.58 | 1019.81 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 1029.65 | 1021.59 | 1020.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 1035.00 | 1025.23 | 1022.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 1026.00 | 1031.44 | 1027.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 13:15:00 | 1026.00 | 1031.44 | 1027.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 1026.00 | 1031.44 | 1027.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 1026.00 | 1031.44 | 1027.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 1023.90 | 1029.93 | 1027.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 1023.90 | 1029.93 | 1027.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 1025.15 | 1028.97 | 1027.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 1021.15 | 1028.97 | 1027.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1025.55 | 1028.29 | 1027.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:45:00 | 1030.95 | 1029.63 | 1027.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 11:30:00 | 1034.50 | 1030.80 | 1028.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:15:00 | 1034.35 | 1031.05 | 1029.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 10:15:00 | 1012.00 | 1029.56 | 1030.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1012.00 | 1029.56 | 1030.03 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 1030.10 | 1026.14 | 1025.70 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 1017.00 | 1024.25 | 1025.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 14:15:00 | 1013.40 | 1021.04 | 1023.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 953.05 | 937.76 | 946.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 953.05 | 937.76 | 946.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 953.05 | 937.76 | 946.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:45:00 | 945.55 | 937.76 | 946.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 967.20 | 943.65 | 948.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:45:00 | 969.60 | 943.65 | 948.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 977.00 | 950.32 | 950.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:45:00 | 979.70 | 950.32 | 950.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 992.65 | 958.78 | 954.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 13:15:00 | 1000.30 | 967.09 | 958.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 1001.05 | 1012.60 | 995.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 10:00:00 | 1001.05 | 1012.60 | 995.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 999.75 | 1010.03 | 996.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 12:15:00 | 1006.20 | 1007.96 | 996.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 1010.00 | 1003.74 | 997.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 12:15:00 | 1063.75 | 1067.68 | 1067.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 1063.75 | 1067.68 | 1067.73 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 1068.55 | 1067.85 | 1067.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 15:15:00 | 1092.05 | 1072.68 | 1070.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 1072.20 | 1073.63 | 1071.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 1072.20 | 1073.63 | 1071.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1072.20 | 1073.63 | 1071.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 1072.20 | 1073.63 | 1071.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1075.05 | 1073.91 | 1071.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:45:00 | 1073.25 | 1073.91 | 1071.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1076.60 | 1074.45 | 1072.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:45:00 | 1076.05 | 1074.45 | 1072.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1074.75 | 1074.51 | 1072.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 1074.75 | 1074.51 | 1072.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 1076.00 | 1074.81 | 1072.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 1066.70 | 1072.73 | 1071.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1066.20 | 1071.42 | 1071.35 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 11:15:00 | 1066.35 | 1070.41 | 1070.90 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 1082.65 | 1071.31 | 1071.04 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 1068.10 | 1071.02 | 1071.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 14:15:00 | 1066.10 | 1069.56 | 1070.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 10:15:00 | 1073.00 | 1069.82 | 1070.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 10:15:00 | 1073.00 | 1069.82 | 1070.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1073.00 | 1069.82 | 1070.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 12:15:00 | 1065.20 | 1069.76 | 1070.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:00:00 | 1065.85 | 1068.53 | 1069.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 1066.25 | 1068.15 | 1069.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 11:00:00 | 1068.05 | 1067.86 | 1068.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 1066.35 | 1067.56 | 1068.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:00:00 | 1064.00 | 1066.85 | 1068.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 14:15:00 | 1107.70 | 1075.04 | 1071.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 1107.70 | 1075.04 | 1071.66 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1066.25 | 1078.99 | 1079.77 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 1089.95 | 1079.06 | 1078.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 1095.45 | 1084.46 | 1081.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 1094.50 | 1095.19 | 1089.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 12:00:00 | 1094.50 | 1095.19 | 1089.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1092.05 | 1094.17 | 1089.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:30:00 | 1090.00 | 1094.17 | 1089.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1094.25 | 1096.02 | 1091.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:00:00 | 1119.40 | 1101.93 | 1095.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 1126.95 | 1130.15 | 1130.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 1126.95 | 1130.15 | 1130.29 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1214.80 | 1147.08 | 1137.97 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 15:15:00 | 1151.20 | 1169.46 | 1171.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1134.55 | 1162.47 | 1167.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 965.50 | 958.93 | 974.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 11:00:00 | 965.50 | 958.93 | 974.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 965.60 | 960.26 | 973.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:00:00 | 956.95 | 959.60 | 971.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 957.60 | 958.15 | 969.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:15:00 | 956.95 | 959.03 | 966.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:15:00 | 909.10 | 916.34 | 927.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:15:00 | 909.72 | 916.34 | 927.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:15:00 | 909.10 | 916.34 | 927.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 894.15 | 886.48 | 898.19 | SL hit (close>ema200) qty=0.50 sl=886.48 alert=retest2 |

### Cycle 134 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 867.65 | 850.05 | 848.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 873.60 | 854.76 | 850.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 11:15:00 | 888.40 | 889.07 | 878.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 11:45:00 | 888.00 | 889.07 | 878.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 890.90 | 891.02 | 884.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 887.25 | 891.02 | 884.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 884.50 | 889.71 | 884.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 884.50 | 889.71 | 884.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 895.20 | 890.81 | 885.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 903.05 | 898.06 | 889.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 14:45:00 | 904.30 | 903.06 | 896.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 880.25 | 893.20 | 894.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 12:15:00 | 880.25 | 893.20 | 894.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 877.70 | 885.32 | 887.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 874.45 | 831.59 | 839.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 874.45 | 831.59 | 839.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 874.45 | 831.59 | 839.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 874.45 | 831.59 | 839.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 869.55 | 839.18 | 842.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 879.05 | 839.18 | 842.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 853.85 | 844.10 | 844.00 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 827.55 | 841.98 | 843.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 15:15:00 | 821.00 | 837.78 | 841.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 10:15:00 | 807.20 | 805.18 | 817.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 807.20 | 805.18 | 817.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 807.20 | 805.18 | 817.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 816.20 | 805.18 | 817.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 816.75 | 807.49 | 817.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 816.75 | 807.49 | 817.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 834.80 | 812.95 | 819.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 837.20 | 812.95 | 819.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 838.85 | 818.13 | 820.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 14:45:00 | 831.50 | 820.33 | 821.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 09:15:00 | 830.75 | 822.73 | 822.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 09:15:00 | 830.75 | 822.73 | 822.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 843.05 | 829.25 | 825.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 11:15:00 | 1100.00 | 1101.17 | 1066.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 11:30:00 | 1100.25 | 1101.17 | 1066.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1102.90 | 1100.29 | 1089.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:15:00 | 1118.00 | 1104.72 | 1103.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:00:00 | 1113.90 | 1106.41 | 1104.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1113.15 | 1108.82 | 1106.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:00:00 | 1113.60 | 1108.15 | 1106.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1103.75 | 1121.65 | 1115.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 1104.75 | 1121.65 | 1115.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1121.90 | 1121.70 | 1115.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:45:00 | 1153.25 | 1123.05 | 1116.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:30:00 | 1124.10 | 1121.93 | 1117.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 1123.75 | 1122.39 | 1119.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:45:00 | 1126.75 | 1125.91 | 1122.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1116.70 | 1124.07 | 1122.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:00:00 | 1116.70 | 1124.07 | 1122.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1118.10 | 1122.87 | 1121.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:30:00 | 1112.00 | 1122.87 | 1121.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1114.30 | 1121.16 | 1121.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:00:00 | 1114.30 | 1121.16 | 1121.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-13 13:15:00 | 1120.00 | 1120.93 | 1120.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 13:15:00 | 1120.00 | 1120.93 | 1120.94 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 14:15:00 | 1149.00 | 1126.54 | 1123.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 1161.80 | 1134.39 | 1127.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 15:15:00 | 1131.00 | 1138.70 | 1133.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 15:15:00 | 1131.00 | 1138.70 | 1133.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 1131.00 | 1138.70 | 1133.39 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 10:15:00 | 1126.55 | 1132.44 | 1133.10 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 11:15:00 | 1147.60 | 1135.48 | 1134.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 14:15:00 | 1164.95 | 1144.97 | 1139.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1174.25 | 1217.88 | 1207.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 1174.25 | 1217.88 | 1207.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1174.25 | 1217.88 | 1207.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1174.25 | 1217.88 | 1207.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1184.60 | 1211.23 | 1205.64 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 1185.05 | 1200.20 | 1201.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 1150.70 | 1182.61 | 1192.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 1168.05 | 1158.26 | 1171.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 1168.05 | 1158.26 | 1171.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1168.05 | 1158.26 | 1171.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 1166.00 | 1158.26 | 1171.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1154.65 | 1157.80 | 1167.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1154.65 | 1157.80 | 1167.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1153.00 | 1156.45 | 1164.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:30:00 | 1145.65 | 1153.58 | 1161.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 09:15:00 | 1088.37 | 1133.50 | 1149.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 1114.15 | 1101.30 | 1118.64 | SL hit (close>ema200) qty=0.50 sl=1101.30 alert=retest2 |

### Cycle 144 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 1097.05 | 1079.39 | 1077.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 1119.75 | 1102.88 | 1092.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 13:15:00 | 1130.70 | 1130.96 | 1118.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 13:30:00 | 1129.50 | 1130.96 | 1118.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 1122.40 | 1129.88 | 1122.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 11:00:00 | 1122.40 | 1129.88 | 1122.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 1126.40 | 1129.19 | 1122.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 15:00:00 | 1131.00 | 1128.01 | 1123.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 1116.00 | 1126.47 | 1123.59 | SL hit (close<static) qty=1.00 sl=1121.70 alert=retest2 |

### Cycle 145 — SELL (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 12:15:00 | 1137.70 | 1148.77 | 1149.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 1122.20 | 1140.23 | 1144.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 1112.90 | 1111.33 | 1123.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:45:00 | 1114.10 | 1111.33 | 1123.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 1125.50 | 1114.40 | 1120.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 1125.50 | 1114.40 | 1120.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 1129.00 | 1117.32 | 1121.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 1116.30 | 1117.32 | 1121.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1130.70 | 1118.71 | 1121.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 1130.70 | 1118.71 | 1121.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1126.00 | 1120.17 | 1121.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:45:00 | 1125.60 | 1120.17 | 1121.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 1106.30 | 1118.01 | 1120.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:30:00 | 1110.20 | 1118.01 | 1120.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 1084.10 | 1066.53 | 1077.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 1084.10 | 1066.53 | 1077.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1093.10 | 1071.84 | 1078.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:45:00 | 1090.50 | 1071.84 | 1078.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1092.90 | 1079.87 | 1080.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 1092.90 | 1079.87 | 1080.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 1104.80 | 1084.86 | 1083.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 14:15:00 | 1113.90 | 1096.57 | 1089.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 1115.60 | 1116.90 | 1108.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 13:00:00 | 1115.60 | 1116.90 | 1108.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1111.60 | 1115.23 | 1109.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1111.60 | 1115.23 | 1109.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1102.60 | 1112.70 | 1108.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1101.00 | 1112.70 | 1108.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1086.70 | 1107.50 | 1106.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 1086.70 | 1107.50 | 1106.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 1082.10 | 1102.42 | 1104.47 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1122.50 | 1101.40 | 1100.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1139.00 | 1119.63 | 1111.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1132.50 | 1156.45 | 1144.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1132.50 | 1156.45 | 1144.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1132.50 | 1156.45 | 1144.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 1133.10 | 1156.45 | 1144.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1131.50 | 1151.46 | 1143.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:45:00 | 1133.20 | 1151.46 | 1143.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 14:15:00 | 1130.10 | 1138.62 | 1139.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 15:15:00 | 1124.70 | 1135.84 | 1137.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 1143.80 | 1137.43 | 1138.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 1143.80 | 1137.43 | 1138.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1143.80 | 1137.43 | 1138.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 10:30:00 | 1137.10 | 1137.36 | 1138.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 15:15:00 | 1140.00 | 1137.87 | 1138.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 15:15:00 | 1140.00 | 1138.30 | 1138.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 1140.00 | 1138.30 | 1138.24 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1135.10 | 1137.66 | 1137.95 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1145.10 | 1139.15 | 1138.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 13:15:00 | 1150.00 | 1142.39 | 1140.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 12:15:00 | 1170.20 | 1170.77 | 1164.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 1170.20 | 1170.77 | 1164.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 1170.00 | 1170.61 | 1165.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 1171.50 | 1170.61 | 1165.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1173.30 | 1170.99 | 1166.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 1180.00 | 1174.49 | 1171.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1181.00 | 1175.47 | 1172.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:45:00 | 1180.80 | 1180.68 | 1178.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1182.10 | 1180.68 | 1178.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1180.50 | 1180.64 | 1178.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:15:00 | 1185.60 | 1181.36 | 1179.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 15:15:00 | 1193.00 | 1182.09 | 1179.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 1186.60 | 1184.24 | 1181.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 1187.00 | 1183.59 | 1181.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1192.00 | 1185.27 | 1182.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 1195.80 | 1185.27 | 1182.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 14:15:00 | 1194.50 | 1186.43 | 1183.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:00:00 | 1195.00 | 1190.53 | 1186.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 15:00:00 | 1194.40 | 1197.07 | 1192.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1189.30 | 1195.52 | 1191.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 1189.00 | 1195.52 | 1191.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1182.90 | 1192.99 | 1190.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 1182.90 | 1192.99 | 1190.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1186.30 | 1191.66 | 1190.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 1183.80 | 1191.66 | 1190.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1195.60 | 1189.80 | 1189.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 1226.00 | 1197.70 | 1193.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 1339.10 | 1340.55 | 1304.19 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:30:00 | 1357.00 | 1342.44 | 1308.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1323.80 | 1334.31 | 1317.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 1307.70 | 1334.31 | 1317.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1291.00 | 1325.65 | 1314.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 1291.00 | 1325.65 | 1314.80 | SL hit (close<ema400) qty=1.00 sl=1314.80 alert=retest1 |

### Cycle 155 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 1289.10 | 1306.57 | 1307.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1273.10 | 1285.91 | 1292.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1279.60 | 1278.49 | 1286.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 1280.50 | 1278.49 | 1286.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1278.70 | 1271.99 | 1279.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 1278.70 | 1271.99 | 1279.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1280.50 | 1273.69 | 1279.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1280.50 | 1273.69 | 1279.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1298.40 | 1278.64 | 1281.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:45:00 | 1290.50 | 1278.64 | 1281.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1303.00 | 1283.51 | 1283.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1307.10 | 1283.51 | 1283.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1306.60 | 1288.13 | 1285.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1319.40 | 1294.38 | 1288.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 1293.00 | 1295.63 | 1290.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 12:00:00 | 1293.00 | 1295.63 | 1290.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1329.10 | 1340.62 | 1331.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:00:00 | 1349.90 | 1342.47 | 1332.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:30:00 | 1365.00 | 1348.54 | 1338.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1312.20 | 1341.14 | 1339.16 | SL hit (close<static) qty=1.00 sl=1315.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 1297.70 | 1332.45 | 1335.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 12:15:00 | 1287.30 | 1323.42 | 1331.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 15:15:00 | 1256.00 | 1252.66 | 1269.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:15:00 | 1259.40 | 1252.66 | 1269.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1248.40 | 1247.73 | 1257.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1249.20 | 1247.73 | 1257.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1247.30 | 1247.64 | 1256.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1247.30 | 1247.64 | 1256.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1253.20 | 1248.30 | 1253.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1264.00 | 1248.30 | 1253.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1256.10 | 1249.86 | 1253.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:45:00 | 1251.30 | 1251.07 | 1253.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 11:15:00 | 1248.80 | 1251.07 | 1253.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 1248.40 | 1248.46 | 1250.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:15:00 | 1188.73 | 1217.69 | 1233.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:15:00 | 1186.36 | 1217.69 | 1233.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:15:00 | 1185.98 | 1217.69 | 1233.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-03 09:15:00 | 1126.17 | 1167.26 | 1196.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 1164.80 | 1120.32 | 1115.07 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 1133.90 | 1141.17 | 1141.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 1126.60 | 1138.25 | 1140.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 1138.30 | 1138.26 | 1139.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 12:00:00 | 1138.30 | 1138.26 | 1139.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1131.60 | 1136.93 | 1139.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 1128.00 | 1136.09 | 1138.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:45:00 | 1129.20 | 1130.70 | 1133.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 1127.80 | 1131.81 | 1133.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1140.70 | 1132.17 | 1131.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 1140.70 | 1132.17 | 1131.34 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 1124.00 | 1130.77 | 1131.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 09:15:00 | 1102.20 | 1125.06 | 1128.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 1117.60 | 1112.95 | 1119.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 15:00:00 | 1117.60 | 1112.95 | 1119.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1119.60 | 1114.28 | 1119.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1122.60 | 1114.28 | 1119.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1120.10 | 1115.45 | 1119.92 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1137.90 | 1123.41 | 1122.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1144.80 | 1129.48 | 1125.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 1133.10 | 1135.48 | 1129.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:15:00 | 1137.40 | 1135.48 | 1129.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1134.50 | 1135.28 | 1130.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1134.50 | 1135.28 | 1130.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1130.00 | 1133.65 | 1130.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 1129.00 | 1133.65 | 1130.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1136.50 | 1134.22 | 1130.99 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 1114.90 | 1129.42 | 1129.52 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 1135.90 | 1126.41 | 1125.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 1140.10 | 1129.15 | 1126.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 1128.00 | 1134.69 | 1132.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1128.00 | 1134.69 | 1132.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1128.00 | 1134.69 | 1132.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 1128.00 | 1134.69 | 1132.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 1096.20 | 1126.99 | 1128.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 1091.20 | 1100.78 | 1108.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 1102.00 | 1096.28 | 1101.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1102.00 | 1096.28 | 1101.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1102.00 | 1096.28 | 1101.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:15:00 | 1092.30 | 1096.89 | 1100.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 1093.80 | 1096.27 | 1100.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1118.40 | 1100.59 | 1101.17 | SL hit (close>static) qty=1.00 sl=1106.50 alert=retest2 |

### Cycle 166 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 1115.00 | 1103.47 | 1102.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 1123.50 | 1107.48 | 1104.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 13:15:00 | 1154.30 | 1154.65 | 1138.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:30:00 | 1153.70 | 1154.65 | 1138.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1285.90 | 1286.80 | 1281.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 1295.00 | 1286.23 | 1283.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 1271.40 | 1283.30 | 1282.85 | SL hit (close<static) qty=1.00 sl=1279.20 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1270.30 | 1280.70 | 1281.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1265.60 | 1272.81 | 1277.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1202.00 | 1199.63 | 1209.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 1202.00 | 1199.63 | 1209.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1198.50 | 1199.08 | 1207.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1186.20 | 1197.26 | 1206.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 1187.60 | 1194.32 | 1199.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1203.60 | 1200.26 | 1200.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 1203.60 | 1200.26 | 1200.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1218.00 | 1206.62 | 1204.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 1207.90 | 1208.01 | 1205.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 1207.90 | 1208.01 | 1205.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1207.90 | 1208.01 | 1205.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 1204.40 | 1208.01 | 1205.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1201.70 | 1206.75 | 1205.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1201.70 | 1206.75 | 1205.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1210.50 | 1207.50 | 1205.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 1213.00 | 1207.82 | 1206.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1213.80 | 1207.33 | 1206.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 1197.40 | 1206.01 | 1205.91 | SL hit (close<static) qty=1.00 sl=1201.70 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 1202.40 | 1205.29 | 1205.59 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 1212.00 | 1206.51 | 1206.04 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 1202.00 | 1205.37 | 1205.58 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 1209.00 | 1206.20 | 1205.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1232.50 | 1213.53 | 1209.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 1236.50 | 1236.98 | 1225.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:30:00 | 1237.20 | 1236.98 | 1225.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1226.90 | 1234.97 | 1226.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1226.90 | 1234.97 | 1226.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1225.00 | 1232.97 | 1225.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:45:00 | 1221.80 | 1232.97 | 1225.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1223.00 | 1230.98 | 1225.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:45:00 | 1222.10 | 1230.98 | 1225.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 1209.40 | 1222.77 | 1223.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 1198.50 | 1217.92 | 1220.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 13:15:00 | 1215.00 | 1211.19 | 1216.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 1215.00 | 1211.19 | 1216.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1218.40 | 1212.63 | 1216.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1218.40 | 1212.63 | 1216.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1217.00 | 1213.50 | 1216.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1231.20 | 1213.50 | 1216.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1223.90 | 1215.58 | 1217.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:45:00 | 1207.30 | 1215.66 | 1217.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 1214.50 | 1214.32 | 1216.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1194.60 | 1188.88 | 1188.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 1194.60 | 1188.88 | 1188.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 1219.20 | 1196.69 | 1192.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 15:15:00 | 1210.20 | 1211.55 | 1202.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 09:15:00 | 1206.50 | 1211.55 | 1202.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1195.70 | 1208.38 | 1202.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 1195.70 | 1208.38 | 1202.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1199.70 | 1206.64 | 1201.97 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 15:15:00 | 1199.00 | 1199.74 | 1199.78 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 1203.20 | 1199.08 | 1198.54 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1172.70 | 1193.80 | 1196.19 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1210.60 | 1198.91 | 1198.07 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 14:15:00 | 1186.80 | 1196.65 | 1197.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 10:15:00 | 1172.90 | 1189.03 | 1193.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 1072.70 | 1069.27 | 1084.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 1072.70 | 1069.27 | 1084.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1076.90 | 1071.35 | 1081.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 1076.90 | 1071.35 | 1081.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1062.10 | 1050.78 | 1060.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 1062.10 | 1050.78 | 1060.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1058.60 | 1052.35 | 1060.33 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1070.00 | 1062.64 | 1062.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 1072.00 | 1065.69 | 1063.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 1065.00 | 1065.89 | 1064.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 1065.00 | 1065.89 | 1064.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1065.00 | 1065.89 | 1064.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 1065.00 | 1065.89 | 1064.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1072.50 | 1067.21 | 1064.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 1086.40 | 1067.21 | 1064.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 1077.70 | 1077.19 | 1072.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 12:00:00 | 1077.90 | 1077.19 | 1072.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:45:00 | 1080.50 | 1075.51 | 1072.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1097.90 | 1093.55 | 1085.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1095.40 | 1093.55 | 1085.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1088.80 | 1094.29 | 1087.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1088.80 | 1094.29 | 1087.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1098.60 | 1095.15 | 1088.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 12:00:00 | 1103.50 | 1096.82 | 1089.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 1077.00 | 1091.44 | 1090.22 | SL hit (close<static) qty=1.00 sl=1085.90 alert=retest2 |

### Cycle 181 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1080.10 | 1089.01 | 1089.32 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 1093.10 | 1088.34 | 1087.93 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1074.70 | 1085.29 | 1086.69 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1094.90 | 1086.61 | 1086.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 1108.00 | 1095.05 | 1091.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 1104.60 | 1108.04 | 1100.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 1104.60 | 1108.04 | 1100.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1114.00 | 1109.23 | 1101.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1107.80 | 1109.23 | 1101.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1108.90 | 1109.17 | 1102.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:00:00 | 1117.00 | 1110.73 | 1103.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1085.10 | 1105.81 | 1107.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1085.10 | 1105.81 | 1107.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 1080.20 | 1093.17 | 1100.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 12:15:00 | 1065.20 | 1061.01 | 1068.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 12:15:00 | 1065.20 | 1061.01 | 1068.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1065.20 | 1061.01 | 1068.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 1061.30 | 1061.01 | 1068.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1063.80 | 1061.57 | 1068.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:30:00 | 1066.60 | 1061.57 | 1068.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1069.50 | 1063.15 | 1068.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 1072.00 | 1063.15 | 1068.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1065.00 | 1063.52 | 1068.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1044.50 | 1063.52 | 1068.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:45:00 | 1064.60 | 1058.95 | 1064.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 1072.30 | 1066.27 | 1065.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 1072.30 | 1066.27 | 1065.88 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1062.10 | 1065.34 | 1065.63 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 1071.60 | 1066.70 | 1066.20 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 14:15:00 | 1062.80 | 1065.68 | 1065.87 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 1067.30 | 1066.01 | 1066.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 1070.30 | 1066.87 | 1066.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 13:15:00 | 1068.00 | 1068.17 | 1067.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 13:15:00 | 1068.00 | 1068.17 | 1067.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1068.00 | 1068.17 | 1067.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:00:00 | 1068.00 | 1068.17 | 1067.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1074.90 | 1069.52 | 1067.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 1073.40 | 1069.52 | 1067.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1058.30 | 1067.67 | 1067.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 1058.10 | 1067.67 | 1067.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1054.60 | 1065.06 | 1066.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1046.00 | 1058.12 | 1062.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 1058.30 | 1052.10 | 1056.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 14:15:00 | 1058.30 | 1052.10 | 1056.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1058.30 | 1052.10 | 1056.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 1058.30 | 1052.10 | 1056.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 1070.00 | 1055.68 | 1058.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 1055.80 | 1055.68 | 1058.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 1058.40 | 1033.83 | 1032.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1058.40 | 1033.83 | 1032.86 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 1035.00 | 1042.50 | 1043.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1030.80 | 1037.89 | 1040.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1039.80 | 1037.81 | 1039.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 1039.80 | 1037.81 | 1039.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1039.80 | 1037.81 | 1039.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:45:00 | 1038.10 | 1037.81 | 1039.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1040.80 | 1038.41 | 1039.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 1040.80 | 1038.41 | 1039.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1041.30 | 1038.99 | 1039.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 1041.30 | 1038.99 | 1039.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1044.40 | 1039.98 | 1040.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1044.40 | 1039.98 | 1040.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1038.00 | 1039.58 | 1039.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 1034.80 | 1038.73 | 1039.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1045.70 | 1021.28 | 1018.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1045.70 | 1021.28 | 1018.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 1049.80 | 1026.98 | 1021.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 1026.90 | 1030.21 | 1025.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 1026.90 | 1030.21 | 1025.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1026.90 | 1030.21 | 1025.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 1023.30 | 1030.21 | 1025.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1022.40 | 1028.64 | 1025.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1022.40 | 1028.64 | 1025.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1018.10 | 1026.54 | 1024.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 1018.00 | 1026.54 | 1024.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 13:15:00 | 1016.60 | 1022.40 | 1023.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 15:15:00 | 1010.00 | 1018.62 | 1021.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 1017.20 | 1015.19 | 1018.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 1017.20 | 1015.19 | 1018.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1017.20 | 1015.19 | 1018.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 1014.00 | 1015.19 | 1018.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1017.00 | 1015.55 | 1018.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:15:00 | 1016.30 | 1015.55 | 1018.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1017.60 | 1015.96 | 1018.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:15:00 | 1017.50 | 1015.96 | 1018.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1017.20 | 1016.21 | 1018.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1012.80 | 1016.57 | 1018.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 10:00:00 | 1012.00 | 1015.65 | 1017.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1015.40 | 1016.97 | 1017.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1016.00 | 1016.69 | 1017.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1017.50 | 1016.86 | 1017.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 1017.50 | 1016.86 | 1017.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 1020.80 | 1017.64 | 1017.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 1020.80 | 1017.64 | 1017.52 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1015.70 | 1017.12 | 1017.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1011.40 | 1015.98 | 1016.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 13:15:00 | 1013.50 | 1012.24 | 1014.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 13:15:00 | 1013.50 | 1012.24 | 1014.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1013.50 | 1012.24 | 1014.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 1013.50 | 1012.24 | 1014.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1017.80 | 1013.35 | 1014.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 1017.80 | 1013.35 | 1014.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1015.70 | 1013.82 | 1014.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 1000.80 | 1011.68 | 1013.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 10:45:00 | 1003.00 | 1009.84 | 1012.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 1001.50 | 1008.33 | 1011.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 1002.30 | 1008.33 | 1011.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1010.00 | 1007.49 | 1010.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 1010.00 | 1007.49 | 1010.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1010.90 | 1008.17 | 1010.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1010.90 | 1008.17 | 1010.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1010.00 | 1008.54 | 1010.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 1006.00 | 1005.45 | 1008.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 996.50 | 1006.18 | 1008.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:15:00 | 1004.80 | 1005.46 | 1008.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 12:30:00 | 1007.20 | 1001.39 | 1001.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1000.90 | 1003.43 | 1003.55 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 15:15:00 | 1007.00 | 1003.81 | 1003.59 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1001.90 | 1003.43 | 1003.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 997.70 | 1002.28 | 1002.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 1004.90 | 1000.25 | 1001.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 15:15:00 | 1004.90 | 1000.25 | 1001.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1004.90 | 1000.25 | 1001.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 981.80 | 1000.25 | 1001.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1008.00 | 981.87 | 984.96 | SL hit (close>static) qty=1.00 sl=1004.90 alert=retest2 |

### Cycle 202 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1007.90 | 987.08 | 987.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 1021.00 | 1001.28 | 994.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 1019.80 | 1019.96 | 1011.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:30:00 | 1021.20 | 1019.96 | 1011.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1011.20 | 1020.42 | 1015.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1011.20 | 1020.42 | 1015.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1011.20 | 1018.58 | 1015.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:15:00 | 1005.10 | 1018.58 | 1015.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1018.70 | 1017.83 | 1016.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:15:00 | 1022.00 | 1018.28 | 1016.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 1027.20 | 1018.30 | 1016.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 1011.10 | 1027.47 | 1027.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1011.10 | 1027.47 | 1027.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 15:15:00 | 1003.00 | 1011.37 | 1016.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 1011.00 | 1007.34 | 1012.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:00:00 | 1011.00 | 1007.34 | 1012.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 1000.60 | 1005.99 | 1011.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 997.00 | 1004.79 | 1010.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 998.00 | 1001.98 | 1004.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 1013.50 | 1004.28 | 1005.02 | SL hit (close>static) qty=1.00 sl=1011.50 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 14:15:00 | 1017.80 | 1006.99 | 1006.18 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 1001.00 | 1005.40 | 1005.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 999.50 | 1004.22 | 1005.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 09:15:00 | 1011.10 | 1003.61 | 1004.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1011.10 | 1003.61 | 1004.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1011.10 | 1003.61 | 1004.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1010.50 | 1003.61 | 1004.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1007.90 | 1004.47 | 1004.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:45:00 | 1001.00 | 1004.07 | 1004.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:30:00 | 1000.00 | 1003.56 | 1004.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 1001.50 | 1003.56 | 1004.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:00:00 | 997.80 | 1002.41 | 1003.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 987.60 | 999.45 | 1002.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:30:00 | 1001.60 | 999.45 | 1002.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1000.00 | 999.56 | 1002.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 984.80 | 994.22 | 999.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 1004.80 | 998.87 | 998.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1004.80 | 998.87 | 998.07 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 992.90 | 997.22 | 997.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 991.20 | 996.01 | 996.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 971.60 | 968.44 | 978.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 09:15:00 | 964.90 | 968.95 | 977.45 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:00:00 | 965.20 | 968.20 | 976.34 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 11:00:00 | 965.00 | 967.56 | 975.31 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 12:45:00 | 965.70 | 967.15 | 973.75 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 990.50 | 971.94 | 974.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 990.50 | 971.94 | 974.79 | SL hit (close>ema400) qty=1.00 sl=974.79 alert=retest1 |

### Cycle 208 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 977.00 | 976.63 | 976.62 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 973.40 | 975.99 | 976.32 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 995.40 | 979.76 | 977.97 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 15:15:00 | 970.00 | 978.29 | 978.90 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 982.50 | 979.49 | 979.36 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 971.30 | 977.97 | 978.70 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 15:15:00 | 982.00 | 979.52 | 979.29 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 965.20 | 976.66 | 978.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 955.65 | 972.45 | 975.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 975.30 | 973.02 | 975.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:30:00 | 975.00 | 973.02 | 975.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 971.85 | 972.79 | 975.54 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 987.60 | 979.10 | 977.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1001.00 | 983.48 | 980.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 1007.55 | 1007.68 | 995.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 15:00:00 | 1007.55 | 1007.68 | 995.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 999.20 | 1008.17 | 1003.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 999.20 | 1008.17 | 1003.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1007.85 | 1008.10 | 1004.08 | EMA400 retest candle locked (from upside) |

### Cycle 217 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 988.75 | 1000.46 | 1001.56 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1011.00 | 1001.63 | 1000.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 1023.20 | 1009.64 | 1004.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1022.90 | 1031.59 | 1023.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 1022.90 | 1031.59 | 1023.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1022.90 | 1031.59 | 1023.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 1022.90 | 1031.59 | 1023.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1024.50 | 1030.17 | 1023.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 1023.00 | 1030.17 | 1023.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1019.00 | 1027.19 | 1023.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:45:00 | 1019.05 | 1027.19 | 1023.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1014.15 | 1024.58 | 1022.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:45:00 | 1013.00 | 1024.58 | 1022.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 1013.00 | 1021.21 | 1021.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 997.25 | 1016.42 | 1019.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 11:15:00 | 1014.10 | 1013.51 | 1017.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:45:00 | 1013.55 | 1013.51 | 1017.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 976.80 | 977.08 | 980.56 | EMA400 retest candle locked (from downside) |

### Cycle 220 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 1021.50 | 988.78 | 984.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1058.85 | 1002.79 | 991.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 15:15:00 | 1041.85 | 1044.38 | 1033.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 09:15:00 | 1034.05 | 1044.38 | 1033.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1037.15 | 1042.93 | 1033.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 1053.50 | 1040.94 | 1035.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1024.70 | 1034.63 | 1035.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 09:15:00 | 1024.70 | 1034.63 | 1035.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1003.00 | 1023.74 | 1029.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 936.85 | 935.42 | 948.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 929.50 | 934.43 | 947.29 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 09:15:00 | 929.10 | 937.50 | 944.10 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 922.60 | 921.10 | 929.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 10:30:00 | 915.00 | 919.69 | 928.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:30:00 | 918.85 | 915.64 | 922.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 883.02 | 897.20 | 908.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 882.64 | 897.20 | 908.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 872.91 | 897.20 | 908.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 897.95 | 895.91 | 906.15 | SL hit (close>ema200) qty=0.50 sl=895.91 alert=retest1 |

### Cycle 222 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 812.20 | 806.29 | 805.83 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 786.35 | 802.68 | 804.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 784.05 | 798.95 | 802.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 836.25 | 775.35 | 780.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 836.25 | 775.35 | 780.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 836.25 | 775.35 | 780.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 836.25 | 775.35 | 780.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 873.50 | 794.98 | 788.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 886.90 | 857.13 | 843.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 875.00 | 877.94 | 863.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 875.00 | 877.94 | 863.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 876.00 | 877.65 | 870.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 891.20 | 877.65 | 870.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 09:15:00 | 980.32 | 948.83 | 936.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 959.20 | 980.69 | 982.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 955.30 | 964.74 | 972.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 967.85 | 964.25 | 970.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 11:30:00 | 965.10 | 964.25 | 970.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 967.55 | 964.91 | 970.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 961.60 | 964.46 | 969.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 09:30:00 | 963.60 | 960.94 | 966.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:00:00 | 957.15 | 960.94 | 966.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 977.00 | 964.15 | 967.80 | SL hit (close>static) qty=1.00 sl=975.80 alert=retest2 |

### Cycle 226 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 983.10 | 970.10 | 969.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 994.05 | 974.89 | 971.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 14:30:00 | 482.50 | 2023-05-17 10:15:00 | 467.35 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2023-05-15 15:15:00 | 476.50 | 2023-05-17 10:15:00 | 467.35 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2023-05-18 12:00:00 | 462.25 | 2023-05-22 09:15:00 | 472.20 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2023-05-19 13:30:00 | 466.55 | 2023-05-22 09:15:00 | 472.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-05-24 09:45:00 | 471.45 | 2023-05-24 12:15:00 | 466.95 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-05-24 11:00:00 | 472.10 | 2023-05-24 12:15:00 | 466.95 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-05-25 12:15:00 | 465.75 | 2023-05-25 13:15:00 | 469.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-05-26 09:15:00 | 465.55 | 2023-05-26 10:15:00 | 470.35 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-05-31 09:15:00 | 482.30 | 2023-05-31 11:15:00 | 475.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-06-06 14:00:00 | 472.40 | 2023-06-12 13:15:00 | 475.75 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-06-07 12:30:00 | 471.95 | 2023-06-12 13:15:00 | 475.75 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-06-07 13:15:00 | 472.35 | 2023-06-12 13:15:00 | 475.75 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-06-08 13:00:00 | 472.55 | 2023-06-12 13:15:00 | 475.75 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2023-06-09 10:15:00 | 470.45 | 2023-06-12 14:15:00 | 473.85 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-06-09 12:00:00 | 470.50 | 2023-06-12 14:15:00 | 473.85 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-06-09 12:30:00 | 470.40 | 2023-06-12 14:15:00 | 473.85 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-06-12 10:30:00 | 470.00 | 2023-06-12 14:15:00 | 473.85 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-06-19 10:00:00 | 501.40 | 2023-06-22 09:15:00 | 498.65 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2023-06-20 09:45:00 | 502.80 | 2023-06-22 09:15:00 | 498.65 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-07-04 09:15:00 | 508.55 | 2023-07-06 14:15:00 | 506.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2023-07-04 15:00:00 | 508.95 | 2023-07-06 14:15:00 | 506.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-07-10 11:00:00 | 498.00 | 2023-07-11 09:15:00 | 499.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2023-07-11 12:45:00 | 495.50 | 2023-07-25 09:15:00 | 485.55 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest2 | 2023-07-26 12:15:00 | 475.55 | 2023-07-31 09:15:00 | 482.40 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-07-26 12:45:00 | 475.20 | 2023-07-31 09:15:00 | 482.40 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2023-07-27 12:00:00 | 475.90 | 2023-07-31 09:15:00 | 482.40 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2023-07-27 13:15:00 | 475.25 | 2023-07-31 09:15:00 | 482.40 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-07-28 10:45:00 | 475.70 | 2023-07-31 09:15:00 | 482.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-07-28 11:15:00 | 475.00 | 2023-07-31 09:15:00 | 482.40 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2023-08-04 14:00:00 | 472.35 | 2023-08-08 09:15:00 | 477.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-08-04 14:45:00 | 474.05 | 2023-08-08 09:15:00 | 477.65 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-08-07 10:45:00 | 472.70 | 2023-08-08 09:15:00 | 477.65 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-08-16 09:15:00 | 508.05 | 2023-08-18 15:15:00 | 493.50 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2023-08-16 11:30:00 | 508.15 | 2023-08-18 15:15:00 | 493.50 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2023-08-16 13:00:00 | 508.45 | 2023-08-18 15:15:00 | 493.50 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2023-08-16 13:45:00 | 508.05 | 2023-08-18 15:15:00 | 493.50 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest1 | 2023-08-25 09:15:00 | 542.05 | 2023-08-28 14:15:00 | 527.55 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2023-08-30 14:15:00 | 521.40 | 2023-08-31 09:15:00 | 529.20 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-09-04 12:45:00 | 543.05 | 2023-09-11 09:15:00 | 597.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-22 13:00:00 | 560.00 | 2023-09-28 14:15:00 | 560.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-09-27 09:15:00 | 558.90 | 2023-09-28 14:15:00 | 560.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-10-12 14:30:00 | 678.00 | 2023-10-16 15:15:00 | 667.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2023-10-18 10:30:00 | 667.90 | 2023-10-23 09:15:00 | 634.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:30:00 | 665.15 | 2023-10-23 12:15:00 | 631.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 10:30:00 | 667.90 | 2023-10-25 09:15:00 | 636.75 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2023-10-20 09:30:00 | 665.15 | 2023-10-25 09:15:00 | 636.75 | STOP_HIT | 0.50 | 4.27% |
| BUY | retest2 | 2023-11-01 10:45:00 | 635.30 | 2023-11-16 14:15:00 | 666.25 | STOP_HIT | 1.00 | 4.87% |
| BUY | retest2 | 2023-11-01 12:45:00 | 635.65 | 2023-11-16 14:15:00 | 666.25 | STOP_HIT | 1.00 | 4.81% |
| BUY | retest2 | 2023-11-02 09:15:00 | 637.10 | 2023-11-16 14:15:00 | 666.25 | STOP_HIT | 1.00 | 4.58% |
| BUY | retest2 | 2023-11-02 10:30:00 | 636.35 | 2023-11-16 14:15:00 | 666.25 | STOP_HIT | 1.00 | 4.70% |
| BUY | retest2 | 2023-11-03 09:15:00 | 637.25 | 2023-11-16 14:15:00 | 666.25 | STOP_HIT | 1.00 | 4.55% |
| BUY | retest2 | 2023-11-03 11:15:00 | 635.00 | 2023-11-16 14:15:00 | 666.25 | STOP_HIT | 1.00 | 4.92% |
| BUY | retest2 | 2023-11-03 11:45:00 | 636.00 | 2023-11-16 14:15:00 | 666.25 | STOP_HIT | 1.00 | 4.76% |
| SELL | retest2 | 2023-12-01 13:00:00 | 648.60 | 2023-12-06 14:15:00 | 655.85 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-12-04 10:45:00 | 651.00 | 2023-12-06 14:15:00 | 655.85 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-12-05 11:00:00 | 650.55 | 2023-12-06 14:15:00 | 655.85 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-12-11 12:45:00 | 666.25 | 2023-12-13 09:15:00 | 659.65 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-12-11 15:00:00 | 668.30 | 2023-12-13 09:15:00 | 659.65 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2023-12-12 10:30:00 | 665.00 | 2023-12-13 09:15:00 | 659.65 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-12-12 11:30:00 | 664.25 | 2023-12-13 09:15:00 | 659.65 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-12-18 09:15:00 | 686.20 | 2023-12-20 13:15:00 | 672.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-12-18 10:30:00 | 685.45 | 2023-12-20 13:15:00 | 672.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2023-12-18 12:15:00 | 683.05 | 2023-12-20 13:15:00 | 672.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2023-12-18 14:00:00 | 683.30 | 2023-12-20 13:15:00 | 672.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-12-19 09:15:00 | 691.85 | 2023-12-20 13:15:00 | 672.50 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2023-12-20 14:15:00 | 688.00 | 2023-12-21 13:15:00 | 680.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-12-21 09:15:00 | 686.80 | 2023-12-21 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-12-28 13:00:00 | 707.30 | 2024-01-01 14:15:00 | 778.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-29 09:15:00 | 727.90 | 2024-01-02 09:15:00 | 800.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-12 14:30:00 | 832.50 | 2024-01-16 11:15:00 | 842.30 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-01-15 09:45:00 | 828.05 | 2024-01-16 11:15:00 | 842.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-01-15 14:15:00 | 832.40 | 2024-01-16 11:15:00 | 842.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-01-19 13:15:00 | 825.00 | 2024-01-20 10:15:00 | 830.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-01-19 14:00:00 | 825.00 | 2024-01-20 10:15:00 | 830.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-01-19 14:30:00 | 820.00 | 2024-01-20 10:15:00 | 830.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-02-01 13:15:00 | 878.30 | 2024-02-02 15:15:00 | 869.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-02-14 09:15:00 | 809.45 | 2024-02-15 15:15:00 | 822.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-02-14 12:30:00 | 811.45 | 2024-02-15 15:15:00 | 822.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-02-14 13:00:00 | 811.55 | 2024-02-15 15:15:00 | 822.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-03-05 11:45:00 | 796.60 | 2024-03-05 15:15:00 | 809.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-03-06 09:30:00 | 797.20 | 2024-03-07 11:15:00 | 806.10 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-03-11 11:30:00 | 812.10 | 2024-03-12 09:15:00 | 793.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-03-11 13:00:00 | 812.10 | 2024-03-12 09:15:00 | 793.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-03-11 13:45:00 | 812.10 | 2024-03-12 09:15:00 | 793.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-03-18 13:00:00 | 746.35 | 2024-03-21 11:15:00 | 752.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-03-18 14:45:00 | 743.30 | 2024-03-21 11:15:00 | 752.20 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-03-19 09:30:00 | 745.55 | 2024-03-21 11:15:00 | 752.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-03-27 09:15:00 | 772.90 | 2024-04-08 11:15:00 | 843.65 | TARGET_HIT | 1.00 | 9.15% |
| BUY | retest2 | 2024-03-28 10:45:00 | 766.95 | 2024-04-08 12:15:00 | 850.19 | TARGET_HIT | 1.00 | 10.85% |
| BUY | retest2 | 2024-05-02 12:30:00 | 895.85 | 2024-05-02 15:15:00 | 892.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-05-06 15:15:00 | 878.00 | 2024-05-09 13:15:00 | 834.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 11:00:00 | 876.65 | 2024-05-09 13:15:00 | 832.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 15:15:00 | 878.00 | 2024-05-13 12:15:00 | 828.40 | STOP_HIT | 0.50 | 5.65% |
| SELL | retest2 | 2024-05-07 11:00:00 | 876.65 | 2024-05-13 12:15:00 | 828.40 | STOP_HIT | 0.50 | 5.50% |
| BUY | retest2 | 2024-05-27 09:15:00 | 845.90 | 2024-05-30 09:15:00 | 818.25 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-06-04 09:15:00 | 764.60 | 2024-06-04 11:15:00 | 726.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:30:00 | 783.75 | 2024-06-04 11:15:00 | 744.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 764.60 | 2024-06-04 12:15:00 | 778.40 | STOP_HIT | 0.50 | -1.80% |
| SELL | retest2 | 2024-06-04 10:30:00 | 783.75 | 2024-06-04 12:15:00 | 778.40 | STOP_HIT | 0.50 | 0.68% |
| SELL | retest2 | 2024-06-05 09:15:00 | 786.80 | 2024-06-05 12:15:00 | 801.60 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-06-26 10:15:00 | 857.00 | 2024-07-12 12:15:00 | 900.60 | STOP_HIT | 1.00 | 5.09% |
| BUY | retest2 | 2024-06-28 09:15:00 | 857.35 | 2024-07-12 12:15:00 | 900.60 | STOP_HIT | 1.00 | 5.04% |
| BUY | retest2 | 2024-07-01 09:15:00 | 861.95 | 2024-07-12 12:15:00 | 900.60 | STOP_HIT | 1.00 | 4.48% |
| SELL | retest2 | 2024-07-23 09:15:00 | 885.50 | 2024-07-23 09:15:00 | 900.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-07-25 09:45:00 | 880.10 | 2024-07-25 11:15:00 | 895.30 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-07-30 09:15:00 | 907.75 | 2024-08-05 09:15:00 | 894.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-07-30 14:30:00 | 910.00 | 2024-08-05 09:15:00 | 894.70 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-07-31 11:45:00 | 906.75 | 2024-08-05 09:15:00 | 894.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-07-31 13:45:00 | 910.65 | 2024-08-05 09:15:00 | 894.70 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-08-01 09:15:00 | 914.45 | 2024-08-05 09:15:00 | 894.70 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-08-08 09:15:00 | 864.50 | 2024-08-12 10:15:00 | 896.65 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2024-09-02 09:15:00 | 1071.85 | 2024-09-03 14:15:00 | 1179.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-09-24 12:30:00 | 1238.15 | 2024-09-25 12:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest1 | 2024-09-24 13:00:00 | 1239.55 | 2024-09-25 12:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-09-26 09:30:00 | 1246.00 | 2024-09-30 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-26 10:00:00 | 1238.85 | 2024-09-30 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-26 10:30:00 | 1240.20 | 2024-09-30 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-09-26 12:00:00 | 1237.80 | 2024-09-30 11:15:00 | 1231.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-10-15 09:30:00 | 1081.10 | 2024-10-16 09:15:00 | 1027.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:30:00 | 1081.10 | 2024-10-16 13:15:00 | 1054.85 | STOP_HIT | 0.50 | 2.43% |
| BUY | retest2 | 2024-10-31 10:45:00 | 1030.95 | 2024-11-04 10:15:00 | 1012.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-10-31 11:30:00 | 1034.50 | 2024-11-04 10:15:00 | 1012.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-10-31 14:15:00 | 1034.35 | 2024-11-04 10:15:00 | 1012.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-11-21 12:15:00 | 1006.20 | 2024-12-03 12:15:00 | 1063.75 | STOP_HIT | 1.00 | 5.72% |
| BUY | retest2 | 2024-11-22 09:15:00 | 1010.00 | 2024-12-03 12:15:00 | 1063.75 | STOP_HIT | 1.00 | 5.32% |
| SELL | retest2 | 2024-12-09 12:15:00 | 1065.20 | 2024-12-10 14:15:00 | 1107.70 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2024-12-09 14:00:00 | 1065.85 | 2024-12-10 14:15:00 | 1107.70 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2024-12-10 09:15:00 | 1066.25 | 2024-12-10 14:15:00 | 1107.70 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2024-12-10 11:00:00 | 1068.05 | 2024-12-10 14:15:00 | 1107.70 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2024-12-10 13:00:00 | 1064.00 | 2024-12-10 14:15:00 | 1107.70 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2024-12-18 13:00:00 | 1119.40 | 2024-12-30 13:15:00 | 1126.95 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-01-15 13:00:00 | 956.95 | 2025-01-21 10:15:00 | 909.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 15:00:00 | 957.60 | 2025-01-21 10:15:00 | 909.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:15:00 | 956.95 | 2025-01-21 10:15:00 | 909.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 13:00:00 | 956.95 | 2025-01-23 09:15:00 | 894.15 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2025-01-15 15:00:00 | 957.60 | 2025-01-23 09:15:00 | 894.15 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2025-01-16 11:15:00 | 956.95 | 2025-01-23 09:15:00 | 894.15 | STOP_HIT | 0.50 | 6.56% |
| BUY | retest2 | 2025-02-01 14:45:00 | 903.05 | 2025-02-04 12:15:00 | 880.25 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-02-03 14:45:00 | 904.30 | 2025-02-04 12:15:00 | 880.25 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-02-17 14:45:00 | 831.50 | 2025-02-18 09:15:00 | 830.75 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-03-07 11:15:00 | 1118.00 | 2025-03-13 13:15:00 | 1120.00 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-03-07 13:00:00 | 1113.90 | 2025-03-13 13:15:00 | 1120.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-03-10 09:15:00 | 1113.15 | 2025-03-13 13:15:00 | 1120.00 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-03-10 13:00:00 | 1113.60 | 2025-03-13 13:15:00 | 1120.00 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-03-11 11:45:00 | 1153.25 | 2025-03-13 13:15:00 | 1120.00 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-03-11 13:30:00 | 1124.10 | 2025-03-13 13:15:00 | 1120.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-03-12 10:45:00 | 1123.75 | 2025-03-13 13:15:00 | 1120.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-03-13 09:45:00 | 1126.75 | 2025-03-13 13:15:00 | 1120.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-03-28 12:30:00 | 1145.65 | 2025-04-01 09:15:00 | 1088.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 12:30:00 | 1145.65 | 2025-04-02 10:15:00 | 1114.15 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-04-02 11:30:00 | 1139.55 | 2025-04-04 15:15:00 | 1082.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 14:45:00 | 1141.35 | 2025-04-04 15:15:00 | 1084.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 11:30:00 | 1139.55 | 2025-04-07 12:15:00 | 1027.21 | TARGET_HIT | 0.50 | 9.86% |
| SELL | retest2 | 2025-04-02 14:45:00 | 1141.35 | 2025-04-08 09:15:00 | 1062.45 | STOP_HIT | 0.50 | 6.91% |
| BUY | retest2 | 2025-04-16 15:00:00 | 1131.00 | 2025-04-17 09:15:00 | 1116.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-04-17 09:30:00 | 1128.90 | 2025-04-17 11:15:00 | 1120.70 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-04-17 15:15:00 | 1135.50 | 2025-04-24 12:15:00 | 1137.70 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-04-21 09:45:00 | 1130.30 | 2025-04-24 12:15:00 | 1137.70 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-04-22 10:15:00 | 1150.20 | 2025-04-24 12:15:00 | 1137.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-05-16 10:30:00 | 1137.10 | 2025-05-16 15:15:00 | 1140.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-05-16 15:15:00 | 1140.00 | 2025-05-16 15:15:00 | 1140.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-05-28 13:30:00 | 1180.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1181.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-05-30 09:45:00 | 1180.80 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-05-30 10:15:00 | 1182.10 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-05-30 14:15:00 | 1185.60 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-05-30 15:15:00 | 1193.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1186.60 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-06-02 11:15:00 | 1187.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-06-02 12:15:00 | 1195.80 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-02 14:15:00 | 1194.50 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-06-03 10:00:00 | 1195.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-06-03 15:00:00 | 1194.40 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-06-10 10:30:00 | 1357.00 | 2025-06-11 09:15:00 | 1291.00 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2025-06-20 10:00:00 | 1349.90 | 2025-06-23 10:15:00 | 1312.20 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-06-20 12:30:00 | 1365.00 | 2025-06-23 10:15:00 | 1312.20 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2025-06-30 10:45:00 | 1251.30 | 2025-07-02 09:15:00 | 1188.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-30 11:15:00 | 1248.80 | 2025-07-02 09:15:00 | 1186.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 10:45:00 | 1248.40 | 2025-07-02 09:15:00 | 1185.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-30 10:45:00 | 1251.30 | 2025-07-03 09:15:00 | 1126.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-30 11:15:00 | 1248.80 | 2025-07-03 12:15:00 | 1161.00 | STOP_HIT | 0.50 | 7.03% |
| SELL | retest2 | 2025-07-01 10:45:00 | 1248.40 | 2025-07-03 12:15:00 | 1161.00 | STOP_HIT | 0.50 | 7.00% |
| SELL | retest2 | 2025-07-23 12:45:00 | 1128.00 | 2025-07-28 10:15:00 | 1140.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-24 11:45:00 | 1129.20 | 2025-07-28 10:15:00 | 1140.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-24 15:15:00 | 1127.80 | 2025-07-28 10:15:00 | 1140.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-12 13:15:00 | 1092.30 | 2025-08-13 09:15:00 | 1118.40 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-08-12 14:00:00 | 1093.80 | 2025-08-13 09:15:00 | 1118.40 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-08-25 15:15:00 | 1295.00 | 2025-08-26 10:15:00 | 1271.40 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-09-04 11:15:00 | 1186.20 | 2025-09-08 09:15:00 | 1203.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-05 11:15:00 | 1187.60 | 2025-09-08 09:15:00 | 1203.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-10 14:45:00 | 1213.00 | 2025-09-11 11:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1213.80 | 2025-09-11 11:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-18 12:45:00 | 1207.30 | 2025-09-24 13:15:00 | 1194.60 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2025-09-19 09:30:00 | 1214.50 | 2025-09-24 13:15:00 | 1194.60 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-10-16 12:15:00 | 1086.40 | 2025-10-24 10:15:00 | 1077.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-17 11:30:00 | 1077.70 | 2025-10-24 12:15:00 | 1080.10 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-10-17 12:00:00 | 1077.90 | 2025-10-24 12:15:00 | 1080.10 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-10-17 14:45:00 | 1080.50 | 2025-10-24 12:15:00 | 1080.10 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-10-23 12:00:00 | 1103.50 | 2025-10-24 12:15:00 | 1080.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-11-03 11:00:00 | 1117.00 | 2025-11-06 09:15:00 | 1085.10 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-11-12 09:15:00 | 1044.50 | 2025-11-13 11:15:00 | 1072.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-12 11:45:00 | 1064.60 | 2025-11-13 11:15:00 | 1072.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1055.80 | 2025-11-26 14:15:00 | 1058.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-12-04 13:15:00 | 1034.80 | 2025-12-10 09:15:00 | 1045.70 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1012.80 | 2025-12-16 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-15 10:00:00 | 1012.00 | 2025-12-16 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1015.40 | 2025-12-16 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-16 10:30:00 | 1016.00 | 2025-12-16 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-12-18 09:30:00 | 1000.80 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-18 10:45:00 | 1003.00 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-12-18 11:30:00 | 1001.50 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-18 12:00:00 | 1002.30 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-19 10:30:00 | 1006.00 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-12-19 12:15:00 | 996.50 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-12-19 14:15:00 | 1004.80 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-12-23 12:30:00 | 1007.20 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-12-30 09:15:00 | 981.80 | 2025-12-31 12:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-01-06 15:15:00 | 1022.00 | 2026-01-09 09:15:00 | 1011.10 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-01-07 10:15:00 | 1027.20 | 2026-01-09 09:15:00 | 1011.10 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-01-13 15:15:00 | 997.00 | 2026-01-16 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-01-16 13:00:00 | 998.00 | 2026-01-16 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-20 11:45:00 | 1001.00 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-01-20 12:30:00 | 1000.00 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-01-20 13:00:00 | 1001.50 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-01-20 14:00:00 | 997.80 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-01-21 10:30:00 | 984.80 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest1 | 2026-01-28 09:15:00 | 964.90 | 2026-01-28 14:15:00 | 990.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest1 | 2026-01-28 10:00:00 | 965.20 | 2026-01-28 14:15:00 | 990.50 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest1 | 2026-01-28 11:00:00 | 965.00 | 2026-01-28 14:15:00 | 990.50 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest1 | 2026-01-28 12:45:00 | 965.70 | 2026-01-28 14:15:00 | 990.50 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-01-29 09:15:00 | 981.10 | 2026-01-29 10:15:00 | 977.00 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2026-02-24 15:15:00 | 1053.50 | 2026-02-26 09:15:00 | 1024.70 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest1 | 2026-03-06 10:45:00 | 929.50 | 2026-03-12 09:15:00 | 883.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-09 09:15:00 | 929.10 | 2026-03-12 09:15:00 | 882.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 10:30:00 | 915.00 | 2026-03-12 09:15:00 | 872.91 | PARTIAL | 0.50 | 4.60% |
| SELL | retest1 | 2026-03-06 10:45:00 | 929.50 | 2026-03-12 11:15:00 | 897.95 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest1 | 2026-03-09 09:15:00 | 929.10 | 2026-03-12 11:15:00 | 897.95 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-03-10 10:30:00 | 915.00 | 2026-03-12 11:15:00 | 897.95 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2026-03-11 09:30:00 | 918.85 | 2026-03-13 10:15:00 | 869.25 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2026-03-11 09:30:00 | 918.85 | 2026-03-16 09:15:00 | 823.50 | TARGET_HIT | 0.50 | 10.38% |
| BUY | retest2 | 2026-04-10 09:15:00 | 891.20 | 2026-04-23 09:15:00 | 980.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 15:00:00 | 961.60 | 2026-05-04 10:15:00 | 977.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-05-04 09:30:00 | 963.60 | 2026-05-04 10:15:00 | 977.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-05-04 10:00:00 | 957.15 | 2026-05-04 10:15:00 | 977.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-05-04 15:15:00 | 959.50 | 2026-05-05 10:15:00 | 983.10 | STOP_HIT | 1.00 | -2.46% |
