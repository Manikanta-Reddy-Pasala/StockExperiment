# DLF Ltd. (DLF)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 606.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 208 |
| ALERT1 | 151 |
| ALERT2 | 150 |
| ALERT2_SKIP | 73 |
| ALERT3 | 406 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 150 |
| PARTIAL | 11 |
| TARGET_HIT | 7 |
| STOP_HIT | 151 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 164 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 50 / 114
- **Target hits / Stop hits / Partials:** 2 / 151 / 11
- **Avg / median % per leg:** 0.14% / -0.76%
- **Sum % (uncompounded):** 22.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 85 | 28 | 32.9% | 1 | 81 | 3 | 0.40% | 34.3% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 0 | 6 | 3 | 3.10% | 27.9% |
| BUY @ 3rd Alert (retest2) | 76 | 22 | 28.9% | 1 | 75 | 0 | 0.08% | 6.4% |
| SELL (all) | 79 | 22 | 27.8% | 1 | 70 | 8 | -0.15% | -11.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.77% | -1.8% |
| SELL @ 3rd Alert (retest2) | 78 | 22 | 28.2% | 1 | 69 | 8 | -0.13% | -10.0% |
| retest1 (combined) | 10 | 6 | 60.0% | 0 | 7 | 3 | 2.62% | 26.2% |
| retest2 (combined) | 154 | 44 | 28.6% | 2 | 144 | 8 | -0.02% | -3.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 15:15:00 | 459.50 | 462.98 | 463.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 457.30 | 461.84 | 462.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 10:15:00 | 462.65 | 462.01 | 462.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 10:15:00 | 462.65 | 462.01 | 462.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 462.65 | 462.01 | 462.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 11:00:00 | 462.65 | 462.01 | 462.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 466.60 | 462.92 | 462.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 12:00:00 | 466.60 | 462.92 | 462.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 12:15:00 | 466.35 | 463.61 | 463.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 13:15:00 | 470.40 | 464.97 | 463.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 09:15:00 | 471.70 | 474.21 | 470.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 471.70 | 474.21 | 470.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 471.70 | 474.21 | 470.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 10:00:00 | 471.70 | 474.21 | 470.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 472.15 | 473.80 | 471.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-23 11:30:00 | 474.20 | 473.56 | 471.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 14:15:00 | 467.80 | 472.03 | 471.04 | SL hit (close<static) qty=1.00 sl=470.55 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 09:15:00 | 465.95 | 470.23 | 470.36 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 10:15:00 | 477.05 | 469.22 | 469.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 11:15:00 | 479.05 | 471.19 | 470.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 10:15:00 | 474.65 | 475.78 | 473.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 10:15:00 | 474.65 | 475.78 | 473.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 474.65 | 475.78 | 473.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:00:00 | 474.65 | 475.78 | 473.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 11:15:00 | 473.30 | 475.29 | 473.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:45:00 | 472.20 | 475.29 | 473.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 12:15:00 | 474.50 | 475.13 | 473.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 13:15:00 | 475.35 | 475.13 | 473.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 09:30:00 | 476.75 | 476.89 | 474.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 12:00:00 | 475.40 | 477.91 | 477.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 12:45:00 | 475.25 | 477.33 | 476.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 14:15:00 | 474.45 | 476.27 | 476.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 474.45 | 476.27 | 476.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 09:15:00 | 469.85 | 474.78 | 475.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 474.40 | 472.48 | 473.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 474.40 | 472.48 | 473.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 474.40 | 472.48 | 473.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 474.40 | 472.48 | 473.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 474.35 | 472.85 | 474.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:15:00 | 477.00 | 472.85 | 474.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 476.15 | 473.51 | 474.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:45:00 | 478.00 | 473.51 | 474.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 475.45 | 473.90 | 474.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 11:30:00 | 473.65 | 473.78 | 474.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 12:00:00 | 473.30 | 473.78 | 474.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-01 13:15:00 | 475.85 | 474.54 | 474.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 13:15:00 | 475.85 | 474.54 | 474.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 14:15:00 | 477.45 | 475.12 | 474.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 09:15:00 | 493.30 | 494.75 | 492.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 493.30 | 494.75 | 492.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 493.30 | 494.75 | 492.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 09:45:00 | 490.50 | 494.75 | 492.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 492.25 | 494.25 | 492.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:00:00 | 492.25 | 494.25 | 492.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 482.55 | 491.91 | 491.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 482.55 | 491.91 | 491.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 483.65 | 490.26 | 490.89 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 13:15:00 | 490.75 | 488.95 | 488.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 10:15:00 | 492.00 | 490.14 | 489.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 10:15:00 | 502.00 | 502.98 | 499.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 11:00:00 | 502.00 | 502.98 | 499.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 497.85 | 501.95 | 499.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:00:00 | 497.85 | 501.95 | 499.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 495.30 | 500.62 | 498.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 13:00:00 | 495.30 | 500.62 | 498.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-06-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 14:15:00 | 489.30 | 497.31 | 497.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 12:15:00 | 486.30 | 492.89 | 494.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 10:15:00 | 490.55 | 489.13 | 491.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-20 11:00:00 | 490.55 | 489.13 | 491.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 489.30 | 489.17 | 491.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 11:30:00 | 491.20 | 489.17 | 491.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 490.60 | 489.18 | 490.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:30:00 | 491.80 | 489.18 | 490.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 488.05 | 488.95 | 490.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:30:00 | 490.75 | 488.95 | 490.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 486.40 | 478.22 | 479.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 486.40 | 478.22 | 479.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 484.75 | 479.52 | 479.55 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 11:15:00 | 484.10 | 480.44 | 479.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 485.55 | 482.79 | 481.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 488.45 | 488.81 | 486.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 488.45 | 488.81 | 486.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 488.45 | 488.81 | 486.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 10:00:00 | 488.45 | 488.81 | 486.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 490.85 | 492.02 | 489.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 490.85 | 492.02 | 489.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 495.20 | 492.66 | 490.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 15:15:00 | 503.00 | 493.38 | 491.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 15:15:00 | 500.10 | 502.55 | 502.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 15:15:00 | 500.10 | 502.55 | 502.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 15:15:00 | 498.60 | 501.00 | 501.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 09:15:00 | 496.85 | 494.35 | 496.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 09:15:00 | 496.85 | 494.35 | 496.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 496.85 | 494.35 | 496.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 09:45:00 | 498.75 | 494.35 | 496.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 496.50 | 494.78 | 496.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 13:30:00 | 494.45 | 494.51 | 496.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-14 09:30:00 | 493.70 | 493.95 | 495.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-14 11:30:00 | 493.95 | 493.86 | 495.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 09:30:00 | 494.50 | 493.51 | 494.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 494.95 | 493.80 | 494.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:45:00 | 496.20 | 493.80 | 494.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 497.30 | 494.50 | 494.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 12:00:00 | 497.30 | 494.50 | 494.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 495.00 | 494.60 | 494.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-17 13:15:00 | 496.90 | 495.06 | 494.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 13:15:00 | 496.90 | 495.06 | 494.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 14:15:00 | 498.00 | 495.65 | 495.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 496.70 | 497.33 | 496.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-18 12:00:00 | 496.70 | 497.33 | 496.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 494.95 | 496.86 | 496.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 13:00:00 | 494.95 | 496.86 | 496.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 13:15:00 | 493.95 | 496.28 | 495.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 13:30:00 | 493.90 | 496.28 | 495.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 15:15:00 | 494.95 | 495.72 | 495.77 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 501.00 | 496.77 | 496.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 11:15:00 | 502.55 | 498.62 | 497.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 09:15:00 | 497.85 | 502.54 | 501.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 09:15:00 | 497.85 | 502.54 | 501.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 497.85 | 502.54 | 501.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:30:00 | 497.20 | 502.54 | 501.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 10:15:00 | 500.50 | 502.13 | 501.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 11:15:00 | 503.50 | 502.13 | 501.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 13:00:00 | 502.30 | 502.07 | 501.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 13:45:00 | 502.30 | 502.13 | 501.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 09:15:00 | 504.10 | 501.69 | 501.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 501.50 | 501.65 | 501.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-24 11:15:00 | 499.30 | 500.86 | 501.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 499.30 | 500.86 | 501.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 13:15:00 | 494.00 | 498.84 | 500.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 10:15:00 | 498.25 | 497.45 | 498.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 10:15:00 | 498.25 | 497.45 | 498.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 498.25 | 497.45 | 498.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:45:00 | 497.60 | 497.45 | 498.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 492.05 | 496.37 | 498.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 12:15:00 | 489.95 | 496.37 | 498.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 11:15:00 | 505.50 | 492.04 | 491.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 11:15:00 | 505.50 | 492.04 | 491.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 14:15:00 | 510.20 | 499.38 | 495.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 09:15:00 | 504.00 | 512.97 | 509.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 09:15:00 | 504.00 | 512.97 | 509.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 504.00 | 512.97 | 509.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:00:00 | 504.00 | 512.97 | 509.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 499.70 | 510.32 | 508.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:00:00 | 499.70 | 510.32 | 508.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2023-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 12:15:00 | 500.85 | 507.16 | 507.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 10:15:00 | 496.05 | 501.74 | 504.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 491.70 | 489.60 | 493.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 491.70 | 489.60 | 493.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 491.70 | 489.60 | 493.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 10:00:00 | 491.70 | 489.60 | 493.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 488.90 | 489.80 | 491.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 11:15:00 | 483.00 | 488.48 | 489.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 10:15:00 | 484.90 | 487.60 | 488.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 12:00:00 | 484.25 | 486.46 | 488.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 15:00:00 | 485.30 | 485.84 | 487.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 489.00 | 486.47 | 487.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 10:00:00 | 489.00 | 486.47 | 487.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 485.60 | 486.30 | 487.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-10 14:15:00 | 491.65 | 487.66 | 487.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 14:15:00 | 491.65 | 487.66 | 487.58 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 12:15:00 | 486.35 | 487.68 | 487.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 482.55 | 486.30 | 487.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 11:15:00 | 472.20 | 472.02 | 476.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 12:00:00 | 472.20 | 472.02 | 476.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 13:15:00 | 475.90 | 473.16 | 476.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:45:00 | 477.05 | 473.16 | 476.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 477.05 | 473.94 | 476.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 15:00:00 | 477.05 | 473.94 | 476.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 477.65 | 474.68 | 476.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:15:00 | 480.90 | 474.68 | 476.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 474.15 | 475.87 | 476.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 13:00:00 | 473.30 | 475.09 | 475.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 09:15:00 | 478.50 | 475.66 | 475.84 | SL hit (close>static) qty=1.00 sl=477.45 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 10:15:00 | 478.50 | 476.23 | 476.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 14:15:00 | 481.05 | 478.01 | 477.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 480.25 | 481.25 | 479.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 14:15:00 | 480.25 | 481.25 | 479.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 480.25 | 481.25 | 479.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 14:45:00 | 480.75 | 481.25 | 479.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 479.75 | 480.75 | 479.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:00:00 | 479.75 | 480.75 | 479.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 480.70 | 480.74 | 479.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 11:15:00 | 481.50 | 480.74 | 479.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 486.80 | 481.30 | 480.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 09:45:00 | 481.85 | 483.57 | 482.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 10:15:00 | 474.60 | 481.78 | 481.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 474.60 | 481.78 | 481.93 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 483.50 | 480.81 | 480.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 493.60 | 483.88 | 482.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 12:15:00 | 504.75 | 504.80 | 501.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-01 12:30:00 | 504.45 | 504.80 | 501.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 517.20 | 518.89 | 515.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 515.45 | 518.89 | 515.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 516.90 | 518.10 | 516.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:00:00 | 516.90 | 518.10 | 516.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 519.15 | 518.31 | 516.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 11:15:00 | 526.50 | 518.31 | 516.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 12:15:00 | 525.70 | 533.10 | 533.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 525.70 | 533.10 | 533.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 522.45 | 527.76 | 530.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 528.40 | 527.40 | 529.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 11:45:00 | 527.30 | 527.40 | 529.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 527.20 | 527.36 | 529.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:30:00 | 528.80 | 527.36 | 529.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 529.40 | 527.85 | 529.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 529.40 | 527.85 | 529.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 528.00 | 527.88 | 529.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 536.35 | 527.88 | 529.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 537.15 | 529.73 | 529.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 538.35 | 529.73 | 529.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 538.25 | 531.43 | 530.70 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 15:15:00 | 530.95 | 532.06 | 532.21 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 532.80 | 532.41 | 532.36 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 530.95 | 532.12 | 532.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 12:15:00 | 529.40 | 531.57 | 531.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 10:15:00 | 519.55 | 518.38 | 521.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 11:00:00 | 519.55 | 518.38 | 521.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 12:15:00 | 519.65 | 517.93 | 519.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 14:00:00 | 517.75 | 517.89 | 519.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 14:15:00 | 522.10 | 518.73 | 519.59 | SL hit (close>static) qty=1.00 sl=520.90 alert=retest2 |

### Cycle 28 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 522.30 | 520.13 | 520.11 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 09:15:00 | 517.75 | 520.38 | 520.49 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 521.35 | 520.57 | 520.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 12:15:00 | 522.55 | 521.01 | 520.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 524.20 | 524.91 | 523.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 13:00:00 | 524.20 | 524.91 | 523.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 522.45 | 524.42 | 523.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 522.45 | 524.42 | 523.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 519.50 | 523.43 | 522.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 519.50 | 523.43 | 522.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 521.95 | 523.14 | 522.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 531.55 | 523.14 | 522.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 12:15:00 | 517.40 | 527.87 | 528.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 517.40 | 527.87 | 528.33 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 11:15:00 | 527.90 | 527.70 | 527.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 12:15:00 | 532.05 | 528.57 | 528.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 12:15:00 | 537.65 | 539.40 | 535.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-09 13:00:00 | 537.65 | 539.40 | 535.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 574.05 | 567.29 | 562.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 10:30:00 | 575.05 | 569.03 | 563.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 09:15:00 | 565.85 | 566.84 | 566.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 09:15:00 | 565.85 | 566.84 | 566.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 563.00 | 566.08 | 566.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 12:15:00 | 566.65 | 565.75 | 566.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 12:15:00 | 566.65 | 565.75 | 566.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 566.65 | 565.75 | 566.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 13:00:00 | 566.65 | 565.75 | 566.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 13:15:00 | 568.95 | 566.39 | 566.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 14:00:00 | 568.95 | 566.39 | 566.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 567.05 | 566.52 | 566.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 09:15:00 | 562.70 | 566.56 | 566.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 534.57 | 542.89 | 550.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 15:15:00 | 522.80 | 522.34 | 529.83 | SL hit (close>ema200) qty=0.50 sl=522.34 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 09:15:00 | 541.00 | 533.00 | 532.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 10:15:00 | 541.55 | 534.71 | 532.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 10:15:00 | 570.50 | 572.51 | 565.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-02 11:00:00 | 570.50 | 572.51 | 565.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 593.20 | 592.54 | 587.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 597.15 | 592.03 | 589.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:45:00 | 598.30 | 593.37 | 590.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 11:45:00 | 598.05 | 594.42 | 591.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 12:30:00 | 598.00 | 595.35 | 592.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 599.20 | 600.04 | 597.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:45:00 | 595.35 | 600.04 | 597.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 625.00 | 628.36 | 625.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 625.00 | 628.36 | 625.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 627.60 | 628.21 | 625.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 15:15:00 | 628.15 | 627.35 | 625.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 13:00:00 | 629.10 | 630.93 | 629.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 14:15:00 | 629.95 | 632.28 | 632.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 14:15:00 | 629.95 | 632.28 | 632.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 620.15 | 629.61 | 631.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 628.95 | 628.27 | 629.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 14:15:00 | 628.95 | 628.27 | 629.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 628.95 | 628.27 | 629.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 15:00:00 | 628.95 | 628.27 | 629.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 627.25 | 628.07 | 629.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 632.35 | 628.07 | 629.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 635.70 | 629.59 | 630.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:45:00 | 636.00 | 629.59 | 630.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 634.30 | 630.53 | 630.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 14:15:00 | 638.80 | 633.49 | 631.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 628.65 | 632.93 | 632.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 628.65 | 632.93 | 632.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 628.65 | 632.93 | 632.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 629.25 | 632.93 | 632.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 627.10 | 631.76 | 631.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 11:00:00 | 627.10 | 631.76 | 631.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 11:15:00 | 625.75 | 630.56 | 631.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 12:15:00 | 625.10 | 629.47 | 630.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 09:15:00 | 628.20 | 627.90 | 629.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 628.20 | 627.90 | 629.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 628.20 | 627.90 | 629.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 15:15:00 | 626.95 | 629.18 | 629.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 637.50 | 630.49 | 630.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 637.50 | 630.49 | 630.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 646.80 | 633.75 | 631.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 645.20 | 645.94 | 640.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 11:15:00 | 645.20 | 645.94 | 640.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 645.20 | 645.94 | 640.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 643.00 | 645.94 | 640.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 650.25 | 652.30 | 648.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 09:15:00 | 661.70 | 650.54 | 649.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 11:15:00 | 654.95 | 652.20 | 650.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 12:00:00 | 655.30 | 652.82 | 650.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 12:15:00 | 646.25 | 651.50 | 650.39 | SL hit (close<static) qty=1.00 sl=647.45 alert=retest2 |

### Cycle 39 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 680.55 | 693.73 | 694.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 674.65 | 689.91 | 692.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 693.50 | 686.08 | 688.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 14:15:00 | 693.50 | 686.08 | 688.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 693.50 | 686.08 | 688.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 15:00:00 | 693.50 | 686.08 | 688.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 693.25 | 687.51 | 689.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 697.75 | 687.51 | 689.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 701.45 | 690.30 | 690.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 716.90 | 701.55 | 696.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 15:15:00 | 717.00 | 717.81 | 712.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 09:30:00 | 716.80 | 717.94 | 713.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 725.15 | 725.03 | 722.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 09:30:00 | 723.25 | 725.03 | 722.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 723.20 | 726.06 | 723.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 15:00:00 | 723.20 | 726.06 | 723.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 722.00 | 725.25 | 723.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:15:00 | 720.35 | 725.25 | 723.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 719.40 | 724.08 | 723.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:15:00 | 714.15 | 724.08 | 723.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 708.60 | 720.98 | 722.04 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 748.45 | 722.62 | 720.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 10:15:00 | 758.65 | 729.83 | 723.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 12:15:00 | 755.75 | 755.82 | 744.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-05 13:00:00 | 755.75 | 755.82 | 744.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 802.70 | 803.46 | 799.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 15:00:00 | 802.70 | 803.46 | 799.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 795.65 | 802.05 | 799.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:00:00 | 795.65 | 802.05 | 799.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 794.70 | 800.58 | 799.07 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 785.50 | 797.56 | 797.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 782.30 | 794.51 | 796.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 782.00 | 776.68 | 780.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 782.00 | 776.68 | 780.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 782.00 | 776.68 | 780.31 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 15:15:00 | 787.90 | 782.42 | 781.98 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 09:15:00 | 773.00 | 780.53 | 781.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 10:15:00 | 764.40 | 773.05 | 776.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 747.40 | 746.64 | 755.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 15:00:00 | 747.40 | 746.64 | 755.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 761.70 | 749.87 | 755.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:45:00 | 751.25 | 750.70 | 755.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 10:15:00 | 763.90 | 757.34 | 756.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 763.90 | 757.34 | 756.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 12:15:00 | 769.50 | 760.70 | 758.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 15:15:00 | 796.80 | 798.05 | 791.84 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:15:00 | 802.95 | 798.05 | 791.84 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 13:45:00 | 801.50 | 801.05 | 795.99 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 795.85 | 800.01 | 795.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-02 14:15:00 | 795.85 | 800.01 | 795.98 | SL hit (close<ema400) qty=1.00 sl=795.98 alert=retest1 |

### Cycle 47 — SELL (started 2024-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 15:15:00 | 787.50 | 795.56 | 795.77 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 12:15:00 | 798.35 | 796.16 | 795.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 09:15:00 | 832.90 | 803.70 | 799.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 14:15:00 | 830.20 | 831.25 | 822.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-08 14:45:00 | 829.80 | 831.25 | 822.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 829.50 | 830.94 | 823.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 829.50 | 830.94 | 823.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 816.75 | 828.10 | 822.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 816.75 | 828.10 | 822.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 822.65 | 827.01 | 822.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 12:30:00 | 829.00 | 828.18 | 823.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-12 11:15:00 | 812.10 | 823.75 | 823.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 11:15:00 | 812.10 | 823.75 | 823.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 14:15:00 | 805.00 | 817.24 | 820.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 821.20 | 816.03 | 819.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 10:15:00 | 821.20 | 816.03 | 819.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 821.20 | 816.03 | 819.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:00:00 | 821.20 | 816.03 | 819.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 812.50 | 815.32 | 818.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 12:15:00 | 811.40 | 815.32 | 818.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-13 13:15:00 | 825.15 | 817.85 | 819.06 | SL hit (close>static) qty=1.00 sl=822.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 14:15:00 | 829.50 | 820.18 | 820.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 12:15:00 | 833.00 | 826.91 | 823.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 15:15:00 | 858.85 | 859.57 | 850.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 09:15:00 | 856.85 | 859.57 | 850.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 848.65 | 857.38 | 850.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:00:00 | 848.65 | 857.38 | 850.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 849.55 | 855.82 | 850.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:45:00 | 847.00 | 855.82 | 850.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 850.05 | 854.66 | 850.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 12:00:00 | 850.05 | 854.66 | 850.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 12:15:00 | 852.75 | 854.28 | 850.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 859.55 | 851.91 | 850.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-29 09:15:00 | 887.45 | 899.89 | 901.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 887.45 | 899.89 | 901.58 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 905.35 | 901.34 | 901.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 15:15:00 | 916.00 | 906.46 | 903.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 13:15:00 | 928.35 | 931.69 | 925.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 13:45:00 | 927.50 | 931.69 | 925.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 918.35 | 928.84 | 925.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 918.35 | 928.84 | 925.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 921.60 | 927.40 | 925.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:30:00 | 920.65 | 927.40 | 925.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 917.80 | 923.22 | 923.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 908.55 | 918.08 | 920.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 841.20 | 840.65 | 862.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:45:00 | 838.30 | 840.65 | 862.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 834.20 | 828.24 | 834.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 13:30:00 | 833.35 | 828.24 | 834.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 835.65 | 829.72 | 834.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:30:00 | 835.80 | 829.72 | 834.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 15:15:00 | 836.50 | 831.08 | 834.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 09:15:00 | 839.00 | 831.08 | 834.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 829.90 | 831.55 | 834.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 11:30:00 | 823.35 | 830.46 | 833.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 15:00:00 | 823.15 | 828.67 | 832.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 09:45:00 | 816.75 | 825.07 | 829.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 14:15:00 | 823.80 | 823.11 | 827.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 825.50 | 823.59 | 826.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 15:15:00 | 828.00 | 823.59 | 826.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 828.00 | 824.47 | 827.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:15:00 | 836.80 | 824.47 | 827.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 845.80 | 828.74 | 828.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 845.80 | 828.74 | 828.75 | SL hit (close>static) qty=1.00 sl=836.45 alert=retest2 |

### Cycle 54 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 847.20 | 832.43 | 830.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 854.45 | 839.48 | 834.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 15:15:00 | 879.00 | 879.83 | 871.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:15:00 | 886.70 | 879.83 | 871.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 10:00:00 | 887.15 | 881.29 | 873.17 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 11:00:00 | 886.85 | 882.41 | 874.41 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 09:15:00 | 931.04 | 904.31 | 889.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 09:15:00 | 931.51 | 904.31 | 889.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 09:15:00 | 931.19 | 904.31 | 889.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-02 09:15:00 | 932.60 | 936.55 | 917.06 | SL hit (close<ema200) qty=0.50 sl=936.55 alert=retest1 |

### Cycle 55 — SELL (started 2024-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 13:15:00 | 907.45 | 920.38 | 920.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 15:15:00 | 903.55 | 914.40 | 917.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 902.40 | 896.04 | 904.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-05 10:00:00 | 902.40 | 896.04 | 904.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 903.55 | 897.54 | 903.98 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 917.50 | 906.32 | 906.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 10:15:00 | 921.05 | 909.27 | 907.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 10:15:00 | 916.40 | 918.02 | 913.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 11:00:00 | 916.40 | 918.02 | 913.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 914.30 | 917.33 | 914.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 12:30:00 | 912.40 | 917.33 | 914.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 904.35 | 914.73 | 913.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:00:00 | 904.35 | 914.73 | 913.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 911.10 | 914.01 | 913.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 09:15:00 | 912.80 | 913.21 | 912.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 14:15:00 | 912.50 | 914.00 | 913.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 14:15:00 | 909.45 | 913.09 | 913.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 14:15:00 | 909.45 | 913.09 | 913.24 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 11:15:00 | 919.45 | 914.06 | 913.61 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 909.55 | 912.83 | 913.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 15:15:00 | 908.00 | 911.43 | 912.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 12:15:00 | 878.70 | 877.32 | 884.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 12:30:00 | 877.85 | 877.32 | 884.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 857.20 | 855.72 | 864.17 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 893.30 | 869.03 | 867.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 904.15 | 893.09 | 888.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 887.40 | 899.75 | 895.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 09:15:00 | 887.40 | 899.75 | 895.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 887.40 | 899.75 | 895.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:00:00 | 887.40 | 899.75 | 895.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 886.70 | 897.14 | 894.36 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 12:15:00 | 883.50 | 891.84 | 892.27 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 12:15:00 | 906.10 | 892.56 | 891.67 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 884.50 | 891.45 | 891.55 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 898.50 | 892.53 | 891.88 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 10:15:00 | 885.10 | 891.51 | 891.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 872.55 | 887.72 | 890.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 895.00 | 883.32 | 885.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 895.00 | 883.32 | 885.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 895.00 | 883.32 | 885.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 895.00 | 883.32 | 885.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 893.50 | 885.35 | 886.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 12:30:00 | 886.80 | 885.28 | 886.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 842.46 | 867.09 | 876.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 12:15:00 | 859.00 | 858.46 | 866.42 | SL hit (close>ema200) qty=0.50 sl=858.46 alert=retest2 |

### Cycle 66 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 844.30 | 839.49 | 838.85 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 10:15:00 | 831.90 | 837.97 | 838.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 11:15:00 | 828.55 | 836.09 | 837.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 09:15:00 | 837.45 | 831.95 | 834.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 837.45 | 831.95 | 834.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 837.45 | 831.95 | 834.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:30:00 | 836.05 | 831.95 | 834.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 839.55 | 833.47 | 834.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:30:00 | 839.05 | 833.47 | 834.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 838.40 | 834.46 | 835.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:45:00 | 832.40 | 833.60 | 834.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 840.30 | 834.94 | 835.18 | SL hit (close>static) qty=1.00 sl=839.90 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 846.05 | 837.16 | 836.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 849.15 | 840.57 | 837.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 851.25 | 851.28 | 847.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 14:00:00 | 851.25 | 851.28 | 847.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 846.40 | 850.92 | 848.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:00:00 | 846.40 | 850.92 | 848.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 843.55 | 849.44 | 848.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:45:00 | 844.95 | 849.44 | 848.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 13:15:00 | 840.80 | 847.72 | 847.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 10:15:00 | 835.95 | 840.08 | 842.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 841.65 | 839.68 | 841.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 13:15:00 | 841.65 | 839.68 | 841.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 841.65 | 839.68 | 841.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:15:00 | 844.85 | 839.68 | 841.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 843.60 | 840.47 | 841.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:30:00 | 837.20 | 838.82 | 840.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 839.90 | 820.43 | 819.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 839.90 | 820.43 | 819.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 868.60 | 844.54 | 832.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 839.40 | 847.43 | 836.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 839.40 | 847.43 | 836.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 839.40 | 847.43 | 836.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 829.65 | 847.43 | 836.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 784.50 | 834.84 | 831.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 784.50 | 834.84 | 831.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 738.80 | 815.63 | 823.11 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 832.05 | 801.66 | 800.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 838.00 | 824.46 | 814.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 10:15:00 | 854.85 | 855.84 | 848.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:45:00 | 854.25 | 855.84 | 848.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 861.00 | 874.52 | 872.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 864.70 | 874.52 | 872.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 859.90 | 871.60 | 871.61 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 873.50 | 868.64 | 868.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 13:15:00 | 873.90 | 869.69 | 869.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 866.40 | 870.37 | 869.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 866.40 | 870.37 | 869.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 866.40 | 870.37 | 869.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:45:00 | 867.65 | 870.37 | 869.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 867.10 | 869.71 | 869.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 868.30 | 869.71 | 869.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 868.05 | 869.38 | 869.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:15:00 | 865.20 | 869.38 | 869.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 859.90 | 867.49 | 868.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 855.20 | 862.36 | 865.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 827.45 | 827.34 | 837.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 10:15:00 | 833.60 | 827.34 | 837.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 824.85 | 821.81 | 826.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 824.85 | 821.81 | 826.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 820.70 | 821.59 | 825.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 819.10 | 821.59 | 825.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 827.80 | 823.97 | 825.05 | SL hit (close>static) qty=1.00 sl=826.50 alert=retest2 |

### Cycle 76 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 837.20 | 827.39 | 826.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 15:15:00 | 842.00 | 835.64 | 832.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 836.85 | 836.99 | 833.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 10:45:00 | 836.45 | 836.99 | 833.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 834.10 | 837.63 | 835.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 835.05 | 837.63 | 835.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 832.00 | 836.50 | 835.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:00:00 | 832.00 | 836.50 | 835.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 834.25 | 835.72 | 835.02 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 15:15:00 | 832.20 | 834.25 | 834.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 829.05 | 832.87 | 833.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 14:15:00 | 835.20 | 832.98 | 833.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 14:15:00 | 835.20 | 832.98 | 833.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 835.20 | 832.98 | 833.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:45:00 | 835.60 | 832.98 | 833.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 835.00 | 833.38 | 833.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 837.25 | 833.38 | 833.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 833.55 | 833.67 | 833.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:15:00 | 834.50 | 833.67 | 833.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 835.70 | 834.08 | 833.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 13:15:00 | 837.65 | 835.26 | 834.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 830.90 | 835.77 | 835.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 10:15:00 | 830.90 | 835.77 | 835.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 830.90 | 835.77 | 835.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 828.65 | 835.77 | 835.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 834.15 | 835.44 | 835.07 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 833.10 | 834.78 | 834.82 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 836.60 | 835.14 | 834.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 838.95 | 835.88 | 835.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 11:15:00 | 833.50 | 835.44 | 835.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 11:15:00 | 833.50 | 835.44 | 835.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 833.50 | 835.44 | 835.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 833.50 | 835.44 | 835.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 12:15:00 | 833.45 | 835.04 | 835.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 829.35 | 832.82 | 833.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 829.05 | 827.05 | 829.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 829.05 | 827.05 | 829.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 829.05 | 827.05 | 829.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 829.05 | 827.05 | 829.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 827.45 | 827.13 | 829.75 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 839.65 | 832.59 | 831.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 848.90 | 835.85 | 833.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 838.55 | 841.16 | 837.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 838.55 | 841.16 | 837.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 838.55 | 841.16 | 837.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 831.70 | 841.16 | 837.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 834.45 | 839.82 | 837.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:45:00 | 833.90 | 839.82 | 837.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 840.35 | 839.92 | 837.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 834.55 | 839.92 | 837.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 836.80 | 839.30 | 837.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:45:00 | 833.60 | 839.30 | 837.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 843.90 | 840.22 | 838.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:30:00 | 836.50 | 840.22 | 838.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 832.20 | 840.20 | 838.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 832.20 | 840.20 | 838.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 814.95 | 835.15 | 836.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 13:15:00 | 792.40 | 820.26 | 825.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 10:15:00 | 824.15 | 816.13 | 821.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 10:15:00 | 824.15 | 816.13 | 821.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 824.15 | 816.13 | 821.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 824.15 | 816.13 | 821.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 816.80 | 816.26 | 820.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 10:15:00 | 814.05 | 818.27 | 820.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 11:15:00 | 812.80 | 817.91 | 819.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 835.90 | 817.85 | 818.16 | SL hit (close>static) qty=1.00 sl=828.50 alert=retest2 |

### Cycle 84 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 832.85 | 820.85 | 819.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 866.45 | 835.19 | 827.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 872.40 | 875.25 | 863.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:30:00 | 883.40 | 878.17 | 866.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 875.05 | 884.79 | 876.85 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 875.05 | 884.79 | 876.85 | SL hit (close<ema400) qty=1.00 sl=876.85 alert=retest1 |

### Cycle 85 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 859.60 | 874.17 | 874.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 844.50 | 858.92 | 865.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 834.20 | 823.48 | 838.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 834.20 | 823.48 | 838.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 834.20 | 823.48 | 838.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:30:00 | 837.00 | 823.48 | 838.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 841.25 | 827.76 | 833.57 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 843.90 | 836.52 | 836.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 845.55 | 838.33 | 837.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 839.55 | 841.50 | 839.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 12:15:00 | 839.55 | 841.50 | 839.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 839.55 | 841.50 | 839.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:00:00 | 839.55 | 841.50 | 839.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 834.80 | 840.16 | 839.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:00:00 | 834.80 | 840.16 | 839.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 832.05 | 838.54 | 838.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:30:00 | 831.55 | 838.54 | 838.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 832.00 | 837.23 | 837.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 12:15:00 | 829.70 | 833.78 | 835.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 13:15:00 | 835.30 | 834.09 | 835.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 13:15:00 | 835.30 | 834.09 | 835.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 835.30 | 834.09 | 835.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:45:00 | 834.20 | 834.09 | 835.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 833.95 | 832.21 | 834.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:45:00 | 828.50 | 833.25 | 834.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 09:15:00 | 852.05 | 826.99 | 826.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 852.05 | 826.99 | 826.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 861.10 | 841.05 | 833.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 13:15:00 | 857.00 | 857.87 | 849.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 14:00:00 | 857.00 | 857.87 | 849.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 856.50 | 862.88 | 858.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 856.50 | 862.88 | 858.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 858.30 | 861.97 | 858.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 15:00:00 | 860.55 | 860.48 | 858.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:00:00 | 861.70 | 860.61 | 859.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 15:00:00 | 860.70 | 861.67 | 860.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 864.70 | 861.11 | 860.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 862.95 | 861.48 | 860.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 854.40 | 859.19 | 859.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 854.40 | 859.19 | 859.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 852.90 | 857.93 | 858.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 852.95 | 849.49 | 852.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 852.95 | 849.49 | 852.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 852.95 | 849.49 | 852.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 852.95 | 849.49 | 852.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 859.80 | 851.55 | 852.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 857.60 | 851.55 | 852.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 854.05 | 852.05 | 852.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 12:15:00 | 851.90 | 852.05 | 852.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 14:15:00 | 845.25 | 839.62 | 839.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 845.25 | 839.62 | 839.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 15:15:00 | 846.05 | 840.91 | 839.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 846.65 | 850.12 | 847.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 846.65 | 850.12 | 847.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 846.65 | 850.12 | 847.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 846.65 | 850.12 | 847.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 848.00 | 849.70 | 847.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 840.10 | 849.70 | 847.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 847.10 | 849.18 | 847.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 15:00:00 | 851.10 | 847.62 | 847.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 12:15:00 | 845.45 | 846.70 | 846.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 845.45 | 846.70 | 846.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 841.75 | 845.40 | 846.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 823.55 | 821.78 | 830.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 09:30:00 | 825.25 | 821.78 | 830.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 830.85 | 823.59 | 830.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 830.85 | 823.59 | 830.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 823.10 | 823.49 | 829.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 12:15:00 | 819.35 | 823.49 | 829.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 13:30:00 | 822.40 | 823.53 | 828.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:30:00 | 821.90 | 824.72 | 827.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 12:15:00 | 833.20 | 826.42 | 827.95 | SL hit (close>static) qty=1.00 sl=830.90 alert=retest2 |

### Cycle 92 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 833.65 | 829.50 | 829.03 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 825.45 | 828.69 | 828.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 823.30 | 827.61 | 828.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 829.50 | 827.57 | 828.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 829.50 | 827.57 | 828.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 829.50 | 827.57 | 828.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 832.50 | 827.57 | 828.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 829.90 | 828.04 | 828.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:30:00 | 829.30 | 828.04 | 828.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 829.60 | 828.35 | 828.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 830.00 | 828.35 | 828.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 831.70 | 829.02 | 828.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 832.65 | 829.75 | 829.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 15:15:00 | 861.00 | 861.18 | 854.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:15:00 | 858.45 | 861.18 | 854.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 858.60 | 860.67 | 854.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 868.95 | 858.25 | 857.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 852.45 | 856.68 | 856.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 852.45 | 856.68 | 856.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 842.55 | 853.86 | 855.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 848.70 | 848.38 | 852.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 848.70 | 848.38 | 852.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 852.40 | 849.18 | 852.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 856.50 | 849.18 | 852.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 858.00 | 850.94 | 852.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 860.25 | 850.94 | 852.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 868.00 | 856.12 | 854.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 870.75 | 859.05 | 856.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 15:15:00 | 913.00 | 914.26 | 900.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 09:15:00 | 908.40 | 914.26 | 900.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 906.15 | 914.58 | 908.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 906.15 | 914.58 | 908.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 912.30 | 914.13 | 908.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:15:00 | 915.05 | 914.13 | 908.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:15:00 | 917.45 | 913.08 | 909.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 915.00 | 914.04 | 912.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 11:15:00 | 902.50 | 910.43 | 911.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 902.50 | 910.43 | 911.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 12:15:00 | 901.25 | 908.60 | 910.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 11:15:00 | 903.25 | 901.94 | 905.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 11:15:00 | 903.25 | 901.94 | 905.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 903.25 | 901.94 | 905.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:00:00 | 903.25 | 901.94 | 905.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 908.85 | 903.32 | 905.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:00:00 | 908.85 | 903.32 | 905.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 907.75 | 904.21 | 905.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:30:00 | 909.15 | 904.21 | 905.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 15:15:00 | 912.55 | 907.32 | 907.11 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 897.00 | 905.26 | 906.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 876.00 | 897.18 | 902.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 833.95 | 831.68 | 845.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 833.95 | 831.68 | 845.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 840.05 | 834.59 | 842.87 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 852.55 | 846.91 | 846.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 15:15:00 | 856.95 | 849.42 | 847.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 856.35 | 857.97 | 853.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 856.35 | 857.97 | 853.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 857.95 | 858.81 | 854.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 857.25 | 858.81 | 854.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 850.25 | 857.10 | 854.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 850.25 | 857.10 | 854.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 846.80 | 855.04 | 853.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 846.95 | 855.04 | 853.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 842.85 | 851.38 | 852.32 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 855.10 | 852.24 | 852.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 13:15:00 | 859.75 | 853.74 | 852.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 862.25 | 873.96 | 868.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 862.25 | 873.96 | 868.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 862.25 | 873.96 | 868.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 862.25 | 873.96 | 868.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 857.20 | 870.61 | 867.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 856.70 | 870.61 | 867.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 856.65 | 865.23 | 865.60 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 12:15:00 | 871.95 | 866.15 | 865.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 13:15:00 | 875.50 | 868.02 | 866.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 09:15:00 | 868.80 | 869.75 | 867.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 868.80 | 869.75 | 867.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 868.80 | 869.75 | 867.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:45:00 | 869.80 | 869.75 | 867.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 867.45 | 869.29 | 867.75 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 862.00 | 865.97 | 866.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 858.00 | 863.52 | 865.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 813.15 | 812.81 | 825.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 12:00:00 | 805.75 | 811.07 | 822.87 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 820.00 | 789.72 | 798.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 820.00 | 789.72 | 798.64 | SL hit (close>ema400) qty=1.00 sl=798.64 alert=retest1 |

### Cycle 106 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 827.60 | 804.64 | 804.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 840.30 | 829.94 | 821.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 826.90 | 831.55 | 825.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 826.90 | 831.55 | 825.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 827.00 | 829.97 | 825.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 822.45 | 829.97 | 825.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 824.65 | 828.90 | 825.48 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 816.80 | 823.10 | 823.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 801.25 | 818.69 | 821.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 793.75 | 790.69 | 800.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:45:00 | 793.35 | 790.69 | 800.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 799.10 | 792.37 | 800.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 799.10 | 792.37 | 800.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 798.25 | 793.55 | 799.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 816.10 | 793.55 | 799.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 813.10 | 797.46 | 801.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 811.20 | 797.46 | 801.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 819.70 | 804.52 | 803.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 826.00 | 811.18 | 807.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 816.95 | 817.11 | 811.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 816.95 | 817.11 | 811.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 811.70 | 816.03 | 811.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 810.65 | 816.03 | 811.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 807.25 | 814.28 | 810.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 807.25 | 814.28 | 810.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 810.15 | 813.45 | 810.86 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 803.60 | 808.41 | 809.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 791.95 | 803.18 | 806.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 11:15:00 | 783.25 | 782.69 | 788.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 12:00:00 | 783.25 | 782.69 | 788.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 767.75 | 756.98 | 766.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 767.75 | 756.98 | 766.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 760.95 | 757.77 | 765.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 768.55 | 757.77 | 765.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 763.05 | 760.45 | 765.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 762.95 | 760.45 | 765.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 768.35 | 762.62 | 764.98 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 772.40 | 766.17 | 765.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 775.00 | 768.67 | 767.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 761.10 | 767.79 | 766.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 761.10 | 767.79 | 766.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 761.10 | 767.79 | 766.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 761.10 | 767.79 | 766.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 760.00 | 766.23 | 766.32 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 11:15:00 | 772.25 | 767.09 | 766.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 12:15:00 | 777.15 | 769.10 | 767.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 826.50 | 827.28 | 818.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 826.00 | 827.28 | 818.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 819.40 | 825.19 | 818.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 820.10 | 825.19 | 818.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 821.25 | 824.40 | 818.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:00:00 | 826.20 | 824.76 | 819.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 816.65 | 822.00 | 820.15 | SL hit (close<static) qty=1.00 sl=818.05 alert=retest2 |

### Cycle 113 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 809.65 | 818.47 | 818.81 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 825.00 | 818.91 | 818.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 830.50 | 823.01 | 820.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 843.80 | 844.76 | 839.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:45:00 | 843.50 | 844.76 | 839.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 846.25 | 849.16 | 846.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 846.25 | 849.16 | 846.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 848.70 | 849.07 | 846.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:45:00 | 856.30 | 851.00 | 847.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 14:30:00 | 855.15 | 852.73 | 849.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 861.75 | 852.92 | 849.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 856.10 | 866.19 | 867.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 856.10 | 866.19 | 867.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 854.45 | 863.84 | 866.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 864.70 | 864.01 | 865.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 864.70 | 864.01 | 865.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 864.70 | 864.01 | 865.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 864.70 | 864.01 | 865.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 865.95 | 864.40 | 865.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 865.95 | 864.40 | 865.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 867.65 | 865.05 | 866.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:30:00 | 868.00 | 865.05 | 866.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 871.05 | 866.25 | 866.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 871.05 | 866.25 | 866.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 869.45 | 866.89 | 866.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 883.10 | 870.13 | 868.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 887.55 | 887.81 | 880.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 10:30:00 | 889.25 | 887.81 | 880.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 877.50 | 884.44 | 880.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 877.50 | 884.44 | 880.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 872.65 | 882.08 | 880.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 872.40 | 882.08 | 880.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 873.20 | 878.71 | 878.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 858.30 | 870.98 | 874.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 870.00 | 869.94 | 873.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:45:00 | 869.10 | 869.94 | 873.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 858.95 | 864.77 | 869.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:45:00 | 862.15 | 864.77 | 869.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 849.80 | 845.94 | 850.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 850.40 | 845.94 | 850.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 841.50 | 839.46 | 842.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 817.80 | 825.85 | 829.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:45:00 | 816.05 | 822.04 | 825.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 833.00 | 826.01 | 825.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 833.00 | 826.01 | 825.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 15:15:00 | 834.60 | 827.73 | 826.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 826.60 | 827.51 | 826.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 826.60 | 827.51 | 826.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 826.60 | 827.51 | 826.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 826.60 | 827.51 | 826.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 826.80 | 827.36 | 826.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:45:00 | 825.90 | 827.36 | 826.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 831.90 | 828.27 | 827.17 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 807.85 | 823.54 | 825.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 805.50 | 815.27 | 820.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 815.25 | 813.74 | 818.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:45:00 | 808.05 | 813.74 | 818.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 803.20 | 804.00 | 808.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 790.00 | 803.20 | 808.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 750.50 | 766.50 | 780.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 14:15:00 | 711.00 | 731.59 | 753.69 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 120 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 747.40 | 738.15 | 737.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 761.20 | 750.34 | 745.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 750.50 | 756.34 | 752.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 750.50 | 756.34 | 752.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 750.50 | 756.34 | 752.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 750.50 | 756.34 | 752.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 740.55 | 753.18 | 751.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 740.55 | 753.18 | 751.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 749.20 | 751.64 | 750.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 744.25 | 751.64 | 750.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 747.00 | 750.72 | 750.32 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 737.50 | 748.07 | 749.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 727.60 | 742.69 | 746.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 09:15:00 | 725.75 | 708.03 | 714.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 725.75 | 708.03 | 714.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 725.75 | 708.03 | 714.26 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 723.95 | 715.32 | 714.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 736.50 | 719.56 | 716.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 746.00 | 750.47 | 740.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 746.00 | 750.47 | 740.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 746.50 | 749.48 | 742.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:30:00 | 752.30 | 749.44 | 743.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:30:00 | 752.50 | 747.64 | 744.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:15:00 | 752.55 | 749.41 | 746.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 754.35 | 748.19 | 745.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 756.40 | 749.83 | 746.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 770.95 | 758.54 | 754.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:15:00 | 767.80 | 763.30 | 757.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 13:45:00 | 767.65 | 764.68 | 759.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 777.50 | 764.03 | 760.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 763.40 | 766.47 | 763.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 763.40 | 766.47 | 763.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 762.00 | 765.58 | 763.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 761.60 | 765.58 | 763.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 765.50 | 765.56 | 763.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 764.25 | 765.56 | 763.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 761.85 | 764.82 | 763.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 755.30 | 764.82 | 763.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 768.70 | 765.60 | 763.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 754.70 | 761.98 | 762.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 754.70 | 761.98 | 762.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 741.20 | 756.59 | 759.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 11:15:00 | 690.40 | 689.45 | 702.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 12:00:00 | 690.40 | 689.45 | 702.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 671.20 | 669.25 | 677.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 676.75 | 669.25 | 677.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 673.80 | 671.13 | 677.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 666.80 | 671.13 | 677.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 692.00 | 673.24 | 674.21 | SL hit (close>static) qty=1.00 sl=678.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 693.20 | 677.23 | 675.94 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 674.45 | 685.07 | 686.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 666.55 | 673.77 | 678.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 642.55 | 634.20 | 641.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 642.55 | 634.20 | 641.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 642.55 | 634.20 | 641.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 642.55 | 634.20 | 641.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 647.45 | 636.85 | 641.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 643.60 | 636.85 | 641.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 647.50 | 638.98 | 642.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:30:00 | 649.95 | 638.98 | 642.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 643.35 | 641.72 | 643.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:15:00 | 642.65 | 641.72 | 643.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:30:00 | 642.20 | 641.59 | 642.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:00:00 | 642.50 | 641.59 | 642.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 656.45 | 644.47 | 643.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 656.45 | 644.47 | 643.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 660.95 | 650.02 | 646.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 663.15 | 665.58 | 660.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 663.15 | 665.58 | 660.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 668.05 | 666.42 | 662.94 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 647.60 | 659.09 | 660.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 643.00 | 655.88 | 658.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 661.75 | 656.33 | 658.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 661.75 | 656.33 | 658.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 661.75 | 656.33 | 658.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 661.75 | 656.33 | 658.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 668.20 | 658.71 | 659.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 668.20 | 658.71 | 659.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 12:15:00 | 665.70 | 660.11 | 659.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 673.45 | 662.77 | 661.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 10:15:00 | 666.65 | 667.94 | 664.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 666.65 | 667.94 | 664.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 666.65 | 667.94 | 664.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 666.65 | 667.94 | 664.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 663.65 | 667.08 | 664.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 663.65 | 667.08 | 664.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 666.95 | 667.06 | 664.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 670.50 | 667.06 | 664.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 12:00:00 | 667.45 | 668.85 | 666.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 14:15:00 | 658.55 | 665.62 | 665.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 658.55 | 665.62 | 665.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 655.30 | 660.11 | 662.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 680.45 | 662.65 | 662.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 680.45 | 662.65 | 662.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 680.45 | 662.65 | 662.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 680.45 | 662.65 | 662.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 675.30 | 665.18 | 663.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 684.60 | 673.95 | 669.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 697.20 | 698.38 | 692.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 13:45:00 | 696.95 | 698.38 | 692.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 698.40 | 697.94 | 693.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 696.80 | 697.94 | 693.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 699.45 | 704.65 | 700.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 699.45 | 704.65 | 700.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 699.35 | 703.59 | 700.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 698.85 | 703.59 | 700.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 696.65 | 702.20 | 699.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 693.75 | 702.20 | 699.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 695.90 | 699.59 | 699.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 696.15 | 699.59 | 699.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 693.00 | 698.27 | 698.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 686.75 | 694.18 | 696.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 690.45 | 684.83 | 688.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 690.45 | 684.83 | 688.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 690.45 | 684.83 | 688.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 690.45 | 684.83 | 688.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 690.65 | 685.99 | 688.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 689.90 | 685.99 | 688.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 689.90 | 687.19 | 688.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 692.00 | 687.19 | 688.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 689.30 | 687.61 | 688.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 686.00 | 687.61 | 688.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 14:45:00 | 686.50 | 684.74 | 687.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 681.25 | 676.99 | 676.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 681.25 | 676.99 | 676.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 685.40 | 678.67 | 677.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 14:15:00 | 680.10 | 681.48 | 679.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 14:15:00 | 680.10 | 681.48 | 679.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 680.10 | 681.48 | 679.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:45:00 | 678.45 | 681.48 | 679.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 680.05 | 681.19 | 679.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 669.10 | 681.19 | 679.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 660.40 | 677.04 | 678.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 658.10 | 670.99 | 674.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 620.00 | 619.06 | 635.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 620.00 | 619.06 | 635.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 622.30 | 617.06 | 622.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:30:00 | 628.50 | 617.06 | 622.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 628.00 | 619.25 | 623.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 628.00 | 619.25 | 623.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 630.85 | 621.57 | 623.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:30:00 | 629.90 | 621.57 | 623.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 623.70 | 623.25 | 624.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 15:00:00 | 623.70 | 623.25 | 624.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 623.20 | 623.24 | 624.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 09:15:00 | 653.80 | 623.24 | 624.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 655.20 | 629.63 | 626.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 659.20 | 635.55 | 629.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 653.80 | 653.89 | 644.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:30:00 | 654.00 | 653.89 | 644.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 683.05 | 685.91 | 681.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 682.65 | 685.91 | 681.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 681.25 | 684.48 | 681.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:00:00 | 681.25 | 684.48 | 681.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 680.80 | 683.74 | 681.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 680.80 | 683.74 | 681.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 679.00 | 682.80 | 681.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:00:00 | 679.00 | 682.80 | 681.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 681.25 | 682.49 | 681.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 678.85 | 682.49 | 681.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 680.90 | 682.17 | 681.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 681.60 | 682.17 | 681.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 664.85 | 678.71 | 679.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 654.00 | 673.76 | 677.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 663.40 | 661.55 | 668.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 663.40 | 661.55 | 668.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 665.25 | 663.07 | 665.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 662.25 | 663.07 | 665.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 679.40 | 663.98 | 664.40 | SL hit (close>static) qty=1.00 sl=670.90 alert=retest2 |

### Cycle 136 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 680.50 | 667.28 | 665.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 09:15:00 | 687.20 | 675.16 | 670.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 686.30 | 690.12 | 685.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 686.30 | 690.12 | 685.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 686.30 | 690.12 | 685.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 686.30 | 690.12 | 685.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 684.10 | 688.91 | 685.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 684.10 | 688.91 | 685.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 681.45 | 687.42 | 684.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 681.55 | 687.42 | 684.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 673.00 | 681.43 | 682.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 668.40 | 678.82 | 681.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 13:15:00 | 679.05 | 677.31 | 679.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 14:00:00 | 679.05 | 677.31 | 679.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 679.05 | 677.66 | 679.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:30:00 | 679.85 | 677.66 | 679.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 679.25 | 677.97 | 679.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 680.40 | 677.97 | 679.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 676.90 | 677.76 | 679.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 673.50 | 676.67 | 678.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 669.00 | 674.27 | 677.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 639.82 | 658.72 | 668.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 635.55 | 658.72 | 668.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 667.75 | 644.33 | 653.55 | SL hit (close>ema200) qty=0.50 sl=644.33 alert=retest2 |

### Cycle 138 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 679.05 | 661.15 | 659.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 680.40 | 665.00 | 661.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 09:15:00 | 753.05 | 753.17 | 740.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:45:00 | 750.45 | 753.17 | 740.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 776.20 | 777.18 | 773.91 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 13:15:00 | 770.25 | 774.12 | 774.42 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 784.40 | 775.86 | 775.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 10:15:00 | 787.50 | 778.19 | 776.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 10:15:00 | 804.80 | 805.77 | 799.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 10:30:00 | 805.50 | 805.77 | 799.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 803.90 | 805.82 | 802.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 803.70 | 805.82 | 802.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 800.75 | 804.81 | 802.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 797.35 | 804.81 | 802.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 800.20 | 803.88 | 801.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:45:00 | 803.00 | 803.72 | 801.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 14:15:00 | 883.30 | 859.01 | 837.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 859.90 | 865.68 | 865.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 849.60 | 862.46 | 864.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 851.80 | 850.78 | 855.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 851.80 | 850.78 | 855.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 856.00 | 851.83 | 855.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 851.60 | 851.83 | 855.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 846.45 | 850.75 | 854.94 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 867.00 | 858.14 | 857.03 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 852.35 | 857.18 | 857.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 848.05 | 854.12 | 855.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 843.00 | 842.09 | 846.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 843.00 | 842.09 | 846.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 843.00 | 842.09 | 846.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:15:00 | 848.45 | 842.09 | 846.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 854.10 | 844.49 | 847.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 854.10 | 844.49 | 847.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 855.20 | 846.63 | 847.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 850.45 | 846.63 | 847.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 853.95 | 848.70 | 848.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 853.95 | 848.70 | 848.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 856.40 | 852.28 | 850.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 852.90 | 853.20 | 851.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 852.90 | 853.20 | 851.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 852.90 | 853.20 | 851.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 851.60 | 853.20 | 851.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 848.60 | 852.28 | 851.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 866.40 | 852.28 | 851.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 14:15:00 | 848.45 | 854.53 | 853.43 | SL hit (close<static) qty=1.00 sl=848.60 alert=retest2 |

### Cycle 145 — SELL (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 10:15:00 | 850.80 | 852.51 | 852.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 838.80 | 849.49 | 851.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 846.90 | 845.35 | 848.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 846.90 | 845.35 | 848.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 846.90 | 845.35 | 848.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 846.90 | 845.35 | 848.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 847.30 | 845.74 | 847.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 852.25 | 845.74 | 847.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 850.85 | 846.76 | 848.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:45:00 | 845.30 | 847.44 | 848.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 845.30 | 841.10 | 843.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 841.15 | 840.37 | 841.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 835.20 | 834.11 | 834.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 835.20 | 834.11 | 834.11 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 828.90 | 833.13 | 833.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 824.10 | 831.33 | 832.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 832.10 | 829.50 | 831.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 832.10 | 829.50 | 831.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 832.10 | 829.50 | 831.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 832.10 | 829.50 | 831.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 838.95 | 831.39 | 832.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 839.25 | 831.39 | 832.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 841.95 | 833.50 | 832.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 844.00 | 835.60 | 833.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 11:15:00 | 835.80 | 836.77 | 835.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 11:15:00 | 835.80 | 836.77 | 835.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 835.80 | 836.77 | 835.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 835.80 | 836.77 | 835.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 836.50 | 836.71 | 835.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:45:00 | 836.05 | 836.71 | 835.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 833.15 | 836.00 | 834.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 833.15 | 836.00 | 834.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 829.90 | 834.78 | 834.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:45:00 | 828.05 | 834.78 | 834.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 15:15:00 | 829.30 | 833.68 | 834.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 827.75 | 831.58 | 832.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 14:15:00 | 831.35 | 831.22 | 832.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 14:15:00 | 831.35 | 831.22 | 832.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 831.35 | 831.22 | 832.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 831.35 | 831.22 | 832.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 826.00 | 830.28 | 831.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 823.55 | 828.24 | 830.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 10:45:00 | 824.65 | 821.25 | 825.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:00:00 | 825.05 | 822.21 | 824.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 825.25 | 825.09 | 825.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 836.45 | 827.36 | 826.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 836.45 | 827.36 | 826.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 837.70 | 829.43 | 827.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 830.55 | 830.92 | 828.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 14:00:00 | 830.55 | 830.92 | 828.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 841.85 | 844.55 | 841.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 840.65 | 844.55 | 841.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 842.85 | 844.91 | 842.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 846.30 | 844.91 | 842.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 847.20 | 845.37 | 843.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 849.30 | 846.12 | 843.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:00:00 | 848.60 | 847.62 | 845.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:30:00 | 849.10 | 848.62 | 845.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:45:00 | 848.05 | 847.95 | 846.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 846.65 | 847.69 | 846.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 846.65 | 847.69 | 846.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 848.15 | 847.78 | 846.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:30:00 | 846.55 | 847.78 | 846.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 845.65 | 847.35 | 846.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 845.65 | 847.35 | 846.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 845.30 | 846.94 | 846.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 834.40 | 846.94 | 846.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 831.10 | 843.77 | 844.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 831.10 | 843.77 | 844.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 827.55 | 835.87 | 839.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 800.00 | 798.94 | 811.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 802.00 | 798.94 | 811.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 791.90 | 787.09 | 791.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 791.90 | 787.09 | 791.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 791.00 | 787.87 | 791.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:30:00 | 794.05 | 787.87 | 791.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 786.30 | 787.56 | 791.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 777.45 | 787.02 | 790.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:15:00 | 780.70 | 783.21 | 787.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 781.05 | 782.73 | 786.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 795.85 | 787.98 | 787.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 795.85 | 787.98 | 787.92 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 774.60 | 785.31 | 786.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 765.60 | 778.55 | 782.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 759.90 | 759.63 | 766.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 759.90 | 759.63 | 766.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 762.90 | 760.21 | 765.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:30:00 | 765.45 | 760.21 | 765.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 763.75 | 754.93 | 757.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 763.75 | 754.93 | 757.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 764.45 | 756.84 | 758.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 765.20 | 756.84 | 758.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 753.05 | 757.15 | 758.36 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 763.15 | 758.69 | 758.51 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 756.75 | 758.30 | 758.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 754.00 | 757.34 | 757.90 | Break + close below crossover candle low |

### Cycle 156 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 773.15 | 757.19 | 756.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 779.35 | 773.86 | 768.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 773.70 | 773.83 | 769.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 773.70 | 773.83 | 769.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 773.35 | 773.73 | 769.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:45:00 | 770.50 | 773.73 | 769.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 770.85 | 773.28 | 770.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 770.85 | 773.28 | 770.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 770.45 | 772.71 | 770.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 777.95 | 772.71 | 770.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 767.75 | 773.61 | 772.69 | SL hit (close<static) qty=1.00 sl=769.20 alert=retest2 |

### Cycle 157 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 767.65 | 771.66 | 771.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 762.95 | 768.73 | 770.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 773.60 | 768.54 | 769.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 773.60 | 768.54 | 769.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 773.60 | 768.54 | 769.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 773.60 | 768.54 | 769.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 774.05 | 769.64 | 770.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 774.05 | 769.64 | 770.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 772.00 | 770.11 | 770.33 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 774.05 | 770.90 | 770.67 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 759.80 | 768.93 | 769.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 750.70 | 757.59 | 762.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 751.35 | 750.35 | 755.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 751.35 | 750.35 | 755.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 748.15 | 744.82 | 747.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 748.15 | 744.82 | 747.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 749.30 | 745.71 | 747.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 747.25 | 745.71 | 747.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 757.60 | 748.09 | 748.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 757.60 | 748.09 | 748.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 761.75 | 750.82 | 750.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 764.00 | 753.46 | 751.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 761.05 | 762.24 | 759.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:00:00 | 761.05 | 762.24 | 759.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 760.35 | 761.87 | 759.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 765.95 | 760.32 | 759.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 757.05 | 759.93 | 759.16 | SL hit (close<static) qty=1.00 sl=757.90 alert=retest2 |

### Cycle 161 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 751.20 | 758.18 | 758.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 749.00 | 756.35 | 757.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 756.85 | 756.25 | 757.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 755.70 | 756.25 | 757.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 760.05 | 757.01 | 757.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 759.75 | 757.01 | 757.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 760.10 | 757.63 | 757.71 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 761.05 | 758.31 | 758.01 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 754.00 | 757.78 | 758.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 11:15:00 | 747.90 | 755.06 | 756.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 754.45 | 752.92 | 754.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 754.45 | 752.92 | 754.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 754.45 | 752.92 | 754.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 749.50 | 754.85 | 755.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 752.10 | 754.56 | 754.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 757.00 | 755.22 | 755.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 757.00 | 755.22 | 755.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 761.65 | 757.68 | 756.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 781.75 | 785.39 | 781.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 781.75 | 785.39 | 781.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 783.15 | 784.95 | 781.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 789.10 | 784.36 | 781.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 777.30 | 783.00 | 781.48 | SL hit (close<static) qty=1.00 sl=781.25 alert=retest2 |

### Cycle 165 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 776.70 | 780.42 | 780.49 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 781.15 | 780.57 | 780.55 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 778.15 | 780.08 | 780.33 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 784.30 | 781.14 | 780.75 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 774.35 | 780.66 | 780.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 763.10 | 776.08 | 778.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 720.40 | 720.30 | 728.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 720.40 | 720.30 | 728.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 715.05 | 714.56 | 718.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 722.50 | 714.56 | 718.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 718.80 | 716.01 | 718.84 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 727.25 | 721.05 | 720.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 733.65 | 727.56 | 725.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 730.85 | 733.47 | 730.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 730.85 | 733.47 | 730.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 730.85 | 733.47 | 730.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 731.25 | 733.47 | 730.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 725.90 | 731.96 | 730.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 725.90 | 731.96 | 730.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 729.15 | 731.39 | 730.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 730.85 | 731.03 | 730.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 723.30 | 729.32 | 729.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 723.30 | 729.32 | 729.56 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 738.75 | 730.06 | 729.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 742.20 | 736.83 | 733.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 740.00 | 740.95 | 738.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 11:00:00 | 740.00 | 740.95 | 738.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 737.90 | 740.34 | 738.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 12:45:00 | 741.70 | 740.35 | 738.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:15:00 | 741.40 | 740.35 | 738.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:45:00 | 741.20 | 740.42 | 738.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:15:00 | 741.20 | 740.42 | 738.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 773.70 | 776.18 | 772.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 773.70 | 776.18 | 772.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 774.00 | 775.74 | 772.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 770.25 | 775.74 | 772.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 774.70 | 775.53 | 772.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 782.50 | 773.62 | 772.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 771.15 | 775.13 | 775.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 771.15 | 775.13 | 775.27 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 781.50 | 775.85 | 775.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 782.25 | 778.05 | 776.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 777.75 | 779.80 | 777.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 14:15:00 | 777.75 | 779.80 | 777.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 777.75 | 779.80 | 777.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 777.75 | 779.80 | 777.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 782.60 | 780.36 | 778.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 777.05 | 779.04 | 777.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 775.00 | 778.23 | 777.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 772.60 | 778.23 | 777.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 774.55 | 776.98 | 777.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 764.35 | 774.26 | 775.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 770.10 | 765.16 | 769.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 770.10 | 765.16 | 769.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 770.10 | 765.16 | 769.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 770.10 | 765.16 | 769.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 777.80 | 767.69 | 770.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 777.80 | 767.69 | 770.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 777.45 | 769.64 | 770.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:15:00 | 783.00 | 769.64 | 770.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 781.25 | 771.96 | 771.91 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 761.80 | 772.45 | 773.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 758.15 | 768.74 | 771.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 758.55 | 757.29 | 761.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:30:00 | 759.20 | 757.29 | 761.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 757.65 | 757.37 | 761.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 759.35 | 757.37 | 761.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 759.50 | 757.79 | 761.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 765.30 | 757.79 | 761.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 759.55 | 758.14 | 761.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:15:00 | 761.20 | 758.14 | 761.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 758.60 | 758.24 | 760.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 760.85 | 758.24 | 760.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 761.25 | 758.92 | 760.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 759.65 | 758.92 | 760.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 760.20 | 759.18 | 760.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:30:00 | 762.00 | 759.18 | 760.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 760.00 | 759.34 | 760.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 762.35 | 759.34 | 760.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 754.30 | 758.33 | 760.10 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 765.10 | 760.78 | 760.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 767.05 | 764.19 | 762.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 761.05 | 763.60 | 762.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 761.05 | 763.60 | 762.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 761.05 | 763.60 | 762.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:15:00 | 762.00 | 763.60 | 762.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 762.00 | 763.28 | 762.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 768.00 | 763.28 | 762.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 762.55 | 764.02 | 763.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 762.30 | 763.67 | 763.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 762.30 | 763.67 | 763.69 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 770.60 | 764.71 | 764.05 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 756.60 | 763.37 | 764.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 751.95 | 758.63 | 761.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 722.85 | 721.92 | 726.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 722.85 | 721.92 | 726.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 733.10 | 724.10 | 726.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 733.10 | 724.10 | 726.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 730.65 | 725.41 | 726.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 730.65 | 725.41 | 726.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 731.50 | 728.34 | 728.05 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 727.50 | 728.23 | 728.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 724.05 | 726.28 | 727.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 726.45 | 726.16 | 726.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 11:15:00 | 726.45 | 726.16 | 726.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 726.45 | 726.16 | 726.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:45:00 | 726.45 | 726.16 | 726.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 725.60 | 725.09 | 726.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 721.60 | 724.47 | 725.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 722.90 | 714.42 | 713.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 722.90 | 714.42 | 713.52 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 694.80 | 711.07 | 713.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 690.75 | 707.00 | 711.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 697.25 | 694.14 | 700.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 697.25 | 694.14 | 700.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 695.90 | 692.97 | 697.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 696.90 | 692.97 | 697.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 692.00 | 689.07 | 691.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 693.35 | 689.07 | 691.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 692.50 | 689.75 | 691.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 692.50 | 689.75 | 691.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 694.30 | 690.66 | 692.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 694.30 | 690.66 | 692.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 693.00 | 691.13 | 692.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 701.55 | 691.13 | 692.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 698.30 | 692.56 | 692.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 704.25 | 692.56 | 692.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 693.35 | 692.79 | 692.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 694.00 | 692.79 | 692.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 693.75 | 692.98 | 692.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 696.05 | 693.60 | 693.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 693.35 | 695.11 | 694.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 693.35 | 695.11 | 694.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 693.35 | 695.11 | 694.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 692.30 | 695.11 | 694.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 693.20 | 694.73 | 694.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 692.80 | 694.73 | 694.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 696.05 | 694.99 | 694.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 694.60 | 694.99 | 694.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 696.50 | 695.29 | 694.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 694.70 | 695.29 | 694.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 698.00 | 696.05 | 694.95 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 690.40 | 693.80 | 694.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 690.35 | 693.11 | 693.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 684.25 | 682.48 | 686.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 684.25 | 682.48 | 686.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 685.05 | 682.99 | 686.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 685.10 | 682.99 | 686.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 683.85 | 681.45 | 684.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 685.60 | 681.45 | 684.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 681.95 | 681.55 | 683.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 684.35 | 681.55 | 683.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 682.50 | 681.85 | 683.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 683.55 | 681.85 | 683.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 687.30 | 682.94 | 684.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 687.30 | 682.94 | 684.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 690.90 | 684.53 | 684.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 690.90 | 684.53 | 684.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 693.75 | 686.37 | 685.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 697.50 | 692.00 | 689.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 693.70 | 693.83 | 691.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 703.00 | 693.83 | 691.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 694.30 | 695.27 | 693.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 696.15 | 695.27 | 693.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 693.70 | 694.96 | 693.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 696.60 | 694.71 | 694.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 691.30 | 693.47 | 693.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 691.30 | 693.47 | 693.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 688.65 | 692.50 | 693.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 686.90 | 683.41 | 686.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 686.90 | 683.41 | 686.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 686.90 | 683.41 | 686.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 686.90 | 683.41 | 686.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 690.50 | 684.82 | 686.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 690.50 | 684.82 | 686.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 688.30 | 685.52 | 686.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:15:00 | 687.95 | 686.13 | 686.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 687.55 | 686.41 | 686.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 687.95 | 686.19 | 686.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 690.10 | 687.49 | 687.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 690.10 | 687.49 | 687.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 690.95 | 688.18 | 687.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 707.90 | 707.93 | 702.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 707.90 | 707.93 | 702.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 706.40 | 707.51 | 704.17 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 695.75 | 701.56 | 702.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 692.55 | 696.21 | 698.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 652.55 | 652.20 | 661.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 652.55 | 652.20 | 661.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 656.80 | 651.43 | 655.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 651.75 | 654.22 | 655.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 619.16 | 630.67 | 639.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 618.10 | 617.56 | 626.19 | SL hit (close>ema200) qty=0.50 sl=617.56 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 614.30 | 607.36 | 607.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 619.25 | 610.78 | 608.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 633.90 | 633.92 | 626.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:45:00 | 632.70 | 633.92 | 626.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 625.95 | 633.41 | 630.51 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 613.20 | 627.03 | 627.97 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 654.15 | 630.10 | 627.58 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 660.80 | 666.19 | 666.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 656.60 | 664.27 | 665.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 639.30 | 635.83 | 645.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 639.30 | 635.83 | 645.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 643.40 | 638.96 | 643.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 643.40 | 638.96 | 643.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 643.00 | 639.77 | 643.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 639.90 | 639.77 | 643.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 638.35 | 639.19 | 639.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:15:00 | 607.90 | 615.05 | 619.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 14:15:00 | 606.43 | 610.81 | 614.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 15:15:00 | 611.90 | 611.03 | 613.92 | SL hit (close>ema200) qty=0.50 sl=611.03 alert=retest2 |

### Cycle 196 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 579.85 | 577.32 | 577.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 585.30 | 580.73 | 578.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 581.05 | 583.34 | 581.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 581.05 | 583.34 | 581.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 581.05 | 583.34 | 581.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 581.05 | 583.34 | 581.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 574.35 | 581.54 | 580.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 574.35 | 581.54 | 580.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 572.10 | 579.65 | 579.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 562.55 | 575.28 | 577.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 536.05 | 534.30 | 543.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 536.05 | 534.30 | 543.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 548.60 | 538.21 | 542.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 548.60 | 538.21 | 542.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 548.10 | 540.19 | 543.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 549.70 | 540.19 | 543.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 555.85 | 546.54 | 545.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 564.35 | 551.99 | 548.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 542.55 | 554.34 | 551.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 542.55 | 554.34 | 551.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 542.55 | 554.34 | 551.18 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 540.25 | 548.55 | 549.05 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 552.75 | 549.13 | 548.88 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 544.70 | 548.02 | 548.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 540.80 | 546.58 | 547.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 526.90 | 522.24 | 528.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 526.90 | 522.24 | 528.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 526.90 | 522.24 | 528.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 527.25 | 522.24 | 528.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 524.70 | 522.73 | 528.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 526.20 | 522.73 | 528.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 539.85 | 525.77 | 528.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 539.85 | 525.77 | 528.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 539.60 | 528.54 | 529.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 539.60 | 528.54 | 529.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 532.50 | 530.36 | 530.19 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 525.80 | 529.67 | 530.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 523.75 | 528.49 | 529.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 523.00 | 512.70 | 518.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 523.00 | 512.70 | 518.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 523.00 | 512.70 | 518.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 523.00 | 512.70 | 518.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 516.00 | 513.36 | 517.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:45:00 | 514.10 | 514.61 | 517.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 515.05 | 511.78 | 513.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 524.70 | 515.84 | 514.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 524.70 | 515.84 | 514.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 525.30 | 519.06 | 516.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 563.70 | 564.16 | 554.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:30:00 | 562.50 | 564.16 | 554.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 560.65 | 567.04 | 562.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 563.10 | 567.04 | 562.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 563.40 | 565.45 | 562.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 565.65 | 565.45 | 562.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 596.50 | 603.43 | 603.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 596.50 | 603.43 | 603.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 594.90 | 601.72 | 602.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 594.35 | 590.18 | 594.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 594.35 | 590.18 | 594.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 594.35 | 590.18 | 594.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:30:00 | 589.50 | 592.24 | 593.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:00:00 | 589.70 | 592.24 | 593.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 588.30 | 591.63 | 593.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 587.70 | 589.33 | 591.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 600.70 | 591.39 | 592.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 600.70 | 591.39 | 592.06 | SL hit (close>static) qty=1.00 sl=599.80 alert=retest2 |

### Cycle 206 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 598.70 | 592.85 | 592.66 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 582.55 | 593.07 | 593.44 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 606.30 | 593.51 | 592.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 607.70 | 601.93 | 597.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 601.35 | 601.81 | 597.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 601.35 | 601.81 | 597.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 597.25 | 600.90 | 597.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 597.70 | 600.90 | 597.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 598.75 | 600.47 | 597.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:00:00 | 600.50 | 600.48 | 598.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 600.40 | 599.03 | 598.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:30:00 | 600.20 | 598.84 | 598.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:00:00 | 600.15 | 598.84 | 598.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 609.30 | 612.77 | 609.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 608.80 | 612.77 | 609.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 610.20 | 612.26 | 609.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 608.60 | 612.26 | 609.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 608.25 | 611.46 | 609.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 608.25 | 611.46 | 609.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 606.30 | 610.43 | 609.11 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-23 11:30:00 | 474.20 | 2023-05-23 14:15:00 | 467.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-05-26 13:15:00 | 475.35 | 2023-05-30 14:15:00 | 474.45 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2023-05-29 09:30:00 | 476.75 | 2023-05-30 14:15:00 | 474.45 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-05-30 12:00:00 | 475.40 | 2023-05-30 14:15:00 | 474.45 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2023-05-30 12:45:00 | 475.25 | 2023-05-30 14:15:00 | 474.45 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2023-06-01 11:30:00 | 473.65 | 2023-06-01 13:15:00 | 475.85 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2023-06-01 12:00:00 | 473.30 | 2023-06-01 13:15:00 | 475.85 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-07-05 15:15:00 | 503.00 | 2023-07-10 15:15:00 | 500.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-07-13 13:30:00 | 494.45 | 2023-07-17 13:15:00 | 496.90 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-07-14 09:30:00 | 493.70 | 2023-07-17 13:15:00 | 496.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-07-14 11:30:00 | 493.95 | 2023-07-17 13:15:00 | 496.90 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2023-07-17 09:30:00 | 494.50 | 2023-07-17 13:15:00 | 496.90 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-07-21 11:15:00 | 503.50 | 2023-07-24 11:15:00 | 499.30 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-07-21 13:00:00 | 502.30 | 2023-07-24 11:15:00 | 499.30 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-07-21 13:45:00 | 502.30 | 2023-07-24 11:15:00 | 499.30 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-07-24 09:15:00 | 504.10 | 2023-07-24 11:15:00 | 499.30 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-07-25 12:15:00 | 489.95 | 2023-07-27 11:15:00 | 505.50 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2023-08-08 11:15:00 | 483.00 | 2023-08-10 14:15:00 | 491.65 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2023-08-09 10:15:00 | 484.90 | 2023-08-10 14:15:00 | 491.65 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-08-09 12:00:00 | 484.25 | 2023-08-10 14:15:00 | 491.65 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2023-08-09 15:00:00 | 485.30 | 2023-08-10 14:15:00 | 491.65 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-08-18 13:00:00 | 473.30 | 2023-08-21 09:15:00 | 478.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-08-23 11:15:00 | 481.50 | 2023-08-25 10:15:00 | 474.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-08-24 09:15:00 | 486.80 | 2023-08-25 10:15:00 | 474.60 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2023-08-25 09:45:00 | 481.85 | 2023-08-25 10:15:00 | 474.60 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-09-07 11:15:00 | 526.50 | 2023-09-12 12:15:00 | 525.70 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-09-25 14:00:00 | 517.75 | 2023-09-25 14:15:00 | 522.10 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-09-29 09:15:00 | 531.55 | 2023-10-04 12:15:00 | 517.40 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2023-10-13 10:30:00 | 575.05 | 2023-10-18 09:15:00 | 565.85 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2023-10-19 09:15:00 | 562.70 | 2023-10-23 14:15:00 | 534.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 09:15:00 | 562.70 | 2023-10-26 15:15:00 | 522.80 | STOP_HIT | 0.50 | 7.09% |
| BUY | retest2 | 2023-11-08 09:15:00 | 597.15 | 2023-11-24 14:15:00 | 629.95 | STOP_HIT | 1.00 | 5.49% |
| BUY | retest2 | 2023-11-08 09:45:00 | 598.30 | 2023-11-24 14:15:00 | 629.95 | STOP_HIT | 1.00 | 5.29% |
| BUY | retest2 | 2023-11-08 11:45:00 | 598.05 | 2023-11-24 14:15:00 | 629.95 | STOP_HIT | 1.00 | 5.33% |
| BUY | retest2 | 2023-11-08 12:30:00 | 598.00 | 2023-11-24 14:15:00 | 629.95 | STOP_HIT | 1.00 | 5.34% |
| BUY | retest2 | 2023-11-20 15:15:00 | 628.15 | 2023-11-24 14:15:00 | 629.95 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2023-11-22 13:00:00 | 629.10 | 2023-11-24 14:15:00 | 629.95 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2023-12-01 15:15:00 | 626.95 | 2023-12-04 09:15:00 | 637.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2023-12-08 09:15:00 | 661.70 | 2023-12-08 12:15:00 | 646.25 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2023-12-08 11:15:00 | 654.95 | 2023-12-08 12:15:00 | 646.25 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-12-08 12:00:00 | 655.30 | 2023-12-08 12:15:00 | 646.25 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-12-11 09:15:00 | 664.95 | 2023-12-20 13:15:00 | 680.55 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2023-12-13 09:15:00 | 670.60 | 2023-12-20 13:15:00 | 680.55 | STOP_HIT | 1.00 | 1.48% |
| SELL | retest2 | 2024-01-25 10:45:00 | 751.25 | 2024-01-29 10:15:00 | 763.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2024-02-02 09:15:00 | 802.95 | 2024-02-02 14:15:00 | 795.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest1 | 2024-02-02 13:45:00 | 801.50 | 2024-02-02 14:15:00 | 795.85 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-02-05 09:30:00 | 797.05 | 2024-02-05 15:15:00 | 787.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-02-09 12:30:00 | 829.00 | 2024-02-12 11:15:00 | 812.10 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-02-13 12:15:00 | 811.40 | 2024-02-13 13:15:00 | 825.15 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-02-20 09:15:00 | 859.55 | 2024-02-29 09:15:00 | 887.45 | STOP_HIT | 1.00 | 3.25% |
| SELL | retest2 | 2024-03-19 11:30:00 | 823.35 | 2024-03-21 09:15:00 | 845.80 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-03-19 15:00:00 | 823.15 | 2024-03-21 09:15:00 | 845.80 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-03-20 09:45:00 | 816.75 | 2024-03-21 09:15:00 | 845.80 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2024-03-20 14:15:00 | 823.80 | 2024-03-21 09:15:00 | 845.80 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest1 | 2024-03-28 09:15:00 | 886.70 | 2024-04-01 09:15:00 | 931.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-28 10:00:00 | 887.15 | 2024-04-01 09:15:00 | 931.51 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-28 11:00:00 | 886.85 | 2024-04-01 09:15:00 | 931.19 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-28 09:15:00 | 886.70 | 2024-04-02 09:15:00 | 932.60 | STOP_HIT | 0.50 | 5.18% |
| BUY | retest1 | 2024-03-28 10:00:00 | 887.15 | 2024-04-02 09:15:00 | 932.60 | STOP_HIT | 0.50 | 5.12% |
| BUY | retest1 | 2024-03-28 11:00:00 | 886.85 | 2024-04-02 09:15:00 | 932.60 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2024-04-10 09:15:00 | 912.80 | 2024-04-10 14:15:00 | 909.45 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-04-10 14:15:00 | 912.50 | 2024-04-10 14:15:00 | 909.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-05-06 12:30:00 | 886.80 | 2024-05-07 12:15:00 | 842.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 12:30:00 | 886.80 | 2024-05-08 12:15:00 | 859.00 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2024-05-16 13:45:00 | 832.40 | 2024-05-16 14:15:00 | 840.30 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-05-28 09:30:00 | 837.20 | 2024-06-03 09:15:00 | 839.90 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-06-28 12:15:00 | 819.10 | 2024-07-01 11:15:00 | 827.80 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-07-25 10:15:00 | 814.05 | 2024-07-26 09:15:00 | 835.90 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-07-25 11:15:00 | 812.80 | 2024-07-26 09:15:00 | 835.90 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2024-07-31 11:30:00 | 883.40 | 2024-08-01 11:15:00 | 875.05 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-08-13 09:45:00 | 828.50 | 2024-08-16 09:15:00 | 852.05 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-08-21 15:00:00 | 860.55 | 2024-08-23 11:15:00 | 854.40 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-08-22 10:00:00 | 861.70 | 2024-08-23 11:15:00 | 854.40 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-08-22 15:00:00 | 860.70 | 2024-08-23 11:15:00 | 854.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-08-23 09:15:00 | 864.70 | 2024-08-23 11:15:00 | 854.40 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-08-27 12:15:00 | 851.90 | 2024-08-30 14:15:00 | 845.25 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2024-09-04 15:00:00 | 851.10 | 2024-09-05 12:15:00 | 845.45 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-09-09 12:15:00 | 819.35 | 2024-09-10 12:15:00 | 833.20 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-09-09 13:30:00 | 822.40 | 2024-09-10 12:15:00 | 833.20 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-09-10 11:30:00 | 821.90 | 2024-09-10 12:15:00 | 833.20 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-09-19 09:15:00 | 868.95 | 2024-09-19 10:15:00 | 852.45 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-09-26 11:15:00 | 915.05 | 2024-09-30 11:15:00 | 902.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-09-26 14:15:00 | 917.45 | 2024-09-30 11:15:00 | 902.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-09-27 15:15:00 | 915.00 | 2024-09-30 11:15:00 | 902.50 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest1 | 2024-10-24 12:00:00 | 805.75 | 2024-10-28 09:15:00 | 820.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-11-27 13:00:00 | 826.20 | 2024-11-28 10:15:00 | 816.65 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-12-06 12:45:00 | 856.30 | 2024-12-13 09:15:00 | 856.10 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-12-06 14:30:00 | 855.15 | 2024-12-13 09:15:00 | 856.10 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-12-09 09:15:00 | 861.75 | 2024-12-13 09:15:00 | 856.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-01-01 09:15:00 | 817.80 | 2025-01-02 14:15:00 | 833.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-01-02 09:45:00 | 816.05 | 2025-01-02 14:15:00 | 833.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-01-09 09:15:00 | 790.00 | 2025-01-10 14:15:00 | 750.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 790.00 | 2025-01-13 14:15:00 | 711.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 10:30:00 | 752.30 | 2025-02-07 13:15:00 | 754.70 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-02-01 09:30:00 | 752.50 | 2025-02-07 13:15:00 | 754.70 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-02-01 12:15:00 | 752.55 | 2025-02-07 13:15:00 | 754.70 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-02-01 13:15:00 | 754.35 | 2025-02-07 13:15:00 | 754.70 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-02-05 09:15:00 | 770.95 | 2025-02-07 13:15:00 | 754.70 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-02-05 12:15:00 | 767.80 | 2025-02-07 13:15:00 | 754.70 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-02-05 13:45:00 | 767.65 | 2025-02-07 13:15:00 | 754.70 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-02-06 09:15:00 | 777.50 | 2025-02-07 13:15:00 | 754.70 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-02-18 09:15:00 | 666.80 | 2025-02-19 09:15:00 | 692.00 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-03-04 11:15:00 | 642.65 | 2025-03-05 09:15:00 | 656.45 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-04 14:30:00 | 642.20 | 2025-03-05 09:15:00 | 656.45 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-03-04 15:00:00 | 642.50 | 2025-03-05 09:15:00 | 656.45 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-03-12 13:15:00 | 670.50 | 2025-03-13 14:15:00 | 658.55 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-03-13 12:00:00 | 667.45 | 2025-03-13 14:15:00 | 658.55 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-03-28 12:15:00 | 686.00 | 2025-04-02 15:15:00 | 681.25 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-03-28 14:45:00 | 686.50 | 2025-04-02 15:15:00 | 681.25 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-04-29 10:15:00 | 662.25 | 2025-04-30 09:15:00 | 679.40 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-05-08 11:30:00 | 673.50 | 2025-05-09 09:15:00 | 639.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:30:00 | 669.00 | 2025-05-09 09:15:00 | 635.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 11:30:00 | 673.50 | 2025-05-12 09:15:00 | 667.75 | STOP_HIT | 0.50 | 0.85% |
| SELL | retest2 | 2025-05-08 13:30:00 | 669.00 | 2025-05-12 09:15:00 | 667.75 | STOP_HIT | 0.50 | 0.19% |
| BUY | retest2 | 2025-06-04 11:45:00 | 803.00 | 2025-06-06 14:15:00 | 883.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-20 12:15:00 | 850.45 | 2025-06-20 14:15:00 | 853.95 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-06-24 09:15:00 | 866.40 | 2025-06-24 14:15:00 | 848.45 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-06-27 11:45:00 | 845.30 | 2025-07-07 13:15:00 | 835.20 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-07-01 10:00:00 | 845.30 | 2025-07-07 13:15:00 | 835.20 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2025-07-02 09:30:00 | 841.15 | 2025-07-07 13:15:00 | 835.20 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-07-11 10:30:00 | 823.55 | 2025-07-15 10:15:00 | 836.45 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-07-14 10:45:00 | 824.65 | 2025-07-15 10:15:00 | 836.45 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-14 13:00:00 | 825.05 | 2025-07-15 10:15:00 | 836.45 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-07-15 09:30:00 | 825.25 | 2025-07-15 10:15:00 | 836.45 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-21 10:30:00 | 849.30 | 2025-07-23 09:15:00 | 831.10 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-07-21 14:00:00 | 848.60 | 2025-07-23 09:15:00 | 831.10 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-21 14:30:00 | 849.10 | 2025-07-23 09:15:00 | 831.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-22 11:45:00 | 848.05 | 2025-07-23 09:15:00 | 831.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-08-01 14:15:00 | 777.45 | 2025-08-04 15:15:00 | 795.85 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-04 10:15:00 | 780.70 | 2025-08-04 15:15:00 | 795.85 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-08-04 10:45:00 | 781.05 | 2025-08-04 15:15:00 | 795.85 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-08-21 09:15:00 | 777.95 | 2025-08-22 09:15:00 | 767.75 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-05 09:15:00 | 765.95 | 2025-09-05 10:15:00 | 757.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-11 11:30:00 | 749.50 | 2025-09-11 15:15:00 | 757.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-11 13:30:00 | 752.10 | 2025-09-11 15:15:00 | 757.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-09-19 09:15:00 | 789.10 | 2025-09-19 10:15:00 | 777.30 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-08 13:15:00 | 730.85 | 2025-10-08 14:15:00 | 723.30 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-14 12:45:00 | 741.70 | 2025-10-28 11:15:00 | 771.15 | STOP_HIT | 1.00 | 3.97% |
| BUY | retest2 | 2025-10-14 13:15:00 | 741.40 | 2025-10-28 11:15:00 | 771.15 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest2 | 2025-10-14 13:45:00 | 741.20 | 2025-10-28 11:15:00 | 771.15 | STOP_HIT | 1.00 | 4.04% |
| BUY | retest2 | 2025-10-14 14:15:00 | 741.20 | 2025-10-28 11:15:00 | 771.15 | STOP_HIT | 1.00 | 4.04% |
| BUY | retest2 | 2025-10-27 09:15:00 | 782.50 | 2025-10-28 11:15:00 | 771.15 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-13 09:15:00 | 768.00 | 2025-11-14 11:15:00 | 762.30 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-14 10:45:00 | 762.55 | 2025-11-14 11:15:00 | 762.30 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-12-01 10:45:00 | 721.60 | 2025-12-05 10:15:00 | 722.90 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-12-29 09:30:00 | 696.60 | 2025-12-29 11:15:00 | 691.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-31 14:15:00 | 687.95 | 2026-01-01 12:15:00 | 690.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-12-31 15:00:00 | 687.55 | 2026-01-01 12:15:00 | 690.10 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-01-01 09:30:00 | 687.95 | 2026-01-01 12:15:00 | 690.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-01-16 13:15:00 | 651.75 | 2026-01-20 13:15:00 | 619.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:15:00 | 651.75 | 2026-01-21 14:15:00 | 618.10 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-17 09:15:00 | 639.90 | 2026-02-25 12:15:00 | 607.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 638.35 | 2026-02-26 14:15:00 | 606.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 09:15:00 | 639.90 | 2026-02-26 15:15:00 | 611.90 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2026-02-19 09:15:00 | 638.35 | 2026-02-26 15:15:00 | 611.90 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2026-04-01 12:45:00 | 514.10 | 2026-04-06 09:15:00 | 524.70 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-04-02 14:30:00 | 515.05 | 2026-04-06 09:15:00 | 524.70 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-04-13 10:15:00 | 563.10 | 2026-04-23 12:15:00 | 596.50 | STOP_HIT | 1.00 | 5.93% |
| BUY | retest2 | 2026-04-13 11:30:00 | 563.40 | 2026-04-23 12:15:00 | 596.50 | STOP_HIT | 1.00 | 5.88% |
| BUY | retest2 | 2026-04-13 12:15:00 | 565.65 | 2026-04-23 12:15:00 | 596.50 | STOP_HIT | 1.00 | 5.45% |
| SELL | retest2 | 2026-04-28 09:30:00 | 589.50 | 2026-04-29 09:15:00 | 600.70 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-04-28 10:00:00 | 589.70 | 2026-04-29 09:15:00 | 600.70 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-04-28 10:30:00 | 588.30 | 2026-04-29 09:15:00 | 600.70 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-04-28 15:00:00 | 587.70 | 2026-04-29 09:15:00 | 600.70 | STOP_HIT | 1.00 | -2.21% |
