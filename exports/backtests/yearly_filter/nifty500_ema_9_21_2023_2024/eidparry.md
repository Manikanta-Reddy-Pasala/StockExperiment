# E.I.D. Parry (India) Ltd. (EIDPARRY)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 834.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 226 |
| ALERT1 | 142 |
| ALERT2 | 141 |
| ALERT2_SKIP | 68 |
| ALERT3 | 380 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 175 |
| PARTIAL | 17 |
| TARGET_HIT | 12 |
| STOP_HIT | 167 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 196 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 131
- **Target hits / Stop hits / Partials:** 12 / 167 / 17
- **Avg / median % per leg:** 0.44% / -0.81%
- **Sum % (uncompounded):** 86.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 92 | 29 | 31.5% | 9 | 83 | 0 | 0.24% | 22.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.13% | -6.4% |
| BUY @ 3rd Alert (retest2) | 89 | 29 | 32.6% | 9 | 80 | 0 | 0.32% | 28.9% |
| SELL (all) | 104 | 36 | 34.6% | 3 | 84 | 17 | 0.61% | 63.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.22% | -1.2% |
| SELL @ 3rd Alert (retest2) | 103 | 36 | 35.0% | 3 | 83 | 17 | 0.63% | 64.8% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.90% | -7.6% |
| retest2 (combined) | 192 | 65 | 33.9% | 12 | 163 | 17 | 0.49% | 93.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 14:15:00 | 497.65 | 496.30 | 496.21 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 10:15:00 | 495.65 | 496.17 | 496.18 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 497.10 | 496.11 | 496.10 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 10:15:00 | 495.40 | 495.97 | 496.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 11:15:00 | 493.40 | 495.46 | 495.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 13:15:00 | 495.05 | 494.91 | 495.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-26 14:00:00 | 495.05 | 494.91 | 495.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 493.80 | 494.18 | 494.95 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 15:15:00 | 495.95 | 495.09 | 495.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 09:15:00 | 497.60 | 495.59 | 495.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 494.50 | 495.37 | 495.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 10:15:00 | 494.50 | 495.37 | 495.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 494.50 | 495.37 | 495.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 494.50 | 495.37 | 495.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 494.80 | 495.26 | 495.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 12:00:00 | 494.80 | 495.26 | 495.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 496.00 | 495.41 | 495.26 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 486.00 | 493.44 | 494.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 09:15:00 | 476.50 | 488.74 | 492.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 482.75 | 477.22 | 483.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 482.75 | 477.22 | 483.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 482.75 | 477.22 | 483.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:30:00 | 483.80 | 477.22 | 483.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 484.00 | 478.58 | 483.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 11:00:00 | 484.00 | 478.58 | 483.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 11:15:00 | 483.45 | 479.55 | 483.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 11:45:00 | 485.45 | 479.55 | 483.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 12:15:00 | 485.95 | 480.83 | 483.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 12:30:00 | 486.00 | 480.83 | 483.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 485.25 | 481.71 | 483.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 13:30:00 | 487.10 | 481.71 | 483.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 10:15:00 | 486.75 | 484.65 | 484.56 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 10:15:00 | 483.00 | 484.54 | 484.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 13:15:00 | 480.15 | 483.12 | 483.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 474.95 | 474.16 | 477.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-07 10:15:00 | 476.30 | 474.16 | 477.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 10:15:00 | 477.50 | 474.83 | 477.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 10:30:00 | 477.85 | 474.83 | 477.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 11:15:00 | 478.25 | 475.51 | 477.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 11:30:00 | 478.25 | 475.51 | 477.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 12:15:00 | 478.10 | 476.03 | 477.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-07 13:30:00 | 476.40 | 476.03 | 477.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 10:00:00 | 475.90 | 475.75 | 476.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 11:15:00 | 476.30 | 476.10 | 476.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 12:00:00 | 476.50 | 475.46 | 475.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 476.40 | 475.66 | 475.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:45:00 | 476.45 | 475.66 | 475.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-09 14:15:00 | 481.05 | 476.74 | 476.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 14:15:00 | 481.05 | 476.74 | 476.32 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 13:15:00 | 471.80 | 476.08 | 476.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-13 12:15:00 | 469.10 | 472.26 | 474.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 15:15:00 | 471.00 | 470.92 | 472.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-14 09:15:00 | 471.55 | 470.92 | 472.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 470.20 | 470.78 | 472.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 15:15:00 | 469.00 | 470.21 | 471.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-19 11:00:00 | 468.70 | 467.12 | 467.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 11:15:00 | 472.50 | 468.19 | 467.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 11:15:00 | 472.50 | 468.19 | 467.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 12:15:00 | 473.75 | 471.19 | 469.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 15:15:00 | 476.10 | 476.20 | 474.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:15:00 | 478.95 | 476.20 | 474.00 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:15:00 | 482.00 | 476.65 | 474.40 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 15:15:00 | 483.00 | 484.25 | 481.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:15:00 | 479.40 | 484.25 | 481.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 473.70 | 482.14 | 481.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-26 09:15:00 | 473.70 | 482.14 | 481.02 | SL hit (close<ema400) qty=1.00 sl=481.02 alert=retest1 |

### Cycle 12 — SELL (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 11:15:00 | 472.20 | 478.69 | 479.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 09:15:00 | 471.80 | 474.84 | 477.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 10:15:00 | 466.20 | 465.96 | 469.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 10:15:00 | 466.20 | 465.96 | 469.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 466.20 | 465.96 | 469.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 10:45:00 | 468.40 | 465.96 | 469.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 463.00 | 463.96 | 466.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 11:15:00 | 460.60 | 463.94 | 466.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 14:15:00 | 459.95 | 462.74 | 465.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-05 10:15:00 | 466.00 | 464.72 | 464.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 466.00 | 464.72 | 464.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 11:15:00 | 467.75 | 465.33 | 464.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 10:15:00 | 465.00 | 466.28 | 465.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 10:15:00 | 465.00 | 466.28 | 465.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 10:15:00 | 465.00 | 466.28 | 465.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 10:45:00 | 464.60 | 466.28 | 465.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 11:15:00 | 465.00 | 466.03 | 465.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 11:45:00 | 465.55 | 466.03 | 465.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 12:15:00 | 466.25 | 466.07 | 465.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 12:45:00 | 465.10 | 466.07 | 465.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 13:15:00 | 465.45 | 465.95 | 465.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 14:00:00 | 465.45 | 465.95 | 465.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 466.00 | 465.96 | 465.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 15:15:00 | 465.85 | 465.96 | 465.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 15:15:00 | 465.85 | 465.94 | 465.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 09:15:00 | 473.10 | 465.94 | 465.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 12:45:00 | 466.95 | 467.46 | 466.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 14:15:00 | 467.00 | 467.35 | 466.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 15:15:00 | 467.95 | 467.19 | 466.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 467.95 | 467.34 | 466.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 09:30:00 | 470.45 | 468.19 | 467.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 10:15:00 | 465.50 | 466.88 | 467.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 10:15:00 | 465.50 | 466.88 | 467.06 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 15:15:00 | 467.80 | 467.09 | 467.04 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 09:15:00 | 465.95 | 466.86 | 466.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 11:15:00 | 461.90 | 465.24 | 466.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 462.00 | 456.38 | 458.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 462.00 | 456.38 | 458.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 462.00 | 456.38 | 458.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:45:00 | 463.60 | 456.38 | 458.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 462.95 | 457.70 | 458.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:45:00 | 462.95 | 457.70 | 458.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 465.00 | 460.13 | 459.55 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 15:15:00 | 457.85 | 460.54 | 460.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 09:15:00 | 457.00 | 459.83 | 460.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 461.50 | 457.07 | 458.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 461.50 | 457.07 | 458.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 461.50 | 457.07 | 458.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:30:00 | 462.70 | 457.07 | 458.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 464.05 | 458.47 | 458.77 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 11:15:00 | 462.50 | 459.27 | 459.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 14:15:00 | 467.55 | 461.74 | 460.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 15:15:00 | 497.00 | 497.87 | 493.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-28 09:15:00 | 493.45 | 497.87 | 493.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 496.60 | 497.62 | 493.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:30:00 | 495.80 | 497.62 | 493.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 12:15:00 | 498.90 | 497.19 | 494.33 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 10:15:00 | 492.00 | 493.51 | 493.64 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 09:15:00 | 497.60 | 493.93 | 493.68 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 492.55 | 493.42 | 493.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 484.00 | 491.16 | 492.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 490.75 | 489.52 | 491.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 11:00:00 | 490.75 | 489.52 | 491.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 498.05 | 491.22 | 491.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 12:00:00 | 498.05 | 491.22 | 491.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 12:15:00 | 494.25 | 491.83 | 491.95 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 13:15:00 | 494.50 | 492.36 | 492.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 14:15:00 | 498.00 | 493.49 | 492.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 13:15:00 | 498.50 | 498.54 | 496.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-04 14:00:00 | 498.50 | 498.54 | 496.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 493.55 | 497.37 | 496.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:00:00 | 493.55 | 497.37 | 496.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 490.45 | 495.99 | 495.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:00:00 | 490.45 | 495.99 | 495.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2023-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 11:15:00 | 491.95 | 495.18 | 495.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 14:15:00 | 489.00 | 492.45 | 493.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 10:15:00 | 485.30 | 483.59 | 486.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 10:15:00 | 485.30 | 483.59 | 486.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 485.30 | 483.59 | 486.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 11:00:00 | 485.30 | 483.59 | 486.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 462.25 | 459.95 | 462.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:30:00 | 462.65 | 459.95 | 462.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 464.30 | 460.82 | 463.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 10:45:00 | 464.50 | 460.82 | 463.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 466.25 | 461.90 | 463.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:00:00 | 466.25 | 461.90 | 463.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 465.30 | 464.36 | 464.23 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 12:15:00 | 462.05 | 463.84 | 464.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 09:15:00 | 458.05 | 462.00 | 463.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 12:15:00 | 458.00 | 457.71 | 459.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 13:00:00 | 458.00 | 457.71 | 459.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 460.70 | 458.37 | 459.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 460.70 | 458.37 | 459.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 459.50 | 458.59 | 459.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 463.35 | 458.59 | 459.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 462.40 | 459.36 | 459.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:30:00 | 462.70 | 459.36 | 459.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 461.80 | 460.28 | 460.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 12:15:00 | 462.55 | 460.73 | 460.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 458.70 | 460.58 | 460.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 14:15:00 | 458.70 | 460.58 | 460.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 458.70 | 460.58 | 460.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 15:00:00 | 458.70 | 460.58 | 460.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 461.80 | 460.83 | 460.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 469.40 | 462.85 | 461.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 14:15:00 | 474.00 | 475.62 | 475.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 474.00 | 475.62 | 475.62 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 09:15:00 | 478.15 | 476.00 | 475.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 10:15:00 | 483.00 | 478.97 | 478.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 14:15:00 | 483.20 | 483.77 | 481.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 15:00:00 | 483.20 | 483.77 | 481.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 512.50 | 526.49 | 522.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 511.65 | 526.49 | 522.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 514.00 | 523.99 | 521.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 513.40 | 523.99 | 521.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 508.00 | 518.39 | 519.26 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 15:15:00 | 520.90 | 516.81 | 516.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 531.35 | 519.72 | 517.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 12:15:00 | 557.95 | 558.52 | 550.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 12:30:00 | 557.30 | 558.52 | 550.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 556.80 | 557.57 | 551.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 14:30:00 | 552.60 | 557.57 | 551.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 548.55 | 556.08 | 551.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:15:00 | 544.80 | 556.08 | 551.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 544.75 | 553.81 | 551.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:30:00 | 544.35 | 553.81 | 551.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 12:15:00 | 540.50 | 548.99 | 549.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 11:15:00 | 536.55 | 542.01 | 545.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 541.55 | 535.65 | 537.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 541.55 | 535.65 | 537.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 541.55 | 535.65 | 537.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:45:00 | 547.05 | 535.65 | 537.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 543.80 | 537.28 | 538.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:30:00 | 548.50 | 537.28 | 538.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 539.15 | 538.62 | 538.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:45:00 | 538.60 | 538.62 | 538.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 536.00 | 538.10 | 538.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 14:30:00 | 538.70 | 538.10 | 538.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 523.70 | 529.79 | 533.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 14:45:00 | 522.90 | 526.28 | 530.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 09:15:00 | 536.90 | 528.20 | 530.22 | SL hit (close>static) qty=1.00 sl=533.70 alert=retest2 |

### Cycle 33 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 523.30 | 519.46 | 519.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 10:15:00 | 530.15 | 521.59 | 520.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 523.00 | 523.81 | 522.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 523.00 | 523.81 | 522.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 523.00 | 523.81 | 522.21 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 15:15:00 | 520.00 | 521.28 | 521.43 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 524.65 | 521.96 | 521.72 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 15:15:00 | 520.00 | 521.86 | 521.92 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 524.00 | 522.29 | 522.11 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 14:15:00 | 519.15 | 521.92 | 522.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 15:15:00 | 517.40 | 521.02 | 521.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 521.30 | 521.08 | 521.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 521.30 | 521.08 | 521.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 521.30 | 521.08 | 521.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-12 09:45:00 | 521.25 | 521.08 | 521.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 519.50 | 520.76 | 521.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 11:45:00 | 517.75 | 519.76 | 520.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 09:15:00 | 491.86 | 503.14 | 507.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-20 09:15:00 | 501.55 | 498.80 | 502.59 | SL hit (close>ema200) qty=0.50 sl=498.80 alert=retest2 |

### Cycle 39 — BUY (started 2023-11-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 12:15:00 | 469.35 | 466.53 | 466.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 15:15:00 | 469.70 | 467.84 | 467.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 12:15:00 | 478.95 | 479.11 | 477.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 13:00:00 | 478.95 | 479.11 | 477.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 477.70 | 478.95 | 477.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 15:00:00 | 477.70 | 478.95 | 477.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 477.95 | 478.75 | 477.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 479.80 | 478.75 | 477.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 11:30:00 | 480.50 | 479.37 | 478.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 13:00:00 | 480.25 | 479.55 | 478.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 14:15:00 | 497.50 | 501.98 | 502.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 14:15:00 | 497.50 | 501.98 | 502.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 09:15:00 | 495.85 | 500.18 | 501.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 12:15:00 | 500.30 | 498.83 | 500.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 12:15:00 | 500.30 | 498.83 | 500.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 12:15:00 | 500.30 | 498.83 | 500.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 13:00:00 | 500.30 | 498.83 | 500.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 13:15:00 | 497.55 | 498.57 | 499.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 15:00:00 | 495.60 | 497.98 | 499.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 14:00:00 | 494.45 | 495.68 | 497.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 10:00:00 | 495.80 | 495.25 | 496.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 12:30:00 | 495.75 | 495.38 | 496.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 492.20 | 494.38 | 495.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 11:15:00 | 486.85 | 492.84 | 494.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 14:00:00 | 484.95 | 488.64 | 491.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 15:00:00 | 487.15 | 488.34 | 491.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 09:15:00 | 511.50 | 494.19 | 493.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 09:15:00 | 511.50 | 494.19 | 493.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 10:15:00 | 516.50 | 498.65 | 495.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 14:15:00 | 535.05 | 535.30 | 527.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 15:00:00 | 535.05 | 535.30 | 527.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 530.75 | 534.51 | 530.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 12:45:00 | 530.80 | 534.51 | 530.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 528.25 | 533.26 | 530.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 14:00:00 | 528.25 | 533.26 | 530.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 529.70 | 532.54 | 530.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 15:00:00 | 529.70 | 532.54 | 530.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 530.50 | 532.14 | 530.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 09:15:00 | 544.80 | 532.14 | 530.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-01 13:30:00 | 531.60 | 535.32 | 534.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:45:00 | 532.55 | 556.90 | 552.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 12:15:00 | 539.95 | 549.14 | 549.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 12:15:00 | 539.95 | 549.14 | 549.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 14:15:00 | 531.00 | 543.49 | 546.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 539.45 | 535.59 | 539.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 539.45 | 535.59 | 539.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 539.45 | 535.59 | 539.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:30:00 | 538.25 | 535.59 | 539.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 540.60 | 536.59 | 539.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 11:00:00 | 540.60 | 536.59 | 539.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 11:15:00 | 540.35 | 537.34 | 539.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 11:45:00 | 541.50 | 537.34 | 539.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 12:15:00 | 538.75 | 537.62 | 539.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 12:45:00 | 540.15 | 537.62 | 539.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 13:15:00 | 538.80 | 537.86 | 539.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 14:00:00 | 538.80 | 537.86 | 539.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 14:15:00 | 540.55 | 538.40 | 539.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 15:00:00 | 540.55 | 538.40 | 539.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 537.20 | 538.16 | 539.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:15:00 | 539.20 | 538.16 | 539.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 534.40 | 537.41 | 538.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 09:30:00 | 532.55 | 537.14 | 538.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 11:30:00 | 531.40 | 535.85 | 537.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 13:30:00 | 533.00 | 534.61 | 536.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 11:15:00 | 543.80 | 536.59 | 536.66 | SL hit (close>static) qty=1.00 sl=543.10 alert=retest2 |

### Cycle 43 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 539.45 | 537.16 | 536.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 551.90 | 541.93 | 539.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 558.10 | 559.51 | 552.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 15:00:00 | 558.10 | 559.51 | 552.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 557.50 | 561.10 | 558.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 557.50 | 561.10 | 558.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 547.70 | 558.42 | 557.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:30:00 | 548.20 | 558.42 | 557.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 540.05 | 554.74 | 556.12 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 558.90 | 555.54 | 555.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 12:15:00 | 564.90 | 561.81 | 559.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 09:15:00 | 562.00 | 562.96 | 560.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 10:00:00 | 562.00 | 562.96 | 560.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 565.55 | 564.93 | 563.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 13:15:00 | 569.00 | 565.77 | 563.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 13:45:00 | 568.50 | 566.22 | 564.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 14:45:00 | 570.10 | 567.25 | 564.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 569.95 | 567.25 | 565.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 563.45 | 566.49 | 564.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:00:00 | 563.45 | 566.49 | 564.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 567.80 | 566.75 | 565.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:15:00 | 568.85 | 566.75 | 565.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 13:15:00 | 558.40 | 565.35 | 565.00 | SL hit (close<static) qty=1.00 sl=562.60 alert=retest2 |

### Cycle 46 — SELL (started 2023-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 14:15:00 | 556.70 | 563.62 | 564.25 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 14:15:00 | 570.45 | 562.40 | 561.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 572.20 | 565.58 | 563.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 13:15:00 | 574.00 | 576.25 | 572.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-05 14:00:00 | 574.00 | 576.25 | 572.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 575.00 | 576.00 | 572.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 15:15:00 | 573.00 | 576.00 | 572.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 15:15:00 | 573.00 | 575.40 | 572.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:45:00 | 577.00 | 573.70 | 573.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 10:15:00 | 578.65 | 573.70 | 573.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 09:30:00 | 576.20 | 578.07 | 576.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 10:45:00 | 579.05 | 578.24 | 576.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 581.65 | 579.97 | 578.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:30:00 | 578.90 | 579.97 | 578.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 585.00 | 583.44 | 581.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:30:00 | 583.10 | 583.44 | 581.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 12:15:00 | 582.60 | 583.71 | 581.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 13:00:00 | 582.60 | 583.71 | 581.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 13:15:00 | 584.60 | 583.89 | 582.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 14:30:00 | 586.50 | 584.15 | 582.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 11:15:00 | 579.70 | 582.64 | 582.20 | SL hit (close<static) qty=1.00 sl=582.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 14:15:00 | 581.15 | 581.94 | 581.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 575.95 | 579.85 | 580.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 570.00 | 568.98 | 572.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 09:30:00 | 570.85 | 568.98 | 572.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 12:15:00 | 570.50 | 569.16 | 571.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 12:45:00 | 570.45 | 569.16 | 571.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 13:15:00 | 569.80 | 569.29 | 571.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 15:00:00 | 567.75 | 568.98 | 571.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 10:15:00 | 578.80 | 566.91 | 567.73 | SL hit (close>static) qty=1.00 sl=572.50 alert=retest2 |

### Cycle 49 — BUY (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 11:15:00 | 575.90 | 568.70 | 568.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 09:15:00 | 587.40 | 576.88 | 572.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 10:15:00 | 622.35 | 623.50 | 613.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 11:00:00 | 622.35 | 623.50 | 613.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 619.65 | 622.56 | 618.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 12:30:00 | 625.15 | 623.28 | 619.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 13:15:00 | 628.25 | 623.28 | 619.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 14:15:00 | 626.30 | 623.41 | 619.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-06 14:15:00 | 635.25 | 640.12 | 640.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 14:15:00 | 635.25 | 640.12 | 640.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 09:15:00 | 605.00 | 632.76 | 636.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 10:15:00 | 592.75 | 587.42 | 596.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-12 11:00:00 | 592.75 | 587.42 | 596.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 597.00 | 589.34 | 596.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 12:45:00 | 587.05 | 588.39 | 595.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 14:00:00 | 589.60 | 588.63 | 595.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 11:30:00 | 590.75 | 585.67 | 587.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 13:45:00 | 590.65 | 588.17 | 588.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 589.00 | 588.35 | 588.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 595.85 | 588.35 | 588.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-15 09:15:00 | 597.95 | 590.27 | 589.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 597.95 | 590.27 | 589.40 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 15:15:00 | 587.50 | 589.44 | 589.61 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 598.80 | 591.31 | 590.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 09:15:00 | 607.25 | 597.71 | 594.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 599.00 | 601.28 | 597.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 15:00:00 | 599.00 | 601.28 | 597.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 600.85 | 601.19 | 598.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 648.10 | 601.19 | 598.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-26 13:15:00 | 626.80 | 631.80 | 631.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 13:15:00 | 626.80 | 631.80 | 631.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 622.45 | 628.26 | 629.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 628.00 | 627.30 | 628.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 10:15:00 | 628.00 | 627.30 | 628.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 628.00 | 627.30 | 628.51 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 632.80 | 629.22 | 629.14 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 09:15:00 | 626.65 | 628.80 | 628.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 14:15:00 | 623.40 | 626.74 | 627.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 629.00 | 626.74 | 627.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 629.00 | 626.74 | 627.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 629.00 | 626.74 | 627.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 10:15:00 | 622.55 | 625.80 | 626.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 11:00:00 | 622.60 | 625.16 | 626.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 591.42 | 599.32 | 604.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 591.47 | 599.32 | 604.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 09:15:00 | 560.29 | 581.38 | 592.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 10:15:00 | 564.00 | 556.66 | 556.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 12:15:00 | 565.10 | 559.33 | 557.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 557.00 | 560.55 | 558.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 557.00 | 560.55 | 558.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 557.00 | 560.55 | 558.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 10:00:00 | 557.00 | 560.55 | 558.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 555.00 | 559.44 | 558.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:00:00 | 555.00 | 559.44 | 558.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 12:15:00 | 553.75 | 557.34 | 557.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 15:15:00 | 550.95 | 554.73 | 556.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 13:15:00 | 551.25 | 550.43 | 553.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 14:00:00 | 551.25 | 550.43 | 553.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 553.05 | 548.69 | 551.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:45:00 | 553.60 | 548.69 | 551.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 554.25 | 549.80 | 551.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 551.75 | 549.80 | 551.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-28 12:45:00 | 551.65 | 549.84 | 551.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 565.75 | 551.78 | 551.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 565.75 | 551.78 | 551.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 576.20 | 564.01 | 558.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 11:15:00 | 589.35 | 590.07 | 585.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 12:00:00 | 589.35 | 590.07 | 585.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 586.90 | 590.27 | 587.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:00:00 | 586.90 | 590.27 | 587.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 586.50 | 589.51 | 587.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:30:00 | 586.30 | 589.51 | 587.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 11:15:00 | 589.35 | 589.48 | 587.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 12:45:00 | 590.00 | 589.96 | 587.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:15:00 | 591.00 | 589.19 | 588.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 14:15:00 | 609.75 | 613.53 | 613.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 14:15:00 | 609.75 | 613.53 | 613.64 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 09:15:00 | 619.65 | 614.35 | 613.97 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 15:15:00 | 609.90 | 613.94 | 614.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 603.65 | 610.05 | 612.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 11:15:00 | 606.60 | 606.15 | 609.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 12:00:00 | 606.60 | 606.15 | 609.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 615.00 | 607.48 | 608.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 10:00:00 | 614.10 | 608.81 | 609.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 617.75 | 610.60 | 610.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 11:15:00 | 626.10 | 619.05 | 615.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 13:15:00 | 619.00 | 620.15 | 616.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 14:00:00 | 619.00 | 620.15 | 616.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 619.25 | 619.97 | 616.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 14:45:00 | 617.00 | 619.97 | 616.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 617.80 | 619.54 | 616.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 628.30 | 619.54 | 616.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 14:15:00 | 622.45 | 625.56 | 625.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 14:15:00 | 622.45 | 625.56 | 625.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 10:15:00 | 618.70 | 623.43 | 624.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 11:15:00 | 625.20 | 623.78 | 624.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 11:15:00 | 625.20 | 623.78 | 624.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 625.20 | 623.78 | 624.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:00:00 | 625.20 | 623.78 | 624.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 629.00 | 624.83 | 625.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 13:00:00 | 629.00 | 624.83 | 625.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 13:15:00 | 629.00 | 625.66 | 625.50 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 619.00 | 624.33 | 624.91 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 15:15:00 | 627.00 | 624.58 | 624.56 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 09:15:00 | 620.70 | 623.80 | 624.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 13:15:00 | 612.95 | 620.32 | 622.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 15:15:00 | 619.70 | 619.25 | 621.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-06 09:15:00 | 619.45 | 619.25 | 621.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 615.20 | 618.44 | 620.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 11:15:00 | 607.20 | 615.68 | 617.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 12:30:00 | 607.35 | 612.73 | 616.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 13:45:00 | 607.65 | 611.86 | 615.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-08 13:15:00 | 626.40 | 617.63 | 616.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 13:15:00 | 626.40 | 617.63 | 616.56 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 13:15:00 | 610.50 | 615.61 | 616.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 09:15:00 | 608.90 | 613.49 | 615.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 09:15:00 | 609.65 | 607.85 | 610.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 09:15:00 | 609.65 | 607.85 | 610.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 609.65 | 607.85 | 610.64 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 621.15 | 612.61 | 611.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 624.90 | 618.32 | 615.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 632.80 | 634.96 | 631.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 632.80 | 634.96 | 631.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 632.80 | 634.96 | 631.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 663.90 | 637.22 | 634.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 10:15:00 | 641.70 | 645.43 | 644.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:15:00 | 641.20 | 644.09 | 643.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:45:00 | 644.10 | 644.78 | 644.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 12:15:00 | 637.60 | 643.35 | 643.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 12:15:00 | 637.60 | 643.35 | 643.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 13:15:00 | 629.75 | 640.63 | 642.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 633.80 | 630.74 | 633.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 633.80 | 630.74 | 633.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 633.80 | 630.74 | 633.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 633.80 | 630.74 | 633.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 636.95 | 631.98 | 633.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 647.85 | 631.98 | 633.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 644.00 | 634.39 | 634.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:15:00 | 653.00 | 634.39 | 634.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 660.80 | 639.67 | 636.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 11:15:00 | 675.10 | 646.76 | 640.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 09:15:00 | 670.80 | 675.50 | 666.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:45:00 | 670.90 | 675.50 | 666.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 670.55 | 674.62 | 669.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 670.55 | 674.62 | 669.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 667.00 | 673.10 | 669.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 673.45 | 673.10 | 669.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 10:00:00 | 671.70 | 676.85 | 673.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 651.70 | 671.82 | 671.68 | SL hit (close<static) qty=1.00 sl=665.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 648.05 | 667.07 | 669.53 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 698.65 | 669.80 | 668.82 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 09:15:00 | 704.40 | 707.41 | 707.63 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 10:15:00 | 712.30 | 708.39 | 708.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 11:15:00 | 721.20 | 710.95 | 709.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 11:15:00 | 719.00 | 719.39 | 715.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 12:00:00 | 719.00 | 719.39 | 715.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 722.75 | 720.06 | 716.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 14:30:00 | 725.85 | 721.97 | 717.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:15:00 | 725.95 | 722.48 | 718.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 13:15:00 | 798.44 | 755.78 | 739.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 772.50 | 777.94 | 778.47 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 790.00 | 780.35 | 779.51 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 772.10 | 777.83 | 778.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 766.15 | 774.82 | 777.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 15:15:00 | 756.70 | 755.85 | 762.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 15:15:00 | 756.70 | 755.85 | 762.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 756.70 | 755.85 | 762.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 759.40 | 755.85 | 762.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 758.00 | 756.28 | 761.66 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 766.20 | 762.75 | 762.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 15:15:00 | 770.00 | 766.58 | 764.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 765.60 | 766.38 | 764.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 10:00:00 | 765.60 | 766.38 | 764.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 765.30 | 766.16 | 764.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:45:00 | 764.60 | 766.16 | 764.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 755.95 | 764.12 | 764.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 755.95 | 764.12 | 764.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 750.15 | 761.33 | 762.79 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 761.60 | 760.14 | 760.01 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 759.10 | 759.88 | 759.93 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 764.70 | 760.84 | 760.36 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 13:15:00 | 755.95 | 759.41 | 759.77 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 778.15 | 763.02 | 761.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 787.35 | 775.84 | 769.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 13:15:00 | 781.90 | 783.01 | 775.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 14:00:00 | 781.90 | 783.01 | 775.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 772.30 | 780.86 | 775.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:45:00 | 773.55 | 780.86 | 775.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 776.95 | 780.08 | 775.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:45:00 | 767.50 | 776.46 | 774.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 759.30 | 773.02 | 772.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 753.65 | 773.02 | 772.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 762.95 | 771.01 | 772.06 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 787.95 | 774.48 | 772.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 793.20 | 781.43 | 776.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 13:15:00 | 793.05 | 793.72 | 788.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:00:00 | 793.05 | 793.72 | 788.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 793.95 | 794.47 | 792.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 794.95 | 794.47 | 792.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 795.85 | 794.75 | 792.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 790.05 | 794.75 | 792.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 787.20 | 793.24 | 791.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 787.20 | 793.24 | 791.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 792.00 | 792.99 | 791.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 787.85 | 792.99 | 791.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 793.40 | 793.07 | 792.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:15:00 | 795.00 | 793.07 | 792.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 770.00 | 789.41 | 790.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 770.00 | 789.41 | 790.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 768.75 | 785.28 | 788.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 15:15:00 | 781.00 | 777.23 | 782.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 15:15:00 | 781.00 | 777.23 | 782.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 781.00 | 777.23 | 782.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 786.00 | 779.16 | 783.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 786.40 | 780.61 | 783.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 786.40 | 780.61 | 783.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 786.20 | 781.80 | 783.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 786.20 | 781.80 | 783.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 786.05 | 782.65 | 783.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 786.95 | 782.65 | 783.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 789.90 | 784.99 | 784.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 795.00 | 786.63 | 785.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 11:15:00 | 781.70 | 785.65 | 785.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 781.70 | 785.65 | 785.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 781.70 | 785.65 | 785.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:45:00 | 780.05 | 785.65 | 785.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 763.95 | 781.31 | 783.13 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 781.75 | 776.29 | 775.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 836.00 | 789.87 | 782.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 12:15:00 | 827.00 | 827.48 | 813.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:30:00 | 827.55 | 827.48 | 813.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 819.05 | 823.15 | 819.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 819.05 | 823.15 | 819.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 819.20 | 822.36 | 819.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 823.70 | 822.36 | 819.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 09:15:00 | 816.00 | 821.09 | 818.90 | SL hit (close<static) qty=1.00 sl=819.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 806.70 | 816.62 | 817.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 803.35 | 812.36 | 815.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 772.45 | 760.54 | 772.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 772.45 | 760.54 | 772.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 772.45 | 760.54 | 772.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 772.45 | 760.54 | 772.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 766.00 | 761.63 | 772.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 761.30 | 763.87 | 770.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 773.55 | 765.58 | 765.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 773.55 | 765.58 | 765.32 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 755.75 | 764.34 | 764.96 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 776.60 | 766.79 | 766.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 799.15 | 779.50 | 773.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 780.00 | 791.05 | 784.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 780.00 | 791.05 | 784.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 780.00 | 791.05 | 784.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 780.00 | 791.05 | 784.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 779.60 | 788.76 | 783.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 778.35 | 788.76 | 783.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 788.85 | 788.31 | 784.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 782.95 | 788.31 | 784.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 782.75 | 787.20 | 784.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 782.75 | 787.20 | 784.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 776.30 | 785.02 | 783.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 776.30 | 785.02 | 783.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 768.75 | 780.00 | 781.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 11:15:00 | 761.75 | 773.79 | 778.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 760.70 | 758.39 | 764.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 15:00:00 | 760.70 | 758.39 | 764.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 787.95 | 764.56 | 766.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 787.95 | 764.56 | 766.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 783.20 | 768.29 | 767.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 14:15:00 | 806.00 | 780.61 | 774.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 11:15:00 | 810.15 | 812.15 | 800.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 12:00:00 | 810.15 | 812.15 | 800.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 810.80 | 813.18 | 808.88 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 12:15:00 | 801.25 | 811.20 | 812.11 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 10:15:00 | 815.90 | 812.24 | 812.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 837.05 | 819.12 | 815.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 15:15:00 | 824.45 | 828.01 | 822.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 15:15:00 | 824.45 | 828.01 | 822.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 824.45 | 828.01 | 822.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 10:45:00 | 833.45 | 829.19 | 824.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 15:15:00 | 818.00 | 824.09 | 824.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 15:15:00 | 818.00 | 824.09 | 824.31 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 827.80 | 824.83 | 824.62 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 814.25 | 822.72 | 823.68 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 13:15:00 | 834.55 | 825.79 | 824.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 840.90 | 831.16 | 827.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 10:15:00 | 836.70 | 837.95 | 834.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 10:15:00 | 836.70 | 837.95 | 834.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 836.70 | 837.95 | 834.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:15:00 | 831.50 | 837.95 | 834.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 836.70 | 837.70 | 834.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:00:00 | 840.00 | 838.16 | 834.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:30:00 | 839.30 | 838.92 | 835.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:45:00 | 840.95 | 839.03 | 836.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 10:15:00 | 840.70 | 839.03 | 836.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 843.95 | 840.01 | 837.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 854.55 | 841.80 | 839.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 14:00:00 | 847.55 | 845.02 | 842.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 832.50 | 840.95 | 841.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 10:15:00 | 832.50 | 840.95 | 841.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 11:15:00 | 831.50 | 839.06 | 840.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 12:15:00 | 812.60 | 811.43 | 817.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 13:00:00 | 812.60 | 811.43 | 817.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 827.45 | 814.40 | 817.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 827.45 | 814.40 | 817.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 840.25 | 819.57 | 819.34 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 818.25 | 822.65 | 823.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 808.95 | 819.91 | 821.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 809.65 | 807.40 | 812.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 809.65 | 807.40 | 812.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 811.95 | 808.31 | 812.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 829.20 | 808.31 | 812.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 827.95 | 812.24 | 813.50 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 828.50 | 815.49 | 814.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 833.60 | 824.48 | 819.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 14:15:00 | 828.00 | 828.69 | 824.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 15:00:00 | 828.00 | 828.69 | 824.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 828.50 | 830.35 | 827.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 828.50 | 830.35 | 827.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 827.65 | 829.81 | 827.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 827.65 | 829.81 | 827.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 822.10 | 828.27 | 827.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 828.85 | 828.27 | 827.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 10:30:00 | 828.20 | 828.44 | 827.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 12:00:00 | 828.55 | 828.46 | 827.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:00:00 | 830.95 | 829.04 | 827.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 824.50 | 829.45 | 828.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 824.10 | 829.45 | 828.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 823.75 | 828.31 | 828.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 823.10 | 828.31 | 828.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-26 11:15:00 | 824.40 | 827.53 | 827.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 11:15:00 | 824.40 | 827.53 | 827.73 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 830.00 | 827.70 | 827.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 853.60 | 832.88 | 830.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 842.25 | 845.85 | 838.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 15:00:00 | 842.25 | 845.85 | 838.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 839.40 | 844.81 | 839.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 836.80 | 844.81 | 839.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 837.05 | 843.26 | 839.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 837.05 | 843.26 | 839.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 835.30 | 841.67 | 838.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:30:00 | 839.95 | 841.67 | 838.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 835.50 | 840.43 | 838.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 835.50 | 840.43 | 838.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 854.55 | 843.26 | 840.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:30:00 | 844.65 | 843.26 | 840.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 864.60 | 864.13 | 856.10 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 848.30 | 853.31 | 853.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 846.05 | 851.86 | 853.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 813.35 | 806.71 | 819.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 813.35 | 806.71 | 819.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 818.10 | 808.99 | 819.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 818.10 | 808.99 | 819.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 821.30 | 811.45 | 819.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 821.30 | 811.45 | 819.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 820.75 | 813.31 | 819.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 831.70 | 813.31 | 819.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 827.50 | 816.15 | 820.21 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 832.90 | 824.34 | 823.38 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 812.70 | 821.76 | 822.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 806.70 | 818.75 | 821.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 817.00 | 814.41 | 817.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 817.00 | 814.41 | 817.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 817.00 | 814.41 | 817.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 815.15 | 814.41 | 817.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 812.95 | 814.12 | 817.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 14:15:00 | 810.45 | 813.11 | 815.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 808.85 | 812.64 | 815.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 825.90 | 813.55 | 813.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 825.90 | 813.55 | 813.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 13:15:00 | 832.05 | 820.55 | 816.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 818.35 | 823.03 | 819.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 818.35 | 823.03 | 819.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 818.35 | 823.03 | 819.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 818.35 | 823.03 | 819.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 814.25 | 821.28 | 818.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:30:00 | 816.25 | 821.28 | 818.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 813.85 | 819.79 | 818.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 810.75 | 819.79 | 818.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 812.15 | 817.85 | 817.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 812.15 | 817.85 | 817.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 812.40 | 816.76 | 817.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 804.40 | 812.11 | 814.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 802.50 | 801.78 | 806.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 14:00:00 | 802.50 | 801.78 | 806.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 776.35 | 772.14 | 776.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:00:00 | 776.35 | 772.14 | 776.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 773.45 | 772.40 | 775.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:15:00 | 770.50 | 773.78 | 775.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 785.85 | 769.55 | 767.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 785.85 | 769.55 | 767.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 796.70 | 785.80 | 777.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 791.30 | 803.38 | 795.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 791.30 | 803.38 | 795.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 791.30 | 803.38 | 795.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 791.30 | 803.38 | 795.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 785.50 | 799.81 | 794.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 785.50 | 799.81 | 794.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 788.30 | 797.51 | 794.09 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 14:15:00 | 781.25 | 790.47 | 791.38 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 812.45 | 793.83 | 792.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 816.60 | 808.62 | 802.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 841.45 | 844.03 | 832.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 10:00:00 | 841.45 | 844.03 | 832.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 830.15 | 839.89 | 832.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 830.15 | 839.89 | 832.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 829.85 | 837.88 | 831.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:45:00 | 828.75 | 837.88 | 831.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 827.20 | 835.74 | 831.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 827.20 | 835.74 | 831.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 833.60 | 835.32 | 831.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 830.30 | 835.32 | 831.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 827.35 | 833.72 | 831.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 849.20 | 833.72 | 831.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 11:15:00 | 824.20 | 829.74 | 829.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 824.20 | 829.74 | 829.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 817.00 | 826.17 | 828.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 827.05 | 822.49 | 825.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 827.05 | 822.49 | 825.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 827.05 | 822.49 | 825.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 833.45 | 822.49 | 825.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 823.50 | 822.69 | 825.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 826.95 | 822.69 | 825.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 823.10 | 822.78 | 825.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:45:00 | 823.75 | 822.78 | 825.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 819.65 | 822.15 | 824.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:30:00 | 824.45 | 822.15 | 824.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 820.00 | 820.23 | 823.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 795.75 | 820.23 | 823.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 10:15:00 | 755.96 | 775.83 | 787.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 809.00 | 777.60 | 782.25 | SL hit (close>ema200) qty=0.50 sl=777.60 alert=retest2 |

### Cycle 121 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 816.65 | 791.59 | 788.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 820.25 | 797.32 | 791.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 833.90 | 833.90 | 826.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:00:00 | 833.90 | 833.90 | 826.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 852.75 | 857.13 | 850.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:30:00 | 850.35 | 857.13 | 850.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 852.30 | 855.66 | 850.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 852.30 | 855.66 | 850.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 852.00 | 854.93 | 850.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 862.10 | 854.93 | 850.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:45:00 | 855.40 | 862.20 | 859.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:15:00 | 856.95 | 859.44 | 858.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-12 15:15:00 | 940.94 | 930.78 | 921.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 13:15:00 | 921.30 | 928.96 | 929.58 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 935.50 | 930.02 | 929.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 09:15:00 | 984.10 | 941.36 | 935.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 943.00 | 964.29 | 953.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 943.00 | 964.29 | 953.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 943.00 | 964.29 | 953.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:30:00 | 954.80 | 964.29 | 953.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 951.60 | 961.75 | 953.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:00:00 | 958.00 | 958.64 | 953.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 958.65 | 961.84 | 960.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 09:15:00 | 922.60 | 952.34 | 956.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 922.60 | 952.34 | 956.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 14:15:00 | 917.00 | 932.78 | 943.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 931.55 | 930.35 | 940.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 09:45:00 | 934.35 | 930.35 | 940.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 900.20 | 897.77 | 907.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 911.05 | 897.77 | 907.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 898.10 | 900.41 | 905.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 901.60 | 900.41 | 905.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 893.20 | 883.58 | 889.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 893.20 | 883.58 | 889.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 894.30 | 885.72 | 890.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 892.75 | 885.72 | 890.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 915.60 | 895.46 | 893.78 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 892.95 | 907.94 | 908.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 890.25 | 904.40 | 907.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 891.35 | 888.78 | 894.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 14:00:00 | 891.35 | 888.78 | 894.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 814.85 | 799.99 | 808.50 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 816.85 | 813.22 | 813.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 842.80 | 821.06 | 816.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 15:15:00 | 849.90 | 851.31 | 845.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:15:00 | 860.25 | 851.31 | 845.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 829.50 | 846.32 | 844.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 829.50 | 846.32 | 844.41 | SL hit (close<ema400) qty=1.00 sl=844.41 alert=retest1 |

### Cycle 128 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 831.10 | 841.49 | 842.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 830.25 | 839.24 | 841.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 821.20 | 820.19 | 828.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 821.20 | 820.19 | 828.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 841.50 | 825.16 | 829.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 841.50 | 825.16 | 829.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 841.60 | 828.45 | 830.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 843.30 | 828.45 | 830.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 844.70 | 834.02 | 832.82 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 830.00 | 833.98 | 833.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 822.00 | 831.58 | 832.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 786.85 | 784.98 | 800.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 786.85 | 784.98 | 800.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 798.40 | 789.58 | 800.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 798.40 | 789.58 | 800.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 788.75 | 789.41 | 799.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 801.35 | 789.41 | 799.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 787.50 | 789.03 | 798.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 811.85 | 795.22 | 800.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 820.00 | 800.18 | 801.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 819.00 | 800.18 | 801.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 820.00 | 804.14 | 803.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 830.75 | 817.40 | 811.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 813.70 | 817.25 | 813.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 813.70 | 817.25 | 813.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 813.70 | 817.25 | 813.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 813.70 | 817.25 | 813.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 820.65 | 817.93 | 813.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:30:00 | 822.45 | 817.42 | 815.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 811.15 | 817.17 | 816.35 | SL hit (close<static) qty=1.00 sl=812.25 alert=retest2 |

### Cycle 132 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 812.80 | 815.80 | 815.90 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 09:15:00 | 819.80 | 816.60 | 816.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 10:15:00 | 830.25 | 819.33 | 817.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 09:15:00 | 835.60 | 837.32 | 831.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 10:15:00 | 836.25 | 837.32 | 831.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 848.50 | 855.65 | 847.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 845.25 | 855.65 | 847.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 849.50 | 854.42 | 848.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 15:15:00 | 855.05 | 848.68 | 846.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 823.25 | 844.61 | 845.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 823.25 | 844.61 | 845.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 820.20 | 839.73 | 843.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 794.60 | 792.50 | 805.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 794.60 | 792.50 | 805.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 806.90 | 794.87 | 802.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 806.85 | 794.87 | 802.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 797.35 | 795.37 | 801.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 791.45 | 795.13 | 800.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 751.88 | 777.62 | 789.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-17 10:15:00 | 712.31 | 741.85 | 763.50 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 135 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 679.70 | 668.03 | 666.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 682.30 | 673.89 | 669.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 698.10 | 699.74 | 691.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 698.10 | 699.74 | 691.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 694.85 | 698.03 | 692.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 691.85 | 698.03 | 692.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 695.25 | 697.47 | 692.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:45:00 | 694.40 | 697.47 | 692.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 691.00 | 696.18 | 692.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:15:00 | 690.00 | 696.18 | 692.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 682.25 | 693.39 | 691.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 682.25 | 693.39 | 691.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 683.20 | 689.44 | 689.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 678.95 | 684.79 | 687.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 687.00 | 683.26 | 685.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 687.00 | 683.26 | 685.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 687.00 | 683.26 | 685.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 693.30 | 683.26 | 685.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 698.25 | 686.26 | 686.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 698.25 | 686.26 | 686.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 699.50 | 688.91 | 687.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 14:15:00 | 701.35 | 693.88 | 690.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 12:15:00 | 696.40 | 696.52 | 693.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 13:00:00 | 696.40 | 696.52 | 693.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 696.15 | 696.20 | 693.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:30:00 | 695.45 | 696.20 | 693.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 690.00 | 694.96 | 693.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 697.80 | 694.96 | 693.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 09:15:00 | 767.58 | 755.80 | 747.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 10:15:00 | 782.05 | 784.23 | 784.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 12:15:00 | 776.65 | 782.09 | 783.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 782.30 | 781.50 | 782.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 14:15:00 | 782.30 | 781.50 | 782.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 782.30 | 781.50 | 782.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 782.30 | 781.50 | 782.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 783.00 | 781.80 | 782.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 790.75 | 781.80 | 782.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 785.15 | 782.47 | 783.02 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 788.85 | 783.75 | 783.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 11:15:00 | 799.55 | 786.91 | 785.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 773.90 | 789.39 | 787.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 773.90 | 789.39 | 787.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 773.90 | 789.39 | 787.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 773.90 | 789.39 | 787.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 783.30 | 788.17 | 787.32 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 779.00 | 786.34 | 786.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 775.85 | 784.24 | 785.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 743.10 | 743.00 | 757.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 757.20 | 743.00 | 757.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 754.20 | 745.24 | 757.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:15:00 | 762.50 | 745.24 | 757.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 760.80 | 748.35 | 757.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:30:00 | 760.65 | 748.35 | 757.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 761.65 | 751.01 | 758.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:30:00 | 762.90 | 751.01 | 758.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 762.00 | 758.48 | 760.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:15:00 | 754.50 | 758.48 | 760.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 750.60 | 756.90 | 759.15 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 777.25 | 758.58 | 758.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 807.80 | 781.12 | 771.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 10:15:00 | 834.10 | 834.40 | 820.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:45:00 | 833.80 | 834.40 | 820.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 852.45 | 852.23 | 844.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:00:00 | 852.45 | 852.23 | 844.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 842.40 | 850.03 | 845.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 842.40 | 850.03 | 845.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 846.95 | 849.41 | 845.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:30:00 | 848.95 | 849.27 | 845.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:00:00 | 848.70 | 849.27 | 845.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:30:00 | 848.65 | 849.38 | 846.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:00:00 | 855.35 | 849.10 | 846.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 842.20 | 856.34 | 853.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 842.20 | 856.34 | 853.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 835.45 | 852.16 | 851.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 835.45 | 852.16 | 851.68 | SL hit (close<static) qty=1.00 sl=842.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 839.55 | 849.64 | 850.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 828.75 | 836.97 | 841.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 832.10 | 827.14 | 833.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 832.10 | 827.14 | 833.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 832.10 | 827.14 | 833.88 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 10:15:00 | 845.70 | 833.89 | 832.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 11:15:00 | 851.00 | 837.31 | 834.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 828.80 | 839.88 | 837.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 828.80 | 839.88 | 837.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 828.80 | 839.88 | 837.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:00:00 | 828.80 | 839.88 | 837.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 843.70 | 840.64 | 837.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 846.65 | 840.64 | 837.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:00:00 | 847.30 | 844.12 | 840.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 14:15:00 | 845.85 | 849.81 | 845.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 826.50 | 840.86 | 842.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 826.50 | 840.86 | 842.33 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 871.00 | 843.08 | 841.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 876.30 | 849.73 | 844.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 980.80 | 980.99 | 967.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 13:45:00 | 981.75 | 980.99 | 967.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 972.25 | 978.79 | 970.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:00:00 | 981.75 | 977.59 | 972.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 984.95 | 979.06 | 973.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:30:00 | 981.70 | 978.61 | 974.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 12:00:00 | 979.80 | 978.85 | 974.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 973.30 | 977.74 | 974.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 973.30 | 977.74 | 974.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 980.00 | 978.19 | 975.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:30:00 | 982.50 | 979.02 | 975.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 990.65 | 979.39 | 976.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:45:00 | 985.80 | 989.49 | 989.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 984.45 | 988.48 | 988.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 984.45 | 988.48 | 988.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 978.75 | 986.54 | 987.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 989.00 | 985.57 | 986.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 989.00 | 985.57 | 986.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 989.00 | 985.57 | 986.83 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 989.00 | 987.56 | 987.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 13:15:00 | 993.05 | 988.66 | 988.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 986.95 | 988.32 | 987.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 986.95 | 988.32 | 987.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 986.95 | 988.32 | 987.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 986.95 | 988.32 | 987.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 989.00 | 988.45 | 988.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 975.30 | 988.45 | 988.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 970.00 | 984.76 | 986.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 964.15 | 975.17 | 980.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 957.60 | 957.41 | 964.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:45:00 | 958.55 | 957.41 | 964.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 986.75 | 961.59 | 964.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 986.75 | 961.59 | 964.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 984.75 | 966.22 | 965.95 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 973.00 | 975.35 | 975.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 970.90 | 974.09 | 974.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 974.55 | 972.99 | 974.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 10:15:00 | 974.55 | 972.99 | 974.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 974.55 | 972.99 | 974.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 974.55 | 972.99 | 974.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 973.80 | 973.15 | 974.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:00:00 | 969.70 | 972.35 | 973.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 09:15:00 | 977.30 | 972.33 | 973.19 | SL hit (close>static) qty=1.00 sl=976.70 alert=retest2 |

### Cycle 151 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 977.35 | 973.78 | 973.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 999.05 | 979.02 | 976.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 989.45 | 999.73 | 992.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 989.45 | 999.73 | 992.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 989.45 | 999.73 | 992.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 989.45 | 999.73 | 992.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 989.75 | 997.73 | 992.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 989.75 | 997.73 | 992.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 984.50 | 995.09 | 991.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 978.65 | 995.09 | 991.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 985.00 | 990.27 | 990.23 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 982.00 | 988.62 | 989.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 965.95 | 982.65 | 986.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 979.70 | 972.87 | 978.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 979.70 | 972.87 | 978.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 979.70 | 972.87 | 978.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 980.95 | 972.87 | 978.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 967.90 | 971.87 | 977.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 966.15 | 969.83 | 975.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 962.25 | 958.36 | 962.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 974.60 | 962.13 | 960.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 974.60 | 962.13 | 960.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 979.05 | 965.51 | 962.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 964.25 | 968.43 | 965.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 964.25 | 968.43 | 965.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 964.25 | 968.43 | 965.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 966.35 | 968.43 | 965.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 971.90 | 969.12 | 966.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:15:00 | 975.55 | 969.12 | 966.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:15:00 | 974.00 | 971.57 | 968.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-26 09:15:00 | 1073.11 | 1028.36 | 1004.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 1080.90 | 1095.54 | 1097.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1073.60 | 1088.40 | 1092.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 1080.20 | 1078.30 | 1085.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 1080.20 | 1078.30 | 1085.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1085.40 | 1080.24 | 1084.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:15:00 | 1088.10 | 1080.24 | 1084.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1085.90 | 1081.37 | 1084.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 1089.30 | 1081.37 | 1084.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1080.00 | 1081.10 | 1084.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 1077.00 | 1083.45 | 1084.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 11:15:00 | 1090.50 | 1084.74 | 1084.98 | SL hit (close>static) qty=1.00 sl=1088.70 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 12:15:00 | 1101.50 | 1088.09 | 1086.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 13:15:00 | 1105.30 | 1091.53 | 1088.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 1156.00 | 1156.84 | 1144.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 13:30:00 | 1154.20 | 1156.84 | 1144.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1162.00 | 1158.97 | 1148.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 1166.30 | 1158.97 | 1148.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 1163.20 | 1159.78 | 1150.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 1145.50 | 1151.07 | 1151.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 1145.50 | 1151.07 | 1151.46 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 1155.10 | 1151.81 | 1151.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 12:15:00 | 1173.90 | 1159.37 | 1155.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 13:15:00 | 1162.40 | 1167.62 | 1162.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 13:15:00 | 1162.40 | 1167.62 | 1162.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1162.40 | 1167.62 | 1162.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 1162.40 | 1167.62 | 1162.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1168.90 | 1167.88 | 1163.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 13:00:00 | 1178.60 | 1171.89 | 1166.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 1162.60 | 1167.66 | 1167.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 1162.60 | 1167.66 | 1167.98 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 1183.80 | 1170.89 | 1169.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 1206.60 | 1184.11 | 1178.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 13:15:00 | 1228.00 | 1232.15 | 1219.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 14:00:00 | 1228.00 | 1232.15 | 1219.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1228.50 | 1231.12 | 1222.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 1219.60 | 1231.12 | 1222.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1224.70 | 1229.44 | 1224.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 1224.70 | 1229.44 | 1224.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1196.30 | 1222.81 | 1221.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1196.30 | 1222.81 | 1221.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 1203.00 | 1218.85 | 1220.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 1195.70 | 1214.22 | 1217.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 1199.90 | 1199.29 | 1204.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 15:00:00 | 1199.90 | 1199.29 | 1204.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1179.20 | 1192.38 | 1200.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 1143.40 | 1185.86 | 1191.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 1086.23 | 1135.46 | 1156.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 1137.00 | 1135.77 | 1154.58 | SL hit (close>ema200) qty=0.50 sl=1135.77 alert=retest2 |

### Cycle 161 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 1161.20 | 1152.24 | 1151.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 1171.10 | 1156.01 | 1153.61 | Break + close above crossover candle high |

### Cycle 162 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 1134.70 | 1152.39 | 1152.42 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1157.00 | 1152.76 | 1152.47 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 1138.70 | 1150.45 | 1151.50 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1159.80 | 1153.07 | 1152.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 1166.10 | 1159.65 | 1156.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 1156.00 | 1167.45 | 1164.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 13:15:00 | 1156.00 | 1167.45 | 1164.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1156.00 | 1167.45 | 1164.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 1156.00 | 1167.45 | 1164.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1146.30 | 1163.22 | 1162.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1146.30 | 1163.22 | 1162.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 1146.80 | 1159.93 | 1161.34 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 1171.50 | 1161.92 | 1160.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 11:15:00 | 1175.60 | 1170.91 | 1166.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 1182.90 | 1183.05 | 1177.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 15:00:00 | 1182.90 | 1183.05 | 1177.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1165.20 | 1179.48 | 1176.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1142.00 | 1179.48 | 1176.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1144.50 | 1172.48 | 1173.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1130.80 | 1150.58 | 1160.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1126.10 | 1125.18 | 1137.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1126.10 | 1125.18 | 1137.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1125.00 | 1125.20 | 1132.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 1115.00 | 1125.20 | 1132.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:45:00 | 1123.20 | 1125.15 | 1131.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 1135.40 | 1126.26 | 1130.31 | SL hit (close>static) qty=1.00 sl=1134.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 14:15:00 | 1128.10 | 1124.26 | 1124.09 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1120.30 | 1123.47 | 1123.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 1112.90 | 1119.98 | 1121.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1096.50 | 1077.53 | 1085.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1096.50 | 1077.53 | 1085.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1096.50 | 1077.53 | 1085.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1096.50 | 1077.53 | 1085.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1088.10 | 1079.64 | 1085.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 1087.20 | 1085.40 | 1087.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 1085.10 | 1086.00 | 1087.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:45:00 | 1087.40 | 1085.44 | 1086.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:00:00 | 1085.40 | 1084.42 | 1086.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1085.00 | 1084.53 | 1086.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:30:00 | 1085.40 | 1084.53 | 1086.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1085.10 | 1084.65 | 1085.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 1085.90 | 1084.65 | 1085.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1085.10 | 1084.74 | 1085.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:30:00 | 1085.80 | 1084.74 | 1085.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 1079.00 | 1083.59 | 1085.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 1079.00 | 1083.59 | 1085.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1078.10 | 1082.49 | 1084.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1089.00 | 1082.66 | 1081.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 1089.00 | 1082.66 | 1081.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 1089.70 | 1084.07 | 1082.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 1086.20 | 1092.19 | 1089.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1086.20 | 1092.19 | 1089.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1086.20 | 1092.19 | 1089.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1086.20 | 1092.19 | 1089.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1095.70 | 1092.89 | 1089.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 1101.10 | 1094.55 | 1092.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 1099.00 | 1096.07 | 1093.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 1083.90 | 1094.17 | 1092.86 | SL hit (close<static) qty=1.00 sl=1084.60 alert=retest2 |

### Cycle 172 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1065.20 | 1088.38 | 1090.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1059.20 | 1069.17 | 1075.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1073.10 | 1069.32 | 1074.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1073.10 | 1069.32 | 1074.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1073.10 | 1069.32 | 1074.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 1073.60 | 1069.32 | 1074.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1067.80 | 1069.01 | 1073.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:30:00 | 1065.00 | 1070.20 | 1071.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:30:00 | 1064.90 | 1068.62 | 1070.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 1011.75 | 1040.34 | 1052.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 1011.66 | 1040.34 | 1052.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 1024.90 | 1020.89 | 1032.30 | SL hit (close>ema200) qty=0.50 sl=1020.89 alert=retest2 |

### Cycle 173 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1036.00 | 1030.79 | 1030.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1040.90 | 1032.81 | 1031.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 1038.30 | 1043.76 | 1039.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 12:15:00 | 1038.30 | 1043.76 | 1039.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1038.30 | 1043.76 | 1039.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:30:00 | 1040.10 | 1043.76 | 1039.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1043.00 | 1043.61 | 1040.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 1037.10 | 1043.61 | 1040.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 1041.00 | 1043.52 | 1040.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:45:00 | 1051.00 | 1045.15 | 1041.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1049.50 | 1047.18 | 1044.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 1053.80 | 1048.48 | 1046.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 1050.20 | 1053.56 | 1050.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1043.60 | 1051.56 | 1049.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 1042.00 | 1051.56 | 1049.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1035.80 | 1048.41 | 1048.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 1035.80 | 1048.41 | 1048.18 | SL hit (close<static) qty=1.00 sl=1040.10 alert=retest2 |

### Cycle 174 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 1043.10 | 1047.35 | 1047.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 1035.30 | 1043.48 | 1045.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 1035.30 | 1033.39 | 1037.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:30:00 | 1033.80 | 1033.39 | 1037.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1049.50 | 1037.00 | 1038.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 1049.50 | 1037.00 | 1038.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1035.70 | 1036.74 | 1038.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:45:00 | 1032.00 | 1037.44 | 1038.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 14:15:00 | 1041.00 | 1038.30 | 1038.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 1041.00 | 1038.30 | 1038.27 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 1035.50 | 1037.74 | 1038.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 1029.90 | 1034.76 | 1036.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1035.40 | 1034.38 | 1035.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1035.40 | 1034.38 | 1035.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1035.40 | 1034.38 | 1035.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 1036.50 | 1034.38 | 1035.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1040.00 | 1035.50 | 1036.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 1043.00 | 1035.50 | 1036.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1033.80 | 1035.16 | 1035.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 1032.70 | 1035.16 | 1035.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:30:00 | 1031.50 | 1034.90 | 1035.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 1032.90 | 1034.38 | 1035.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 1032.30 | 1034.38 | 1035.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 1025.90 | 1022.21 | 1025.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 1025.00 | 1022.21 | 1025.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 1025.00 | 1022.77 | 1025.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 1025.50 | 1022.77 | 1025.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1026.90 | 1023.60 | 1025.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 1026.90 | 1023.60 | 1025.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1030.00 | 1024.88 | 1026.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 1037.80 | 1027.46 | 1027.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 1037.80 | 1027.46 | 1027.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 1041.00 | 1030.17 | 1028.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1043.10 | 1043.39 | 1037.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1043.10 | 1043.39 | 1037.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1044.70 | 1044.37 | 1038.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 1043.50 | 1044.37 | 1038.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1034.20 | 1042.34 | 1038.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 1034.20 | 1042.34 | 1038.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 1038.80 | 1041.63 | 1038.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 1031.40 | 1041.63 | 1038.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1036.90 | 1040.69 | 1038.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1036.90 | 1040.69 | 1038.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1035.60 | 1039.67 | 1038.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 1035.90 | 1039.67 | 1038.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1039.00 | 1039.53 | 1038.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:15:00 | 1043.00 | 1039.53 | 1038.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1072.90 | 1077.38 | 1077.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1072.90 | 1077.38 | 1077.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 1060.60 | 1072.24 | 1074.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 1028.10 | 1027.91 | 1036.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1028.10 | 1027.91 | 1036.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1028.10 | 1027.91 | 1036.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1015.70 | 1027.92 | 1033.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 1050.10 | 1025.01 | 1029.28 | SL hit (close>static) qty=1.00 sl=1039.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1071.80 | 1032.22 | 1031.06 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1026.80 | 1038.77 | 1039.23 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 1050.50 | 1036.86 | 1036.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 1059.10 | 1041.31 | 1038.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1027.80 | 1042.99 | 1040.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1027.80 | 1042.99 | 1040.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1027.80 | 1042.99 | 1040.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1027.80 | 1042.99 | 1040.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1025.50 | 1039.49 | 1039.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 1025.50 | 1039.49 | 1039.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 1026.80 | 1036.95 | 1038.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 13:15:00 | 1020.10 | 1025.51 | 1028.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 1020.00 | 1018.46 | 1021.97 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1013.50 | 1018.46 | 1021.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1025.90 | 1019.94 | 1022.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1025.90 | 1019.94 | 1022.33 | SL hit (close>ema400) qty=1.00 sl=1022.33 alert=retest1 |

### Cycle 183 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 1029.90 | 1024.53 | 1024.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1034.40 | 1027.28 | 1025.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 1030.30 | 1031.35 | 1029.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 1030.30 | 1031.35 | 1029.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1029.10 | 1030.90 | 1029.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 1029.10 | 1030.90 | 1029.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1029.10 | 1030.54 | 1029.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 1029.40 | 1030.54 | 1029.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1029.90 | 1030.41 | 1029.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1029.10 | 1030.41 | 1029.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1029.90 | 1030.31 | 1029.26 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 1026.30 | 1029.20 | 1029.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1024.20 | 1028.20 | 1028.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 1033.40 | 1024.63 | 1026.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 1033.40 | 1024.63 | 1026.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1033.40 | 1024.63 | 1026.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1033.40 | 1024.63 | 1026.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1039.00 | 1027.50 | 1027.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 1039.00 | 1027.50 | 1027.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 11:15:00 | 1035.00 | 1029.00 | 1028.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 1042.80 | 1033.89 | 1030.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 1029.50 | 1033.75 | 1031.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 1029.50 | 1033.75 | 1031.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1029.50 | 1033.75 | 1031.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 1024.80 | 1033.75 | 1031.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1029.00 | 1032.80 | 1031.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:00:00 | 1031.30 | 1032.50 | 1031.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:45:00 | 1034.60 | 1035.40 | 1033.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:45:00 | 1035.00 | 1034.99 | 1034.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 1030.60 | 1033.82 | 1033.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 13:15:00 | 1030.60 | 1033.82 | 1033.89 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 1034.90 | 1034.04 | 1033.99 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1019.90 | 1031.36 | 1032.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1014.40 | 1026.38 | 1030.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 1006.20 | 1003.20 | 1012.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 1006.20 | 1003.20 | 1012.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 1019.00 | 1006.36 | 1013.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 1028.00 | 1006.36 | 1013.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1031.50 | 1011.39 | 1014.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 1032.20 | 1011.39 | 1014.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 1045.30 | 1018.17 | 1017.58 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 1025.50 | 1032.02 | 1032.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1019.70 | 1025.46 | 1028.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1020.50 | 1018.82 | 1023.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1020.50 | 1018.82 | 1023.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1020.50 | 1018.82 | 1023.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1020.50 | 1018.82 | 1023.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1024.00 | 1020.38 | 1023.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 1023.00 | 1020.38 | 1023.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1027.80 | 1021.86 | 1023.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1027.80 | 1021.86 | 1023.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1027.00 | 1022.89 | 1024.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 1022.10 | 1022.89 | 1024.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:00:00 | 1022.30 | 1022.77 | 1023.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1041.60 | 1016.89 | 1018.46 | SL hit (close>static) qty=1.00 sl=1029.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1043.00 | 1022.11 | 1020.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 1051.70 | 1036.23 | 1029.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 1053.40 | 1055.63 | 1046.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 1053.40 | 1055.63 | 1046.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1052.80 | 1055.06 | 1046.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 1050.30 | 1055.06 | 1046.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1063.40 | 1067.03 | 1060.66 | EMA400 retest candle locked (from upside) |

### Cycle 192 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 1051.40 | 1058.29 | 1058.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 1041.00 | 1054.83 | 1056.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 1018.00 | 1015.10 | 1022.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 1009.40 | 1015.10 | 1022.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1006.40 | 1013.36 | 1021.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 1005.40 | 1011.81 | 1019.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1003.40 | 1011.81 | 1019.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 1005.50 | 1009.65 | 1016.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:45:00 | 1005.40 | 1009.60 | 1016.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1001.90 | 1001.99 | 1007.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 995.60 | 1000.07 | 1005.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 997.20 | 998.77 | 1003.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 993.40 | 999.01 | 1003.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 955.13 | 976.39 | 988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 953.23 | 976.39 | 988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 955.22 | 976.39 | 988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 955.13 | 976.39 | 988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 947.34 | 973.54 | 985.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 13:15:00 | 945.82 | 960.48 | 975.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 13:15:00 | 943.73 | 960.48 | 975.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 13:15:00 | 942.20 | 941.55 | 956.10 | SL hit (close>ema200) qty=0.50 sl=941.55 alert=retest2 |

### Cycle 193 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 900.60 | 894.28 | 893.94 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 888.80 | 893.81 | 893.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 885.00 | 892.05 | 893.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 895.20 | 890.83 | 892.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 895.20 | 890.83 | 892.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 895.20 | 890.83 | 892.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 895.20 | 890.83 | 892.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 904.60 | 893.59 | 893.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 912.30 | 898.42 | 895.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 900.00 | 912.16 | 905.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 900.00 | 912.16 | 905.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 900.00 | 912.16 | 905.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 900.00 | 912.16 | 905.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 899.05 | 909.54 | 905.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 897.05 | 909.54 | 905.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 891.45 | 901.60 | 902.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 885.40 | 898.36 | 900.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 890.90 | 886.48 | 891.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 890.90 | 886.48 | 891.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 890.90 | 886.48 | 891.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 890.90 | 886.48 | 891.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 890.00 | 887.18 | 891.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 906.75 | 887.18 | 891.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 908.35 | 891.42 | 893.28 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 907.85 | 894.70 | 894.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 910.25 | 897.81 | 896.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 902.25 | 903.51 | 900.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 902.25 | 903.51 | 900.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 906.00 | 904.01 | 900.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 901.15 | 904.01 | 900.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 896.30 | 905.86 | 903.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 897.20 | 905.86 | 903.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 890.40 | 902.77 | 902.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 889.00 | 902.77 | 902.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 888.40 | 899.90 | 900.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 886.35 | 895.36 | 898.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 891.30 | 886.19 | 890.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 891.30 | 886.19 | 890.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 891.30 | 886.19 | 890.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:15:00 | 900.45 | 886.19 | 890.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 911.70 | 891.29 | 892.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 911.70 | 891.29 | 892.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 921.35 | 897.30 | 894.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 927.15 | 906.66 | 899.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 921.60 | 923.63 | 915.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:45:00 | 922.70 | 923.63 | 915.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 908.75 | 920.36 | 917.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 908.20 | 920.36 | 917.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 911.85 | 918.66 | 917.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 907.35 | 918.66 | 917.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 925.95 | 918.29 | 917.20 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 909.05 | 916.44 | 916.46 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 10:15:00 | 925.00 | 917.23 | 916.71 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 908.30 | 917.14 | 917.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 892.20 | 910.69 | 914.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 907.25 | 904.10 | 909.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:45:00 | 907.55 | 904.10 | 909.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 918.00 | 905.63 | 908.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 918.00 | 905.63 | 908.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 912.00 | 906.90 | 908.90 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 918.85 | 910.08 | 910.05 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 907.90 | 910.48 | 910.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 13:15:00 | 905.90 | 909.57 | 910.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 909.80 | 909.61 | 910.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 909.80 | 909.61 | 910.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 909.80 | 909.61 | 910.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 911.25 | 909.61 | 910.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 908.00 | 909.29 | 910.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 903.00 | 909.29 | 910.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 893.80 | 906.19 | 908.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 884.90 | 899.52 | 903.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 885.35 | 887.58 | 894.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 886.25 | 874.21 | 872.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 13:15:00 | 886.25 | 874.21 | 872.96 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 863.20 | 872.01 | 872.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 857.85 | 866.56 | 869.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 870.00 | 864.92 | 867.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 870.00 | 864.92 | 867.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 870.00 | 864.92 | 867.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:45:00 | 872.00 | 864.92 | 867.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 871.00 | 866.13 | 867.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 841.60 | 866.13 | 867.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:00:00 | 863.50 | 862.93 | 865.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 12:15:00 | 820.32 | 831.43 | 841.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 799.52 | 825.32 | 835.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 818.85 | 806.67 | 817.24 | SL hit (close>ema200) qty=0.50 sl=806.67 alert=retest2 |

### Cycle 207 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 813.00 | 798.90 | 797.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 820.30 | 807.42 | 802.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 808.85 | 815.50 | 810.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 808.85 | 815.50 | 810.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 808.85 | 815.50 | 810.29 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 794.90 | 806.46 | 807.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 12:15:00 | 793.00 | 800.49 | 803.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 770.50 | 770.35 | 781.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 770.50 | 770.35 | 781.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 770.50 | 770.35 | 781.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 788.90 | 770.35 | 781.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 789.65 | 776.30 | 781.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 789.65 | 776.30 | 781.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 796.00 | 780.24 | 782.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 796.00 | 780.24 | 782.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 793.00 | 784.30 | 784.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 810.75 | 789.59 | 786.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 794.00 | 802.04 | 796.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 794.00 | 802.04 | 796.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 794.00 | 802.04 | 796.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 794.00 | 802.04 | 796.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 789.10 | 799.45 | 795.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 788.60 | 799.45 | 795.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 787.65 | 793.50 | 793.63 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 795.65 | 793.93 | 793.81 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 789.80 | 793.10 | 793.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 785.45 | 791.57 | 792.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 790.10 | 788.52 | 790.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 13:00:00 | 790.10 | 788.52 | 790.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 784.35 | 787.69 | 790.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 14:30:00 | 779.80 | 785.12 | 788.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 808.70 | 788.38 | 789.56 | SL hit (close>static) qty=1.00 sl=790.45 alert=retest2 |

### Cycle 213 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 812.25 | 793.15 | 791.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 816.45 | 797.81 | 793.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 784.70 | 803.16 | 799.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 784.70 | 803.16 | 799.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 784.70 | 803.16 | 799.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 784.70 | 803.16 | 799.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 788.50 | 800.22 | 798.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 792.05 | 797.89 | 797.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 871.25 | 858.89 | 852.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 850.80 | 855.11 | 855.53 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 860.50 | 855.85 | 855.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 12:15:00 | 870.25 | 859.33 | 857.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 854.80 | 860.97 | 859.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 854.80 | 860.97 | 859.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 854.80 | 860.97 | 859.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 851.25 | 860.97 | 859.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 856.70 | 860.12 | 858.88 | EMA400 retest candle locked (from upside) |

### Cycle 216 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 851.55 | 857.53 | 857.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 850.00 | 855.15 | 856.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 11:15:00 | 855.30 | 854.58 | 856.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 11:15:00 | 855.30 | 854.58 | 856.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 855.30 | 854.58 | 856.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:45:00 | 854.90 | 854.58 | 856.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 855.70 | 854.81 | 855.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 855.80 | 854.81 | 855.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 853.20 | 854.49 | 855.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 852.75 | 854.49 | 855.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 856.25 | 853.72 | 854.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:15:00 | 864.45 | 853.72 | 854.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 863.00 | 855.57 | 855.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:30:00 | 867.60 | 855.57 | 855.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 861.50 | 856.76 | 856.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 871.70 | 861.14 | 858.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 872.15 | 876.65 | 869.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 15:00:00 | 872.15 | 876.65 | 869.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 869.30 | 875.18 | 869.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 863.75 | 875.18 | 869.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 858.30 | 871.80 | 868.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 853.35 | 871.80 | 868.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 848.60 | 867.16 | 866.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 848.60 | 867.16 | 866.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 848.45 | 863.42 | 864.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 845.25 | 859.79 | 863.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 855.95 | 851.78 | 856.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:00:00 | 855.95 | 851.78 | 856.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 858.55 | 853.13 | 857.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 858.55 | 853.13 | 857.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 862.00 | 854.90 | 857.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 861.15 | 854.90 | 857.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 859.15 | 855.52 | 857.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:45:00 | 857.85 | 855.52 | 857.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 855.15 | 855.45 | 856.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 847.80 | 852.51 | 855.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 849.15 | 849.37 | 852.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:15:00 | 849.70 | 849.82 | 852.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:30:00 | 849.65 | 850.26 | 852.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 845.40 | 844.74 | 848.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 845.40 | 844.74 | 848.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 854.25 | 846.64 | 849.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 854.25 | 846.64 | 849.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 855.95 | 848.50 | 849.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:30:00 | 853.55 | 848.50 | 849.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 857.25 | 851.29 | 850.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 857.25 | 851.29 | 850.90 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 14:15:00 | 844.15 | 849.86 | 850.28 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 855.30 | 851.21 | 850.79 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 843.40 | 850.48 | 850.73 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 859.75 | 852.33 | 851.55 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 849.00 | 851.91 | 851.94 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 858.00 | 852.89 | 852.25 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 14:15:00 | 847.40 | 851.41 | 851.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 839.90 | 848.73 | 850.51 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 10:30:00 | 504.30 | 2023-05-24 14:15:00 | 497.65 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2023-05-16 12:45:00 | 503.05 | 2023-05-24 14:15:00 | 497.65 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2023-05-17 10:15:00 | 502.10 | 2023-05-24 14:15:00 | 497.65 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2023-06-07 13:30:00 | 476.40 | 2023-06-09 14:15:00 | 481.05 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-06-08 10:00:00 | 475.90 | 2023-06-09 14:15:00 | 481.05 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-06-08 11:15:00 | 476.30 | 2023-06-09 14:15:00 | 481.05 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-06-09 12:00:00 | 476.50 | 2023-06-09 14:15:00 | 481.05 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-06-14 15:15:00 | 469.00 | 2023-06-19 11:15:00 | 472.50 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-06-19 11:00:00 | 468.70 | 2023-06-19 11:15:00 | 472.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest1 | 2023-06-22 09:15:00 | 478.95 | 2023-06-26 09:15:00 | 473.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest1 | 2023-06-22 10:15:00 | 482.00 | 2023-06-26 09:15:00 | 473.70 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2023-07-03 11:15:00 | 460.60 | 2023-07-05 10:15:00 | 466.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2023-07-03 14:15:00 | 459.95 | 2023-07-05 10:15:00 | 466.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-07-07 09:15:00 | 473.10 | 2023-07-11 10:15:00 | 465.50 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2023-07-07 12:45:00 | 466.95 | 2023-07-11 10:15:00 | 465.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-07-07 14:15:00 | 467.00 | 2023-07-11 10:15:00 | 465.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2023-07-07 15:15:00 | 467.95 | 2023-07-11 10:15:00 | 465.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-07-10 09:30:00 | 470.45 | 2023-07-11 10:15:00 | 465.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-08-24 09:15:00 | 469.40 | 2023-08-29 14:15:00 | 474.00 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2023-09-27 14:45:00 | 522.90 | 2023-09-28 09:15:00 | 536.90 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2023-09-28 14:30:00 | 522.00 | 2023-10-06 09:15:00 | 523.30 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2023-09-29 15:00:00 | 520.40 | 2023-10-06 09:15:00 | 523.30 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-10-03 09:45:00 | 522.15 | 2023-10-06 09:15:00 | 523.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2023-10-12 11:45:00 | 517.75 | 2023-10-19 09:15:00 | 491.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-12 11:45:00 | 517.75 | 2023-10-20 09:15:00 | 501.55 | STOP_HIT | 0.50 | 3.13% |
| BUY | retest2 | 2023-11-08 09:15:00 | 479.80 | 2023-11-15 14:15:00 | 497.50 | STOP_HIT | 1.00 | 3.69% |
| BUY | retest2 | 2023-11-08 11:30:00 | 480.50 | 2023-11-15 14:15:00 | 497.50 | STOP_HIT | 1.00 | 3.54% |
| BUY | retest2 | 2023-11-08 13:00:00 | 480.25 | 2023-11-15 14:15:00 | 497.50 | STOP_HIT | 1.00 | 3.59% |
| SELL | retest2 | 2023-11-16 15:00:00 | 495.60 | 2023-11-23 09:15:00 | 511.50 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2023-11-17 14:00:00 | 494.45 | 2023-11-23 09:15:00 | 511.50 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2023-11-20 10:00:00 | 495.80 | 2023-11-23 09:15:00 | 511.50 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2023-11-20 12:30:00 | 495.75 | 2023-11-23 09:15:00 | 511.50 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2023-11-22 11:15:00 | 486.85 | 2023-11-23 09:15:00 | 511.50 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2023-11-22 14:00:00 | 484.95 | 2023-11-23 09:15:00 | 511.50 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2023-11-22 15:00:00 | 487.15 | 2023-11-23 09:15:00 | 511.50 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2023-11-30 09:15:00 | 544.80 | 2023-12-07 12:15:00 | 539.95 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-12-01 13:30:00 | 531.60 | 2023-12-07 12:15:00 | 539.95 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2023-12-07 09:45:00 | 532.55 | 2023-12-07 12:15:00 | 539.95 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2023-12-13 09:30:00 | 532.55 | 2023-12-14 11:15:00 | 543.80 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2023-12-13 11:30:00 | 531.40 | 2023-12-14 11:15:00 | 543.80 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2023-12-13 13:30:00 | 533.00 | 2023-12-14 11:15:00 | 543.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2023-12-28 13:15:00 | 569.00 | 2023-12-29 13:15:00 | 558.40 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2023-12-28 13:45:00 | 568.50 | 2023-12-29 13:15:00 | 558.40 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2023-12-28 14:45:00 | 570.10 | 2023-12-29 13:15:00 | 558.40 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2023-12-29 09:15:00 | 569.95 | 2023-12-29 13:15:00 | 558.40 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2023-12-29 11:15:00 | 568.85 | 2023-12-29 13:15:00 | 558.40 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-01-09 09:45:00 | 577.00 | 2024-01-15 11:15:00 | 579.70 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2024-01-09 10:15:00 | 578.65 | 2024-01-15 14:15:00 | 581.15 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2024-01-10 09:30:00 | 576.20 | 2024-01-15 14:15:00 | 581.15 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-01-10 10:45:00 | 579.05 | 2024-01-15 14:15:00 | 581.15 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-01-12 14:30:00 | 586.50 | 2024-01-15 14:15:00 | 581.15 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-01-19 15:00:00 | 567.75 | 2024-01-23 10:15:00 | 578.80 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-01-31 12:30:00 | 625.15 | 2024-02-06 14:15:00 | 635.25 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2024-01-31 13:15:00 | 628.25 | 2024-02-06 14:15:00 | 635.25 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2024-01-31 14:15:00 | 626.30 | 2024-02-06 14:15:00 | 635.25 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2024-02-12 12:45:00 | 587.05 | 2024-02-15 09:15:00 | 597.95 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-02-12 14:00:00 | 589.60 | 2024-02-15 09:15:00 | 597.95 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-02-14 11:30:00 | 590.75 | 2024-02-15 09:15:00 | 597.95 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-02-14 13:45:00 | 590.65 | 2024-02-15 09:15:00 | 597.95 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-02-20 09:15:00 | 648.10 | 2024-02-26 13:15:00 | 626.80 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-03-04 10:15:00 | 622.55 | 2024-03-12 10:15:00 | 591.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 11:00:00 | 622.60 | 2024-03-12 10:15:00 | 591.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 10:15:00 | 622.55 | 2024-03-13 09:15:00 | 560.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-04 11:00:00 | 622.60 | 2024-03-13 09:15:00 | 560.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-28 11:15:00 | 551.75 | 2024-04-01 09:15:00 | 565.75 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-03-28 12:45:00 | 551.65 | 2024-04-01 09:15:00 | 565.75 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-04-08 12:45:00 | 590.00 | 2024-04-15 14:15:00 | 609.75 | STOP_HIT | 1.00 | 3.35% |
| BUY | retest2 | 2024-04-09 09:15:00 | 591.00 | 2024-04-15 14:15:00 | 609.75 | STOP_HIT | 1.00 | 3.17% |
| BUY | retest2 | 2024-04-24 09:15:00 | 628.30 | 2024-04-29 14:15:00 | 622.45 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-05-07 11:15:00 | 607.20 | 2024-05-08 13:15:00 | 626.40 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-05-07 12:30:00 | 607.35 | 2024-05-08 13:15:00 | 626.40 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-05-07 13:45:00 | 607.65 | 2024-05-08 13:15:00 | 626.40 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2024-05-22 09:15:00 | 663.90 | 2024-05-24 12:15:00 | 637.60 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2024-05-24 10:15:00 | 641.70 | 2024-05-24 12:15:00 | 637.60 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-05-24 11:15:00 | 641.20 | 2024-05-24 12:15:00 | 637.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-05-24 11:45:00 | 644.10 | 2024-05-24 12:15:00 | 637.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-06-03 09:15:00 | 673.45 | 2024-06-04 10:15:00 | 651.70 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-06-04 10:00:00 | 671.70 | 2024-06-04 10:15:00 | 651.70 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-06-14 14:30:00 | 725.85 | 2024-06-19 13:15:00 | 798.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-18 10:15:00 | 725.95 | 2024-06-19 13:15:00 | 798.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-18 13:15:00 | 795.00 | 2024-07-19 09:15:00 | 770.00 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-08-01 09:15:00 | 823.70 | 2024-08-01 09:15:00 | 816.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-08-06 14:00:00 | 761.30 | 2024-08-08 11:15:00 | 773.55 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-02 10:45:00 | 833.45 | 2024-09-03 15:15:00 | 818.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-09-06 13:00:00 | 840.00 | 2024-09-11 10:15:00 | 832.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-09-06 13:30:00 | 839.30 | 2024-09-11 10:15:00 | 832.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-09 09:45:00 | 840.95 | 2024-09-11 10:15:00 | 832.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-09-09 10:15:00 | 840.70 | 2024-09-11 10:15:00 | 832.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-09-10 09:15:00 | 854.55 | 2024-09-11 10:15:00 | 832.50 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-09-10 14:00:00 | 847.55 | 2024-09-11 10:15:00 | 832.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-09-25 09:15:00 | 828.85 | 2024-09-26 11:15:00 | 824.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-09-25 10:30:00 | 828.20 | 2024-09-26 11:15:00 | 824.40 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-09-25 12:00:00 | 828.55 | 2024-09-26 11:15:00 | 824.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-09-25 14:00:00 | 830.95 | 2024-09-26 11:15:00 | 824.40 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-11 14:15:00 | 810.45 | 2024-10-15 10:15:00 | 825.90 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-10-14 09:15:00 | 808.85 | 2024-10-15 10:15:00 | 825.90 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-10-24 15:15:00 | 770.50 | 2024-10-30 09:15:00 | 785.85 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-11-11 09:15:00 | 849.20 | 2024-11-11 11:15:00 | 824.20 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-11-13 09:15:00 | 795.75 | 2024-11-18 10:15:00 | 755.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 09:15:00 | 795.75 | 2024-11-19 09:15:00 | 809.00 | STOP_HIT | 0.50 | -1.67% |
| BUY | retest2 | 2024-11-28 09:15:00 | 862.10 | 2024-12-12 15:15:00 | 940.94 | TARGET_HIT | 1.00 | 9.15% |
| BUY | retest2 | 2024-11-29 10:45:00 | 855.40 | 2024-12-12 15:15:00 | 942.65 | TARGET_HIT | 1.00 | 10.20% |
| BUY | retest2 | 2024-11-29 13:15:00 | 856.95 | 2024-12-13 09:15:00 | 948.31 | TARGET_HIT | 1.00 | 10.66% |
| BUY | retest2 | 2024-12-19 13:00:00 | 958.00 | 2024-12-23 09:15:00 | 922.60 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2024-12-20 15:00:00 | 958.65 | 2024-12-23 09:15:00 | 922.60 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest1 | 2025-01-21 09:15:00 | 860.25 | 2025-01-21 10:15:00 | 829.50 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-01-31 12:30:00 | 822.45 | 2025-02-01 12:15:00 | 811.15 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-02-07 15:15:00 | 855.05 | 2025-02-10 09:15:00 | 823.25 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-02-13 13:00:00 | 791.45 | 2025-02-14 10:15:00 | 751.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 791.45 | 2025-02-17 10:15:00 | 712.31 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-17 09:15:00 | 697.80 | 2025-03-21 09:15:00 | 767.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 11:30:00 | 848.95 | 2025-04-25 10:15:00 | 835.45 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-04-23 12:00:00 | 848.70 | 2025-04-25 10:15:00 | 835.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-04-23 12:30:00 | 848.65 | 2025-04-25 10:15:00 | 835.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-04-24 10:00:00 | 855.35 | 2025-04-25 10:15:00 | 835.45 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-05-07 11:15:00 | 846.65 | 2025-05-09 09:15:00 | 826.50 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-05-07 15:00:00 | 847.30 | 2025-05-09 09:15:00 | 826.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-05-08 14:15:00 | 845.85 | 2025-05-09 09:15:00 | 826.50 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-05-21 14:00:00 | 981.75 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-05-21 15:00:00 | 984.95 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-05-22 10:30:00 | 981.70 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-05-22 12:00:00 | 979.80 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-05-22 14:30:00 | 982.50 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-05-23 09:15:00 | 990.65 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-27 10:45:00 | 985.80 | 2025-05-27 11:15:00 | 984.45 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-06-09 15:00:00 | 969.70 | 2025-06-10 09:15:00 | 977.30 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-17 11:30:00 | 966.15 | 2025-06-23 10:15:00 | 974.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-06-19 10:15:00 | 962.25 | 2025-06-23 10:15:00 | 974.60 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-06-24 11:15:00 | 975.55 | 2025-06-26 09:15:00 | 1073.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 15:15:00 | 974.00 | 2025-06-26 09:15:00 | 1071.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-10 09:45:00 | 1077.00 | 2025-07-10 11:15:00 | 1090.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-07-16 10:15:00 | 1166.30 | 2025-07-18 09:15:00 | 1145.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-07-16 11:15:00 | 1163.20 | 2025-07-18 09:15:00 | 1145.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-23 13:00:00 | 1178.60 | 2025-07-24 13:15:00 | 1162.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-08-07 13:15:00 | 1143.40 | 2025-08-08 15:15:00 | 1086.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 13:15:00 | 1143.40 | 2025-08-11 09:15:00 | 1137.00 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2025-09-01 09:15:00 | 1115.00 | 2025-09-01 13:15:00 | 1135.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-01 10:45:00 | 1123.20 | 2025-09-01 13:15:00 | 1135.40 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-02 12:30:00 | 1120.60 | 2025-09-04 12:15:00 | 1130.10 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-03 10:45:00 | 1124.60 | 2025-09-04 12:15:00 | 1130.10 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-09-04 09:15:00 | 1113.00 | 2025-09-04 14:15:00 | 1128.10 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-09-04 10:15:00 | 1116.90 | 2025-09-04 14:15:00 | 1128.10 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-10 14:15:00 | 1087.20 | 2025-09-16 11:15:00 | 1089.00 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-09-10 15:15:00 | 1085.10 | 2025-09-16 11:15:00 | 1089.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-11 09:45:00 | 1087.40 | 2025-09-16 11:15:00 | 1089.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-09-11 12:00:00 | 1085.40 | 2025-09-16 11:15:00 | 1089.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-09-19 10:00:00 | 1101.10 | 2025-09-19 13:15:00 | 1083.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-09-19 12:00:00 | 1099.00 | 2025-09-19 13:15:00 | 1083.90 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-09-25 13:30:00 | 1065.00 | 2025-09-29 09:15:00 | 1011.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 14:30:00 | 1064.90 | 2025-09-29 09:15:00 | 1011.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 13:30:00 | 1065.00 | 2025-09-30 11:15:00 | 1024.90 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-09-25 14:30:00 | 1064.90 | 2025-09-30 11:15:00 | 1024.90 | STOP_HIT | 0.50 | 3.76% |
| BUY | retest2 | 2025-10-07 09:45:00 | 1051.00 | 2025-10-09 11:15:00 | 1035.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1049.50 | 2025-10-09 11:15:00 | 1035.80 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-08 12:00:00 | 1053.80 | 2025-10-09 11:15:00 | 1035.80 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-10-09 10:00:00 | 1050.20 | 2025-10-09 11:15:00 | 1035.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-10-14 11:45:00 | 1032.00 | 2025-10-14 14:15:00 | 1041.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-16 12:15:00 | 1032.70 | 2025-10-21 13:15:00 | 1037.80 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-10-16 13:30:00 | 1031.50 | 2025-10-21 13:15:00 | 1037.80 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-10-16 14:30:00 | 1032.90 | 2025-10-21 13:15:00 | 1037.80 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-10-16 15:00:00 | 1032.30 | 2025-10-21 13:15:00 | 1037.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-24 15:15:00 | 1043.00 | 2025-10-31 14:15:00 | 1072.90 | STOP_HIT | 1.00 | 2.87% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1015.70 | 2025-11-11 12:15:00 | 1050.10 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-11-11 14:45:00 | 1016.20 | 2025-11-12 09:15:00 | 1071.80 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest1 | 2025-11-25 09:15:00 | 1013.50 | 2025-11-25 09:15:00 | 1025.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-03 12:00:00 | 1031.30 | 2025-12-05 13:15:00 | 1030.60 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-04 09:45:00 | 1034.60 | 2025-12-05 13:15:00 | 1030.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-12-05 09:45:00 | 1035.00 | 2025-12-05 13:15:00 | 1030.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-12-19 09:15:00 | 1022.10 | 2025-12-22 09:15:00 | 1041.60 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-12-19 10:00:00 | 1022.30 | 2025-12-22 09:15:00 | 1041.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-05 10:30:00 | 1005.40 | 2026-01-08 15:15:00 | 955.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 1003.40 | 2026-01-08 15:15:00 | 953.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:30:00 | 1005.50 | 2026-01-08 15:15:00 | 955.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:45:00 | 1005.40 | 2026-01-08 15:15:00 | 955.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 12:30:00 | 995.60 | 2026-01-09 09:15:00 | 947.34 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-07 14:30:00 | 997.20 | 2026-01-09 13:15:00 | 945.82 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2026-01-08 09:15:00 | 993.40 | 2026-01-09 13:15:00 | 943.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:30:00 | 1005.40 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2026-01-05 11:15:00 | 1003.40 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 6.10% |
| SELL | retest2 | 2026-01-05 13:30:00 | 1005.50 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 6.30% |
| SELL | retest2 | 2026-01-05 14:45:00 | 1005.40 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2026-01-07 12:30:00 | 995.60 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2026-01-07 14:30:00 | 997.20 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2026-01-08 09:15:00 | 993.40 | 2026-01-12 13:15:00 | 942.20 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2026-02-20 09:15:00 | 884.90 | 2026-02-27 13:15:00 | 886.25 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2026-02-23 09:30:00 | 885.35 | 2026-02-27 13:15:00 | 886.25 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2026-03-04 09:15:00 | 841.60 | 2026-03-06 12:15:00 | 820.32 | PARTIAL | 0.50 | 2.53% |
| SELL | retest2 | 2026-03-04 11:00:00 | 863.50 | 2026-03-09 09:15:00 | 799.52 | PARTIAL | 0.50 | 7.41% |
| SELL | retest2 | 2026-03-04 09:15:00 | 841.60 | 2026-03-10 10:15:00 | 818.85 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2026-03-04 11:00:00 | 863.50 | 2026-03-10 10:15:00 | 818.85 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-30 14:30:00 | 779.80 | 2026-04-01 09:15:00 | 808.70 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-04-02 12:15:00 | 792.05 | 2026-04-15 09:15:00 | 871.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 13:30:00 | 847.80 | 2026-04-30 13:15:00 | 857.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-04-29 09:45:00 | 849.15 | 2026-04-30 13:15:00 | 857.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-04-29 11:15:00 | 849.70 | 2026-04-30 13:15:00 | 857.25 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-04-29 12:30:00 | 849.65 | 2026-04-30 13:15:00 | 857.25 | STOP_HIT | 1.00 | -0.89% |
