# Aarti Industries Ltd. (AARTIIND)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 486.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 51 |
| ALERT2 | 49 |
| ALERT2_SKIP | 24 |
| ALERT3 | 135 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 55 |
| PARTIAL | 1 |
| TARGET_HIT | 11 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 44
- **Target hits / Stop hits / Partials:** 11 / 46 / 1
- **Avg / median % per leg:** 0.97% / -0.82%
- **Sum % (uncompounded):** 56.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 12 | 44.4% | 11 | 16 | 0 | 3.38% | 91.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.89% | -1.8% |
| BUY @ 3rd Alert (retest2) | 25 | 12 | 48.0% | 11 | 14 | 0 | 3.73% | 93.2% |
| SELL (all) | 31 | 2 | 6.5% | 0 | 30 | 1 | -1.14% | -35.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 2 | 6.5% | 0 | 30 | 1 | -1.14% | -35.2% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.89% | -1.8% |
| retest2 (combined) | 56 | 14 | 25.0% | 11 | 44 | 1 | 1.03% | 57.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 473.20 | 476.77 | 477.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 466.50 | 474.72 | 476.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 471.00 | 470.45 | 472.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:00:00 | 471.00 | 470.45 | 472.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 470.00 | 470.36 | 472.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 466.40 | 470.12 | 472.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:00:00 | 467.60 | 469.39 | 471.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 468.65 | 469.25 | 470.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 468.45 | 469.05 | 470.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 469.40 | 469.12 | 469.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:30:00 | 471.95 | 469.12 | 469.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 468.00 | 468.70 | 469.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 468.05 | 468.70 | 469.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 464.95 | 467.96 | 469.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-27 12:15:00 | 475.05 | 470.50 | 470.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 475.05 | 470.50 | 470.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 477.60 | 472.48 | 471.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 477.35 | 478.15 | 475.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:15:00 | 480.10 | 478.15 | 475.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 10:45:00 | 480.95 | 478.84 | 476.42 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 476.25 | 478.20 | 476.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 476.25 | 478.20 | 476.73 | SL hit (close<ema400) qty=1.00 sl=476.73 alert=retest1 |

### Cycle 3 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 471.00 | 475.93 | 476.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 476.40 | 472.90 | 472.68 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 470.90 | 472.61 | 472.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 470.00 | 472.09 | 472.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 473.10 | 471.95 | 472.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 473.10 | 471.95 | 472.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 473.10 | 471.95 | 472.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 473.60 | 471.95 | 472.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 474.50 | 472.46 | 472.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 474.20 | 472.46 | 472.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 477.20 | 473.41 | 472.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 13:15:00 | 479.85 | 474.70 | 473.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 15:15:00 | 480.75 | 481.16 | 478.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 09:15:00 | 478.00 | 481.16 | 478.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 479.60 | 480.85 | 478.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 479.55 | 480.85 | 478.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 479.45 | 480.57 | 478.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 479.80 | 480.57 | 478.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 477.00 | 479.85 | 478.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 477.00 | 479.85 | 478.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 477.95 | 479.47 | 478.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 480.80 | 478.30 | 478.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 475.20 | 482.75 | 482.51 | SL hit (close<static) qty=1.00 sl=476.10 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 478.55 | 481.91 | 482.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 469.60 | 476.95 | 479.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 459.05 | 458.78 | 464.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 459.05 | 458.78 | 464.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 466.75 | 461.71 | 464.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 467.40 | 461.71 | 464.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 465.50 | 462.46 | 464.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 462.05 | 462.46 | 464.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 438.95 | 447.21 | 452.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 443.60 | 441.97 | 447.69 | SL hit (close>ema200) qty=0.50 sl=441.97 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 459.95 | 447.02 | 446.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 468.00 | 456.60 | 452.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 14:15:00 | 479.30 | 479.60 | 475.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:45:00 | 479.00 | 479.60 | 475.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 477.95 | 479.11 | 476.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 477.95 | 479.11 | 476.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 479.05 | 479.10 | 476.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 477.55 | 479.10 | 476.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 484.90 | 480.06 | 477.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:15:00 | 485.35 | 483.42 | 481.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 475.00 | 479.95 | 480.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 475.00 | 479.95 | 480.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 472.50 | 476.60 | 478.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 446.95 | 445.04 | 449.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 446.95 | 445.04 | 449.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 447.35 | 444.79 | 447.28 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 451.85 | 448.88 | 448.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 455.10 | 450.66 | 449.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 454.70 | 455.32 | 453.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 454.70 | 455.32 | 453.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 448.20 | 454.13 | 453.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 448.20 | 454.13 | 453.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 446.25 | 452.55 | 452.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 444.10 | 447.81 | 450.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 426.35 | 424.41 | 430.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 426.35 | 424.41 | 430.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 430.75 | 426.19 | 430.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:45:00 | 430.15 | 426.19 | 430.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 426.75 | 426.30 | 429.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 430.00 | 426.30 | 429.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 432.40 | 427.90 | 430.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 432.40 | 427.90 | 430.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 432.60 | 428.84 | 430.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 432.60 | 428.84 | 430.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 427.50 | 429.68 | 430.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:30:00 | 428.85 | 429.68 | 430.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 430.95 | 429.24 | 429.90 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 431.45 | 430.31 | 430.27 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 428.90 | 430.01 | 430.15 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 439.10 | 431.60 | 430.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 11:15:00 | 442.05 | 435.14 | 432.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 445.30 | 445.57 | 441.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 445.30 | 445.57 | 441.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 428.20 | 442.00 | 440.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 428.20 | 442.00 | 440.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 421.15 | 437.83 | 438.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 417.20 | 426.56 | 432.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 13:15:00 | 385.30 | 384.64 | 391.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:45:00 | 385.55 | 384.64 | 391.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 376.25 | 373.94 | 375.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:45:00 | 376.65 | 373.94 | 375.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 372.05 | 373.56 | 375.56 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 384.90 | 377.20 | 376.63 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 376.35 | 378.47 | 378.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 375.45 | 377.87 | 378.27 | Break + close below crossover candle low |

### Cycle 18 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 381.60 | 378.61 | 378.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 383.30 | 380.50 | 379.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 391.35 | 391.66 | 388.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 387.10 | 390.23 | 388.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 387.10 | 390.23 | 388.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 387.85 | 390.23 | 388.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 384.75 | 389.13 | 388.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 384.60 | 389.13 | 388.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 381.90 | 386.75 | 387.26 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 389.10 | 386.43 | 386.21 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 382.50 | 385.73 | 386.09 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 387.00 | 386.25 | 386.25 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 384.10 | 385.82 | 386.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 381.05 | 384.87 | 385.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 380.10 | 379.26 | 381.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 379.25 | 379.26 | 381.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 380.55 | 379.52 | 381.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 379.00 | 379.52 | 381.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:30:00 | 379.00 | 379.22 | 381.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 383.20 | 378.58 | 379.12 | SL hit (close>static) qty=1.00 sl=381.90 alert=retest2 |

### Cycle 24 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 383.40 | 379.55 | 379.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 385.45 | 380.73 | 380.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 379.90 | 381.12 | 380.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 379.90 | 381.12 | 380.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 379.90 | 381.12 | 380.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 379.90 | 381.12 | 380.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 383.60 | 381.61 | 380.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 386.20 | 381.71 | 380.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 385.00 | 383.52 | 382.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 380.10 | 382.00 | 382.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 380.10 | 382.00 | 382.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 377.85 | 380.89 | 381.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 388.75 | 380.61 | 380.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 388.75 | 380.61 | 380.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 388.75 | 380.61 | 380.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 388.75 | 380.61 | 380.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 388.45 | 382.18 | 381.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 393.00 | 388.88 | 386.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 389.40 | 390.24 | 388.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 389.40 | 390.24 | 388.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 388.50 | 389.88 | 388.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 388.50 | 389.88 | 388.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 390.00 | 389.90 | 388.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 392.55 | 389.90 | 388.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 389.30 | 391.22 | 391.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 389.30 | 391.22 | 391.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 387.55 | 389.66 | 390.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 390.00 | 389.59 | 390.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 390.00 | 389.59 | 390.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 390.00 | 389.59 | 390.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 390.00 | 389.59 | 390.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 391.20 | 389.91 | 390.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 391.20 | 389.91 | 390.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 391.45 | 390.22 | 390.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:45:00 | 390.55 | 390.22 | 390.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 390.10 | 390.20 | 390.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 389.30 | 390.20 | 390.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 395.60 | 391.32 | 390.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 395.60 | 391.32 | 390.81 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 388.90 | 390.35 | 390.46 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 392.10 | 390.67 | 390.58 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 389.70 | 390.54 | 390.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 388.60 | 390.15 | 390.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 391.00 | 390.21 | 390.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 391.00 | 390.21 | 390.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 391.00 | 390.21 | 390.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 391.00 | 390.21 | 390.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 391.30 | 390.43 | 390.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 391.50 | 390.43 | 390.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 394.25 | 391.20 | 390.78 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 389.75 | 391.86 | 391.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 386.15 | 390.12 | 391.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 390.05 | 389.87 | 390.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 390.05 | 389.87 | 390.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 371.55 | 374.06 | 376.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 371.55 | 374.06 | 376.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 376.00 | 374.11 | 375.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 376.80 | 374.11 | 375.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 374.65 | 374.22 | 375.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 376.60 | 374.22 | 375.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 374.10 | 374.20 | 375.15 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 378.25 | 375.82 | 375.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 380.00 | 377.17 | 376.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 379.25 | 379.65 | 378.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 379.25 | 379.65 | 378.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 379.25 | 379.65 | 378.42 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 377.80 | 378.07 | 378.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 376.85 | 377.69 | 377.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 13:15:00 | 377.50 | 377.37 | 377.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 13:15:00 | 377.50 | 377.37 | 377.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 377.50 | 377.37 | 377.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 380.15 | 377.37 | 377.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 376.40 | 377.18 | 377.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 376.40 | 377.18 | 377.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 376.50 | 376.08 | 376.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 376.90 | 376.08 | 376.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 376.50 | 376.16 | 376.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 376.75 | 376.16 | 376.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 376.55 | 376.24 | 376.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 376.50 | 376.24 | 376.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 376.50 | 376.29 | 376.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:30:00 | 376.60 | 376.29 | 376.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 380.10 | 377.05 | 377.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 380.15 | 377.05 | 377.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 381.20 | 377.88 | 377.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 382.10 | 379.63 | 378.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 14:15:00 | 380.90 | 381.19 | 380.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 15:00:00 | 380.90 | 381.19 | 380.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 380.80 | 381.11 | 380.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 380.05 | 380.58 | 379.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 377.85 | 380.03 | 379.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:15:00 | 376.85 | 380.03 | 379.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 376.90 | 379.41 | 379.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 374.05 | 377.35 | 378.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 375.60 | 374.61 | 376.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 375.60 | 374.61 | 376.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 375.60 | 374.61 | 376.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 374.35 | 374.61 | 376.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 376.70 | 375.03 | 376.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 377.55 | 375.03 | 376.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 376.60 | 375.34 | 376.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 376.35 | 375.34 | 376.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 376.50 | 375.62 | 376.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 375.05 | 375.96 | 376.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 376.35 | 375.96 | 376.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 376.75 | 376.12 | 376.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 376.90 | 376.12 | 376.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 375.10 | 374.09 | 375.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 375.10 | 374.09 | 375.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 375.05 | 374.29 | 375.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 374.65 | 374.29 | 375.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 372.15 | 373.86 | 374.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 379.45 | 374.98 | 374.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 379.45 | 374.98 | 374.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 380.80 | 376.74 | 375.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 376.80 | 378.21 | 377.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 10:15:00 | 376.80 | 378.21 | 377.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 376.80 | 378.21 | 377.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 376.80 | 378.21 | 377.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 377.50 | 378.07 | 377.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 378.60 | 378.07 | 377.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:45:00 | 378.80 | 378.21 | 377.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 378.00 | 377.93 | 377.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 381.00 | 378.55 | 377.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 380.55 | 379.80 | 378.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 380.30 | 379.80 | 378.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 377.25 | 379.99 | 379.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 377.30 | 379.99 | 379.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 377.75 | 379.54 | 379.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:15:00 | 377.55 | 379.54 | 379.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 377.55 | 379.15 | 379.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 377.55 | 379.15 | 379.28 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 385.45 | 380.41 | 379.84 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 380.15 | 382.07 | 382.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 379.45 | 381.34 | 381.76 | Break + close below crossover candle low |

### Cycle 42 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 388.20 | 382.64 | 382.27 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 380.45 | 384.66 | 384.87 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 13:15:00 | 388.20 | 385.33 | 385.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 416.10 | 392.88 | 388.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 14:15:00 | 396.25 | 400.46 | 394.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 396.25 | 400.46 | 394.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 396.25 | 400.46 | 394.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 396.25 | 400.46 | 394.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 385.05 | 397.38 | 394.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:45:00 | 397.60 | 396.90 | 394.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 397.40 | 397.32 | 394.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 12:00:00 | 399.00 | 397.32 | 394.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 386.80 | 393.33 | 393.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 386.80 | 393.33 | 393.64 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 394.15 | 392.02 | 391.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 397.45 | 393.66 | 392.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 395.50 | 396.59 | 394.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 395.50 | 396.59 | 394.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 393.30 | 395.93 | 394.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 393.30 | 395.93 | 394.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 393.15 | 395.38 | 394.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 391.70 | 395.38 | 394.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 392.00 | 393.75 | 393.93 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 396.00 | 393.87 | 393.67 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 390.50 | 393.99 | 394.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 389.15 | 390.92 | 392.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 381.35 | 380.51 | 382.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 381.35 | 380.51 | 382.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 380.55 | 380.10 | 381.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:15:00 | 381.15 | 380.10 | 381.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 381.25 | 380.33 | 381.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:30:00 | 380.05 | 380.11 | 381.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:45:00 | 380.15 | 379.11 | 380.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 380.10 | 379.08 | 379.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 380.00 | 380.24 | 380.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 382.20 | 380.63 | 380.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 382.20 | 380.63 | 380.45 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 378.90 | 380.71 | 380.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 375.85 | 379.41 | 380.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 361.25 | 353.77 | 356.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 361.25 | 353.77 | 356.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 361.25 | 353.77 | 356.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 361.95 | 353.77 | 356.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 357.85 | 354.58 | 357.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:45:00 | 353.90 | 354.37 | 356.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:45:00 | 355.85 | 354.93 | 355.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 360.40 | 356.38 | 356.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 360.40 | 356.38 | 356.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 372.40 | 360.13 | 358.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 11:15:00 | 372.75 | 373.06 | 368.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:00:00 | 372.75 | 373.06 | 368.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 366.50 | 371.50 | 369.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 366.50 | 371.50 | 369.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 365.70 | 370.34 | 368.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 365.60 | 370.34 | 368.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 364.90 | 368.07 | 368.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 363.50 | 366.68 | 367.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 367.30 | 366.37 | 367.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 367.30 | 366.37 | 367.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 367.30 | 366.37 | 367.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 367.30 | 366.37 | 367.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 365.55 | 366.20 | 367.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:15:00 | 363.80 | 366.20 | 367.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 365.20 | 365.12 | 366.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 369.35 | 366.73 | 366.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 369.35 | 366.73 | 366.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 373.35 | 368.43 | 367.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 381.10 | 382.28 | 377.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 383.70 | 382.28 | 377.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 379.35 | 380.98 | 378.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 379.35 | 380.98 | 378.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 376.35 | 379.82 | 378.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 381.15 | 379.82 | 378.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 380.50 | 379.95 | 378.78 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 376.10 | 378.18 | 378.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 375.05 | 377.22 | 377.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 377.45 | 376.36 | 377.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 377.45 | 376.36 | 377.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 377.45 | 376.36 | 377.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 377.45 | 376.36 | 377.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 374.00 | 375.89 | 376.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 372.50 | 375.89 | 376.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 372.45 | 375.02 | 376.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 373.55 | 372.49 | 373.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 373.80 | 373.15 | 373.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 375.35 | 372.53 | 372.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:00:00 | 375.35 | 372.53 | 372.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 375.85 | 373.20 | 373.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:45:00 | 375.75 | 373.20 | 373.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-01 15:15:00 | 375.30 | 373.62 | 373.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 375.30 | 373.62 | 373.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 376.80 | 374.42 | 373.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 374.95 | 374.95 | 374.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 374.95 | 374.95 | 374.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 374.20 | 374.80 | 374.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 374.20 | 374.80 | 374.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 375.35 | 374.91 | 374.33 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 372.95 | 373.98 | 374.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 370.40 | 373.04 | 373.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 372.15 | 371.73 | 372.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 372.15 | 371.73 | 372.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 372.15 | 371.73 | 372.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 372.15 | 371.73 | 372.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 373.60 | 372.10 | 372.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 373.60 | 372.10 | 372.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 375.15 | 372.71 | 373.02 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 374.60 | 373.46 | 373.33 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 371.00 | 373.62 | 373.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 368.85 | 372.00 | 372.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 359.70 | 358.24 | 361.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 12:15:00 | 359.70 | 358.24 | 361.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 359.70 | 358.24 | 361.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:45:00 | 360.75 | 358.24 | 361.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 360.90 | 358.77 | 361.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:30:00 | 361.60 | 358.77 | 361.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 361.95 | 359.41 | 361.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 362.30 | 359.41 | 361.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 363.00 | 360.13 | 361.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 364.10 | 360.65 | 361.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 363.35 | 361.19 | 361.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 363.35 | 361.19 | 361.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 364.95 | 362.46 | 362.18 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 358.95 | 361.77 | 361.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 10:15:00 | 357.60 | 360.93 | 361.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 353.50 | 353.36 | 356.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 15:00:00 | 353.50 | 353.36 | 356.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 344.30 | 342.89 | 346.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 345.85 | 342.89 | 346.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 346.10 | 343.60 | 345.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 346.95 | 343.60 | 345.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 348.00 | 344.48 | 345.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 347.90 | 344.48 | 345.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 353.50 | 346.28 | 346.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 353.50 | 346.28 | 346.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 354.30 | 347.89 | 347.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 13:15:00 | 356.60 | 351.26 | 348.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 351.00 | 351.43 | 349.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 351.00 | 351.43 | 349.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 351.00 | 351.43 | 349.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 350.40 | 351.43 | 349.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 352.20 | 351.59 | 349.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:45:00 | 354.10 | 351.69 | 350.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 355.30 | 350.90 | 350.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 12:30:00 | 353.90 | 352.49 | 351.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 14:45:00 | 353.60 | 353.21 | 351.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 350.45 | 353.26 | 352.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 350.45 | 353.26 | 352.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 356.20 | 353.85 | 352.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 357.05 | 354.75 | 353.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 360.90 | 358.29 | 355.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 389.51 | 379.90 | 372.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 450.45 | 461.76 | 461.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 446.75 | 455.36 | 458.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 454.30 | 452.29 | 455.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 454.30 | 452.29 | 455.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 454.10 | 452.65 | 455.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 459.45 | 452.65 | 455.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 453.15 | 452.75 | 455.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 453.15 | 452.75 | 455.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 450.85 | 452.18 | 454.19 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 463.60 | 455.38 | 454.84 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 452.50 | 457.17 | 457.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 450.25 | 455.78 | 456.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 446.70 | 446.31 | 449.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 15:00:00 | 446.70 | 446.31 | 449.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 444.10 | 445.87 | 449.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 442.15 | 445.87 | 449.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 449.95 | 445.37 | 446.79 | SL hit (close>static) qty=1.00 sl=449.55 alert=retest2 |

### Cycle 66 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 453.20 | 448.21 | 447.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 12:15:00 | 455.45 | 449.66 | 448.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 454.00 | 454.64 | 452.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 454.00 | 454.64 | 452.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 454.00 | 454.64 | 452.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 454.00 | 454.64 | 452.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 452.15 | 454.14 | 452.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 450.35 | 454.14 | 452.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 450.60 | 453.43 | 452.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 450.60 | 453.43 | 452.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 446.65 | 451.09 | 451.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 441.80 | 448.49 | 450.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 422.40 | 415.51 | 422.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 422.40 | 415.51 | 422.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 422.40 | 415.51 | 422.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 422.40 | 415.51 | 422.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 419.00 | 416.20 | 422.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 423.10 | 417.23 | 422.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 420.40 | 417.50 | 420.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 420.40 | 417.50 | 420.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 420.00 | 418.00 | 420.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 401.55 | 418.00 | 420.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:00:00 | 418.40 | 412.85 | 414.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 424.50 | 416.98 | 416.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 424.50 | 416.98 | 416.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 427.90 | 419.16 | 417.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 432.00 | 441.52 | 435.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 432.00 | 441.52 | 435.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 432.00 | 441.52 | 435.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 432.00 | 441.52 | 435.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 438.30 | 440.88 | 435.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 439.50 | 440.88 | 435.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 428.75 | 437.28 | 434.63 | SL hit (close<static) qty=1.00 sl=431.50 alert=retest2 |

### Cycle 69 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 420.60 | 432.49 | 432.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 413.90 | 426.84 | 430.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 425.65 | 423.52 | 426.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 425.65 | 423.52 | 426.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 425.65 | 423.52 | 426.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 428.60 | 423.52 | 426.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 424.70 | 423.99 | 426.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 426.65 | 423.99 | 426.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 423.15 | 423.83 | 426.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 419.85 | 423.83 | 426.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 429.95 | 425.10 | 425.37 | SL hit (close>static) qty=1.00 sl=428.90 alert=retest2 |

### Cycle 70 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 430.75 | 426.23 | 425.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 435.20 | 428.02 | 426.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 422.70 | 429.18 | 427.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 422.70 | 429.18 | 427.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 422.70 | 429.18 | 427.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 418.50 | 429.18 | 427.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 418.75 | 427.09 | 427.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 418.30 | 425.33 | 426.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 418.70 | 418.09 | 421.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 12:00:00 | 418.70 | 418.09 | 421.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 420.45 | 418.56 | 421.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 419.30 | 418.56 | 421.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 427.35 | 420.32 | 421.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 427.35 | 420.32 | 421.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 427.05 | 421.67 | 422.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 427.05 | 421.67 | 422.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 427.80 | 422.89 | 422.70 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 410.65 | 420.44 | 421.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 407.45 | 417.84 | 420.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 409.20 | 408.65 | 413.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 409.20 | 408.65 | 413.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 418.00 | 411.03 | 413.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 418.00 | 411.03 | 413.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 417.50 | 412.33 | 413.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 418.50 | 412.33 | 413.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 429.20 | 417.60 | 416.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 434.10 | 420.90 | 417.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 415.60 | 426.24 | 422.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 415.60 | 426.24 | 422.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 415.60 | 426.24 | 422.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 415.60 | 426.24 | 422.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 418.25 | 424.64 | 422.42 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 417.85 | 421.05 | 421.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 413.75 | 419.17 | 420.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 413.95 | 407.35 | 411.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 413.95 | 407.35 | 411.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 413.95 | 407.35 | 411.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 413.95 | 407.35 | 411.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 411.65 | 408.21 | 411.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 410.40 | 411.21 | 412.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 401.90 | 411.15 | 412.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 424.00 | 407.63 | 406.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 424.00 | 407.63 | 406.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 425.85 | 411.28 | 408.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 417.25 | 418.98 | 414.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 417.25 | 418.98 | 414.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 424.00 | 425.97 | 422.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:00:00 | 424.00 | 425.97 | 422.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 424.75 | 425.41 | 422.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 428.90 | 425.41 | 422.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:15:00 | 430.00 | 426.39 | 424.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 428.80 | 425.94 | 424.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:15:00 | 428.85 | 425.35 | 424.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 14:15:00 | 471.79 | 459.09 | 452.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 481.60 | 501.94 | 502.81 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 461.00 | 2025-05-21 10:15:00 | 473.20 | STOP_HIT | 1.00 | 2.65% |
| SELL | retest2 | 2025-05-22 14:15:00 | 466.40 | 2025-05-27 12:15:00 | 475.05 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-05-23 10:00:00 | 467.60 | 2025-05-27 12:15:00 | 475.05 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-05-23 12:15:00 | 468.65 | 2025-05-27 12:15:00 | 475.05 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-26 11:00:00 | 468.45 | 2025-05-27 12:15:00 | 475.05 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest1 | 2025-05-29 09:15:00 | 480.10 | 2025-05-29 13:15:00 | 476.25 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest1 | 2025-05-29 10:45:00 | 480.95 | 2025-05-29 13:15:00 | 476.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-09 09:15:00 | 480.80 | 2025-06-11 13:15:00 | 475.20 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-17 11:15:00 | 462.05 | 2025-06-19 12:15:00 | 438.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 462.05 | 2025-06-20 10:15:00 | 443.60 | STOP_HIT | 0.50 | 3.99% |
| BUY | retest2 | 2025-07-03 10:15:00 | 485.35 | 2025-07-04 11:15:00 | 475.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-08-29 12:15:00 | 379.00 | 2025-09-02 09:15:00 | 383.20 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-29 13:30:00 | 379.00 | 2025-09-02 09:15:00 | 383.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-09-03 09:15:00 | 386.20 | 2025-09-04 15:15:00 | 380.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-03 12:45:00 | 385.00 | 2025-09-04 15:15:00 | 380.10 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-09-11 09:15:00 | 392.55 | 2025-09-12 15:15:00 | 389.30 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-16 13:15:00 | 389.30 | 2025-09-17 09:15:00 | 395.60 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-16 12:15:00 | 376.35 | 2025-10-23 09:15:00 | 379.45 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-10-16 13:15:00 | 376.50 | 2025-10-23 09:15:00 | 379.45 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-10-16 14:30:00 | 375.05 | 2025-10-23 09:15:00 | 379.45 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-10-16 15:00:00 | 376.35 | 2025-10-23 09:15:00 | 379.45 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-24 12:15:00 | 378.60 | 2025-10-28 15:15:00 | 377.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-24 12:45:00 | 378.80 | 2025-10-28 15:15:00 | 377.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-10-24 14:15:00 | 378.00 | 2025-10-28 15:15:00 | 377.55 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-10-24 15:00:00 | 381.00 | 2025-10-28 15:15:00 | 377.55 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-10 10:45:00 | 397.60 | 2025-11-11 09:15:00 | 386.80 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-11-10 11:30:00 | 397.40 | 2025-11-11 09:15:00 | 386.80 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-11-10 12:00:00 | 399.00 | 2025-11-11 09:15:00 | 386.80 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-11-26 12:30:00 | 380.05 | 2025-11-28 11:15:00 | 382.20 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-27 10:45:00 | 380.15 | 2025-11-28 11:15:00 | 382.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-27 13:45:00 | 380.10 | 2025-11-28 11:15:00 | 382.20 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-28 10:45:00 | 380.00 | 2025-11-28 11:15:00 | 382.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-10 11:45:00 | 353.90 | 2025-12-12 09:15:00 | 360.40 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-12-11 14:45:00 | 355.85 | 2025-12-12 09:15:00 | 360.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-18 13:15:00 | 363.80 | 2025-12-19 14:15:00 | 369.35 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-19 09:30:00 | 365.20 | 2025-12-19 14:15:00 | 369.35 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-30 09:15:00 | 372.50 | 2026-01-01 15:15:00 | 375.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-30 10:45:00 | 372.45 | 2026-01-01 15:15:00 | 375.30 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-31 12:45:00 | 373.55 | 2026-01-01 15:15:00 | 375.30 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-01-01 09:15:00 | 373.80 | 2026-01-01 15:15:00 | 375.30 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-01-27 11:45:00 | 354.10 | 2026-02-03 09:15:00 | 389.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-28 09:15:00 | 355.30 | 2026-02-03 09:15:00 | 390.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-28 12:30:00 | 353.90 | 2026-02-03 09:15:00 | 389.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-28 14:45:00 | 353.60 | 2026-02-03 09:15:00 | 388.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 09:30:00 | 357.05 | 2026-02-03 09:15:00 | 392.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 10:30:00 | 360.90 | 2026-02-03 09:15:00 | 396.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 442.15 | 2026-02-24 15:15:00 | 449.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-03-09 09:15:00 | 401.55 | 2026-03-10 13:15:00 | 424.50 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2026-03-10 11:00:00 | 418.40 | 2026-03-10 13:15:00 | 424.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-03-13 11:15:00 | 439.50 | 2026-03-13 12:15:00 | 428.75 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2026-03-17 11:15:00 | 419.85 | 2026-03-18 10:15:00 | 429.95 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-04-01 14:30:00 | 410.40 | 2026-04-08 09:15:00 | 424.00 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2026-04-02 09:15:00 | 401.90 | 2026-04-08 09:15:00 | 424.00 | STOP_HIT | 1.00 | -5.50% |
| BUY | retest2 | 2026-04-13 10:15:00 | 428.90 | 2026-04-22 14:15:00 | 471.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:15:00 | 430.00 | 2026-04-22 14:15:00 | 471.68 | TARGET_HIT | 1.00 | 9.69% |
| BUY | retest2 | 2026-04-15 09:15:00 | 428.80 | 2026-04-22 14:15:00 | 471.74 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2026-04-15 13:15:00 | 428.85 | 2026-04-23 09:15:00 | 473.00 | TARGET_HIT | 1.00 | 10.29% |
| BUY | retest2 | 2026-04-24 14:30:00 | 473.50 | 2026-05-04 09:15:00 | 520.85 | TARGET_HIT | 1.00 | 10.00% |
