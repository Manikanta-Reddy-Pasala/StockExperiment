# IIFL Finance Ltd. (IIFL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 460.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 243 |
| ALERT1 | 147 |
| ALERT2 | 146 |
| ALERT2_SKIP | 87 |
| ALERT3 | 399 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 184 |
| PARTIAL | 25 |
| TARGET_HIT | 22 |
| STOP_HIT | 171 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 214 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 82 / 132
- **Target hits / Stop hits / Partials:** 22 / 167 / 25
- **Avg / median % per leg:** 0.78% / -0.82%
- **Sum % (uncompounded):** 167.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 92 | 35 | 38.0% | 17 | 74 | 1 | 1.28% | 118.0% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.04% | 6.3% |
| BUY @ 3rd Alert (retest2) | 86 | 32 | 37.2% | 17 | 69 | 0 | 1.30% | 111.8% |
| SELL (all) | 122 | 47 | 38.5% | 5 | 93 | 24 | 0.40% | 49.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 122 | 47 | 38.5% | 5 | 93 | 24 | 0.40% | 49.2% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.04% | 6.3% |
| retest2 (combined) | 208 | 79 | 38.0% | 22 | 162 | 24 | 0.77% | 161.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 442.26 | 435.62 | 435.37 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 14:15:00 | 431.92 | 434.88 | 435.05 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 440.03 | 435.48 | 435.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 10:15:00 | 440.80 | 436.54 | 435.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 12:15:00 | 433.32 | 436.14 | 435.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 12:15:00 | 433.32 | 436.14 | 435.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 12:15:00 | 433.32 | 436.14 | 435.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 13:00:00 | 433.32 | 436.14 | 435.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 13:15:00 | 434.68 | 435.85 | 435.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 14:15:00 | 433.52 | 435.85 | 435.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 426.28 | 433.94 | 434.79 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 14:15:00 | 449.35 | 436.30 | 435.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 15:15:00 | 455.57 | 440.15 | 437.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 15:15:00 | 446.83 | 447.03 | 443.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-19 09:15:00 | 444.79 | 447.03 | 443.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 445.95 | 446.81 | 443.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 09:45:00 | 444.49 | 446.81 | 443.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 445.37 | 446.52 | 443.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:45:00 | 443.38 | 446.52 | 443.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 440.46 | 445.31 | 443.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 12:00:00 | 440.46 | 445.31 | 443.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 12:15:00 | 436.87 | 443.62 | 442.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 12:30:00 | 437.02 | 443.62 | 442.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 440.46 | 442.32 | 442.23 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 09:15:00 | 436.14 | 441.09 | 441.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 10:15:00 | 434.83 | 439.83 | 441.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 10:15:00 | 434.34 | 434.09 | 436.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-23 10:45:00 | 436.14 | 434.09 | 436.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 436.19 | 434.51 | 436.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:45:00 | 436.34 | 434.51 | 436.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 434.64 | 434.54 | 436.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 12:45:00 | 436.87 | 434.54 | 436.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 437.02 | 435.03 | 436.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 14:00:00 | 437.02 | 435.03 | 436.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 430.26 | 434.08 | 436.08 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 11:15:00 | 444.64 | 437.51 | 437.05 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 14:15:00 | 432.21 | 436.83 | 436.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 09:15:00 | 429.68 | 434.83 | 435.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 09:15:00 | 426.96 | 426.71 | 429.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 13:15:00 | 427.06 | 425.88 | 428.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 13:15:00 | 427.06 | 425.88 | 428.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 14:00:00 | 427.06 | 425.88 | 428.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 425.94 | 425.20 | 427.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 13:00:00 | 423.12 | 424.93 | 426.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 10:15:00 | 423.03 | 424.10 | 425.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 13:15:00 | 423.66 | 424.45 | 425.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 14:15:00 | 434.78 | 426.68 | 426.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 434.78 | 426.68 | 426.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 452.22 | 433.79 | 430.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 12:15:00 | 453.63 | 454.56 | 447.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-05 13:00:00 | 453.63 | 454.56 | 447.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 476.45 | 465.93 | 461.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 09:30:00 | 462.80 | 465.93 | 461.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 471.79 | 476.92 | 473.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-12 15:00:00 | 471.79 | 476.92 | 473.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 15:15:00 | 473.05 | 476.14 | 473.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 09:15:00 | 473.98 | 476.14 | 473.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 09:15:00 | 469.80 | 474.88 | 473.52 | SL hit (close<static) qty=1.00 sl=470.14 alert=retest2 |

### Cycle 10 — SELL (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 11:15:00 | 486.41 | 493.32 | 493.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 13:15:00 | 484.08 | 490.43 | 492.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 10:15:00 | 476.55 | 473.96 | 479.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 10:15:00 | 476.55 | 473.96 | 479.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 476.55 | 473.96 | 479.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:00:00 | 476.55 | 473.96 | 479.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 462.17 | 463.11 | 469.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:00:00 | 462.17 | 463.11 | 469.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 469.22 | 460.87 | 464.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 469.22 | 460.87 | 464.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 470.92 | 462.88 | 464.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 473.00 | 462.88 | 464.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 477.91 | 465.89 | 465.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 10:15:00 | 479.85 | 468.68 | 467.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 13:15:00 | 479.12 | 480.43 | 476.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 14:00:00 | 479.12 | 480.43 | 476.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 484.81 | 486.30 | 484.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 13:30:00 | 485.10 | 486.30 | 484.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 15:15:00 | 486.65 | 486.16 | 484.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 09:15:00 | 494.62 | 486.16 | 484.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 10:00:00 | 489.76 | 486.88 | 484.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-18 09:15:00 | 538.74 | 532.95 | 529.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 13:15:00 | 554.36 | 559.16 | 559.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 15:15:00 | 552.02 | 556.81 | 558.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 563.39 | 558.13 | 558.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 563.39 | 558.13 | 558.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 563.39 | 558.13 | 558.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:30:00 | 563.88 | 558.13 | 558.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 564.02 | 559.30 | 559.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:00:00 | 564.02 | 559.30 | 559.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 564.85 | 560.41 | 559.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 573.59 | 564.03 | 561.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 583.45 | 592.32 | 582.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 583.45 | 592.32 | 582.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 583.45 | 592.32 | 582.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 10:00:00 | 583.45 | 592.32 | 582.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 578.88 | 589.63 | 581.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 578.88 | 589.63 | 581.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 576.55 | 587.01 | 581.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 12:00:00 | 576.55 | 587.01 | 581.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 15:15:00 | 569.41 | 577.65 | 578.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 562.37 | 574.59 | 576.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 565.87 | 564.82 | 569.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-04 10:00:00 | 565.87 | 564.82 | 569.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 573.54 | 566.57 | 570.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 11:00:00 | 573.54 | 566.57 | 570.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 581.21 | 569.50 | 571.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 12:00:00 | 581.21 | 569.50 | 571.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 14:15:00 | 578.88 | 573.33 | 572.68 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 14:15:00 | 571.35 | 572.97 | 573.07 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 09:15:00 | 577.91 | 573.81 | 573.43 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 569.90 | 573.03 | 573.11 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 11:15:00 | 578.25 | 574.08 | 573.58 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 11:15:00 | 569.02 | 573.27 | 573.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 12:15:00 | 565.82 | 571.78 | 572.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 09:15:00 | 564.17 | 559.31 | 563.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 564.17 | 559.31 | 563.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 564.17 | 559.31 | 563.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:30:00 | 565.04 | 559.31 | 563.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 563.92 | 560.23 | 563.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:30:00 | 565.33 | 560.23 | 563.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 11:15:00 | 558.58 | 559.90 | 563.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 13:30:00 | 554.31 | 558.14 | 561.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 10:15:00 | 559.31 | 553.19 | 553.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 559.31 | 553.19 | 553.07 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 14:15:00 | 552.66 | 552.99 | 553.02 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 09:15:00 | 564.89 | 555.40 | 554.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 14:15:00 | 594.18 | 575.22 | 567.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 10:15:00 | 576.79 | 578.73 | 571.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 10:30:00 | 576.45 | 578.73 | 571.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 569.70 | 576.58 | 572.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:00:00 | 569.70 | 576.58 | 572.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 567.08 | 574.68 | 571.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 567.08 | 574.68 | 571.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 11:15:00 | 573.78 | 574.44 | 572.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 11:45:00 | 571.45 | 574.44 | 572.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 571.94 | 573.94 | 572.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:00:00 | 571.94 | 573.94 | 572.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 570.97 | 573.34 | 572.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:45:00 | 571.06 | 573.34 | 572.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 567.27 | 572.13 | 571.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:30:00 | 567.27 | 572.13 | 571.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 567.57 | 571.22 | 571.41 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-25 10:15:00 | 576.26 | 572.31 | 571.88 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 567.27 | 571.30 | 571.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 12:15:00 | 565.33 | 570.11 | 570.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 14:15:00 | 570.09 | 569.74 | 570.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 14:15:00 | 570.09 | 569.74 | 570.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 570.09 | 569.74 | 570.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 15:00:00 | 570.09 | 569.74 | 570.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 572.67 | 570.40 | 570.73 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 10:15:00 | 574.12 | 571.14 | 571.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 11:15:00 | 576.89 | 572.29 | 571.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 09:15:00 | 572.86 | 574.28 | 573.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 572.86 | 574.28 | 573.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 572.86 | 574.28 | 573.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 10:00:00 | 572.86 | 574.28 | 573.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 570.82 | 573.59 | 572.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 10:45:00 | 571.06 | 573.59 | 572.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 571.01 | 573.07 | 572.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 11:30:00 | 570.09 | 573.07 | 572.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 13:15:00 | 570.43 | 572.33 | 572.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 10:15:00 | 568.73 | 571.19 | 571.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 12:15:00 | 570.97 | 570.88 | 571.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 12:15:00 | 570.97 | 570.88 | 571.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 570.97 | 570.88 | 571.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 12:30:00 | 572.57 | 570.88 | 571.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 564.46 | 569.60 | 570.91 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 11:15:00 | 583.69 | 572.44 | 571.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 14:15:00 | 591.46 | 578.46 | 574.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 09:15:00 | 577.33 | 579.86 | 576.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-01 10:00:00 | 577.33 | 579.86 | 576.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 577.23 | 578.98 | 576.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:45:00 | 576.50 | 578.98 | 576.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 577.86 | 578.78 | 577.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 577.96 | 578.78 | 577.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 578.25 | 578.67 | 577.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:30:00 | 583.25 | 578.91 | 577.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 10:45:00 | 582.14 | 583.61 | 581.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 13:15:00 | 577.38 | 580.90 | 580.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 13:15:00 | 577.38 | 580.90 | 580.98 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 11:15:00 | 582.33 | 578.95 | 578.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 14:15:00 | 600.54 | 584.66 | 581.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 13:15:00 | 599.04 | 600.93 | 592.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-11 14:00:00 | 599.04 | 600.93 | 592.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 579.56 | 596.63 | 592.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 580.24 | 596.63 | 592.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 579.42 | 593.19 | 591.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 576.07 | 593.19 | 591.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 560.48 | 586.65 | 588.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 15:15:00 | 558.53 | 571.06 | 579.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 566.16 | 565.30 | 572.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 14:15:00 | 566.16 | 565.30 | 572.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 566.16 | 565.30 | 572.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 566.16 | 565.30 | 572.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 576.16 | 567.79 | 572.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 576.79 | 567.79 | 572.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 571.16 | 568.46 | 572.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:15:00 | 570.67 | 569.31 | 572.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 14:15:00 | 570.33 | 570.35 | 572.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 15:00:00 | 568.29 | 569.94 | 571.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 10:15:00 | 579.66 | 573.88 | 573.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 10:15:00 | 579.66 | 573.88 | 573.34 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 14:15:00 | 570.43 | 573.97 | 574.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 15:15:00 | 568.25 | 572.83 | 573.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 11:15:00 | 573.73 | 572.40 | 573.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 11:15:00 | 573.73 | 572.40 | 573.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 573.73 | 572.40 | 573.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 12:00:00 | 573.73 | 572.40 | 573.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 575.63 | 573.04 | 573.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 12:30:00 | 575.63 | 573.04 | 573.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 571.31 | 572.70 | 573.21 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 09:15:00 | 576.45 | 573.61 | 573.50 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 14:15:00 | 569.12 | 573.43 | 573.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 10:15:00 | 563.39 | 569.94 | 571.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 571.60 | 563.05 | 564.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 571.60 | 563.05 | 564.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 571.60 | 563.05 | 564.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:00:00 | 571.60 | 563.05 | 564.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 565.33 | 563.51 | 564.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 11:30:00 | 563.58 | 563.62 | 564.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 09:15:00 | 576.02 | 565.90 | 565.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 09:15:00 | 576.02 | 565.90 | 565.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 579.37 | 571.23 | 568.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 571.40 | 572.15 | 569.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 12:30:00 | 571.84 | 572.15 | 569.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 580.87 | 578.06 | 575.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 09:15:00 | 582.91 | 578.06 | 575.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 10:30:00 | 585.15 | 580.90 | 576.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-10-16 09:15:00 | 641.20 | 621.99 | 618.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 635.27 | 642.96 | 643.08 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 09:15:00 | 647.07 | 643.78 | 643.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 10:15:00 | 665.19 | 648.06 | 645.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 12:15:00 | 646.00 | 647.75 | 645.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 12:15:00 | 646.00 | 647.75 | 645.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 646.00 | 647.75 | 645.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 13:15:00 | 643.14 | 647.75 | 645.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 639.06 | 646.01 | 645.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:00:00 | 639.06 | 646.01 | 645.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 14:15:00 | 637.46 | 644.30 | 644.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 631.14 | 639.89 | 642.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 611.96 | 606.65 | 618.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 611.96 | 606.65 | 618.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 611.96 | 606.65 | 618.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:45:00 | 618.76 | 606.65 | 618.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 589.13 | 584.01 | 594.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 592.63 | 584.01 | 594.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 596.61 | 586.53 | 594.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 598.84 | 586.53 | 594.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 598.26 | 588.88 | 594.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:00:00 | 598.26 | 588.88 | 594.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 12:15:00 | 592.38 | 589.58 | 594.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 15:00:00 | 587.19 | 589.51 | 593.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-30 10:15:00 | 603.26 | 592.55 | 594.01 | SL hit (close>static) qty=1.00 sl=598.50 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 12:15:00 | 601.81 | 594.34 | 593.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 15:15:00 | 603.07 | 597.91 | 595.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 594.28 | 597.18 | 595.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 594.28 | 597.18 | 595.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 594.28 | 597.18 | 595.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:00:00 | 594.28 | 597.18 | 595.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 598.36 | 597.42 | 595.69 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 590.34 | 594.53 | 594.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 13:15:00 | 587.38 | 591.55 | 593.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 592.68 | 591.25 | 592.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 09:15:00 | 592.68 | 591.25 | 592.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 592.68 | 591.25 | 592.63 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 13:15:00 | 596.27 | 593.66 | 593.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 14:15:00 | 600.79 | 595.08 | 594.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 09:15:00 | 592.87 | 595.34 | 594.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 592.87 | 595.34 | 594.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 592.87 | 595.34 | 594.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 09:30:00 | 592.38 | 595.34 | 594.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 593.79 | 595.03 | 594.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 10:30:00 | 592.38 | 595.03 | 594.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 12:15:00 | 594.42 | 594.67 | 594.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 12:30:00 | 593.89 | 594.67 | 594.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 595.15 | 594.77 | 594.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:30:00 | 594.62 | 594.77 | 594.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 595.64 | 594.94 | 594.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 15:15:00 | 593.50 | 594.94 | 594.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 593.50 | 594.65 | 594.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 09:15:00 | 597.78 | 594.65 | 594.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 11:15:00 | 597.29 | 595.89 | 595.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 13:00:00 | 597.29 | 596.39 | 595.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 602.20 | 595.99 | 595.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 594.33 | 596.10 | 595.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:00:00 | 594.33 | 596.10 | 595.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 593.55 | 595.59 | 595.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:15:00 | 592.92 | 595.59 | 595.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-08 12:15:00 | 592.87 | 595.04 | 595.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 12:15:00 | 592.87 | 595.04 | 595.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 13:15:00 | 591.46 | 594.33 | 594.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 09:15:00 | 596.76 | 594.59 | 594.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 09:15:00 | 596.76 | 594.59 | 594.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 596.76 | 594.59 | 594.83 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 10:15:00 | 596.95 | 595.07 | 595.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 12:15:00 | 599.33 | 596.41 | 595.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 15:15:00 | 599.57 | 600.04 | 598.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 15:15:00 | 599.57 | 600.04 | 598.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 599.57 | 600.04 | 598.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 580.73 | 600.04 | 598.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 589.57 | 597.95 | 597.87 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 09:15:00 | 594.96 | 597.35 | 597.60 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 10:15:00 | 601.27 | 598.13 | 597.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 11:15:00 | 606.13 | 599.73 | 598.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 12:15:00 | 607.10 | 607.38 | 604.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-15 12:30:00 | 606.61 | 607.38 | 604.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 13:15:00 | 603.75 | 606.65 | 604.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-15 14:00:00 | 603.75 | 606.65 | 604.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 14:15:00 | 616.81 | 608.68 | 605.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 09:15:00 | 617.35 | 608.95 | 605.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 13:00:00 | 617.69 | 613.71 | 609.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 09:15:00 | 601.71 | 614.77 | 611.62 | SL hit (close<static) qty=1.00 sl=602.29 alert=retest2 |

### Cycle 48 — SELL (started 2023-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 12:15:00 | 605.30 | 609.75 | 609.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 13:15:00 | 601.76 | 608.15 | 609.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 579.42 | 574.74 | 581.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 10:00:00 | 579.42 | 574.74 | 581.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 577.96 | 575.39 | 581.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 13:00:00 | 576.02 | 575.92 | 580.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 13:15:00 | 590.10 | 578.75 | 581.63 | SL hit (close>static) qty=1.00 sl=582.33 alert=retest2 |

### Cycle 49 — BUY (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 10:15:00 | 589.71 | 584.33 | 583.67 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 578.93 | 583.73 | 583.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 10:15:00 | 578.25 | 582.63 | 583.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 10:15:00 | 584.08 | 580.82 | 581.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 10:15:00 | 584.08 | 580.82 | 581.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 584.08 | 580.82 | 581.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:45:00 | 584.57 | 580.82 | 581.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 579.71 | 580.60 | 581.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 12:15:00 | 579.17 | 580.60 | 581.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 13:30:00 | 579.17 | 579.98 | 581.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 14:00:00 | 578.01 | 579.98 | 581.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 15:00:00 | 579.27 | 579.84 | 580.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 581.85 | 580.24 | 581.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 09:45:00 | 578.25 | 580.30 | 580.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 10:45:00 | 577.47 | 579.54 | 580.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 12:15:00 | 588.64 | 582.09 | 581.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 588.64 | 582.09 | 581.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 592.72 | 586.98 | 584.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 15:15:00 | 591.56 | 593.91 | 589.71 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:15:00 | 602.68 | 593.91 | 589.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 605.89 | 611.74 | 605.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 606.66 | 611.74 | 605.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 606.18 | 610.63 | 605.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 12:30:00 | 606.13 | 610.63 | 605.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 606.03 | 608.99 | 605.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 15:15:00 | 609.63 | 608.99 | 605.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 12:15:00 | 611.57 | 614.50 | 613.15 | SL hit (close<ema400) qty=1.00 sl=613.15 alert=retest1 |

### Cycle 52 — SELL (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 15:15:00 | 610.50 | 612.29 | 612.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 09:15:00 | 607.10 | 611.25 | 611.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 13:15:00 | 611.62 | 610.12 | 610.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 13:15:00 | 611.62 | 610.12 | 610.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 13:15:00 | 611.62 | 610.12 | 610.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 14:00:00 | 611.62 | 610.12 | 610.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 14:15:00 | 611.03 | 610.30 | 610.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 14:45:00 | 613.51 | 610.30 | 610.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 609.04 | 610.05 | 610.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:15:00 | 613.27 | 610.05 | 610.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 616.81 | 611.40 | 611.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 10:15:00 | 622.35 | 613.59 | 612.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 14:15:00 | 645.23 | 646.66 | 639.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-14 15:00:00 | 645.23 | 646.66 | 639.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 639.69 | 643.81 | 641.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 12:45:00 | 639.69 | 643.81 | 641.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 13:15:00 | 642.12 | 643.47 | 641.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:45:00 | 642.94 | 643.47 | 641.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 628.32 | 640.44 | 639.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 628.32 | 640.44 | 639.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 15:15:00 | 628.47 | 638.05 | 638.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 14:15:00 | 622.64 | 630.16 | 634.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 09:15:00 | 633.86 | 630.84 | 633.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 633.86 | 630.84 | 633.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 633.86 | 630.84 | 633.83 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 12:15:00 | 650.81 | 636.89 | 636.02 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 621.09 | 634.82 | 636.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 607.10 | 629.28 | 633.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-26 10:15:00 | 565.04 | 561.11 | 576.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-26 10:30:00 | 564.51 | 561.11 | 576.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 13:15:00 | 571.74 | 566.29 | 574.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 13:45:00 | 574.85 | 566.29 | 574.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 575.05 | 568.04 | 574.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 14:30:00 | 574.07 | 568.04 | 574.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 576.99 | 569.83 | 575.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:15:00 | 577.67 | 569.83 | 575.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 576.16 | 571.09 | 575.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:45:00 | 579.42 | 571.09 | 575.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 573.59 | 571.59 | 575.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 14:30:00 | 568.20 | 571.93 | 574.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 14:15:00 | 580.58 | 574.73 | 574.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2023-12-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 14:15:00 | 580.58 | 574.73 | 574.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 10:15:00 | 582.67 | 577.45 | 575.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 15:15:00 | 577.96 | 580.25 | 578.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 15:15:00 | 577.96 | 580.25 | 578.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 15:15:00 | 577.96 | 580.25 | 578.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 09:15:00 | 574.80 | 580.25 | 578.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 575.97 | 579.40 | 578.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 11:15:00 | 580.39 | 579.07 | 577.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 15:15:00 | 597.39 | 607.43 | 607.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 15:15:00 | 597.39 | 607.43 | 607.53 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 615.16 | 608.97 | 608.23 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 12:15:00 | 608.07 | 608.48 | 608.49 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 14:15:00 | 610.16 | 608.83 | 608.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 09:15:00 | 619.78 | 610.90 | 609.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 15:15:00 | 638.96 | 640.65 | 634.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 15:15:00 | 638.96 | 640.65 | 634.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 15:15:00 | 638.96 | 640.65 | 634.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 09:15:00 | 646.93 | 640.65 | 634.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 12:15:00 | 630.70 | 638.76 | 635.41 | SL hit (close<static) qty=1.00 sl=632.45 alert=retest2 |

### Cycle 62 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 629.54 | 633.03 | 633.41 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-01-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 10:15:00 | 636.92 | 633.42 | 633.17 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-01-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 12:15:00 | 624.59 | 631.48 | 632.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 13:15:00 | 618.17 | 628.82 | 631.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 10:15:00 | 627.40 | 625.68 | 628.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 10:15:00 | 627.40 | 625.68 | 628.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 627.40 | 625.68 | 628.53 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 13:15:00 | 636.24 | 630.21 | 630.08 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 14:15:00 | 621.67 | 631.61 | 631.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 617.49 | 626.81 | 629.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 595.10 | 589.27 | 597.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-29 10:00:00 | 595.10 | 589.27 | 597.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 601.32 | 591.68 | 597.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 11:00:00 | 601.32 | 591.68 | 597.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 606.61 | 594.66 | 598.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 12:00:00 | 606.61 | 594.66 | 598.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 15:15:00 | 600.30 | 598.61 | 599.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:15:00 | 601.52 | 598.61 | 599.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 606.23 | 600.13 | 600.01 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 596.42 | 600.71 | 600.75 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 12:15:00 | 605.84 | 600.90 | 600.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 13:15:00 | 606.52 | 602.03 | 601.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 597.87 | 602.08 | 601.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 597.87 | 602.08 | 601.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 597.87 | 602.08 | 601.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:45:00 | 596.56 | 602.08 | 601.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 10:15:00 | 592.53 | 600.17 | 600.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 13:15:00 | 588.69 | 595.76 | 598.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 15:15:00 | 578.01 | 577.86 | 583.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-06 09:15:00 | 577.86 | 577.86 | 583.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 570.63 | 576.42 | 582.04 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 12:15:00 | 592.48 | 582.84 | 581.96 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 14:15:00 | 573.30 | 581.53 | 582.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 570.43 | 577.96 | 580.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 12:15:00 | 558.05 | 557.53 | 563.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 13:00:00 | 558.05 | 557.53 | 563.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 561.45 | 558.32 | 562.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:30:00 | 561.11 | 558.32 | 562.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 560.48 | 558.79 | 560.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 560.48 | 558.79 | 560.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 558.00 | 558.63 | 560.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:30:00 | 559.31 | 558.63 | 560.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 563.20 | 559.50 | 560.68 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 566.25 | 561.72 | 561.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 12:15:00 | 582.62 | 566.28 | 563.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 12:15:00 | 588.69 | 591.56 | 585.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 13:00:00 | 588.69 | 591.56 | 585.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 583.74 | 589.30 | 585.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 15:00:00 | 583.74 | 589.30 | 585.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 583.30 | 588.10 | 585.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:15:00 | 591.12 | 588.10 | 585.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 10:30:00 | 585.00 | 587.49 | 585.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 14:30:00 | 586.07 | 586.56 | 585.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 15:00:00 | 586.65 | 586.56 | 585.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 582.91 | 585.83 | 585.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 582.91 | 585.83 | 585.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 583.35 | 585.33 | 585.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:00:00 | 583.35 | 585.33 | 585.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-22 10:15:00 | 583.74 | 585.01 | 585.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 10:15:00 | 583.74 | 585.01 | 585.08 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-02-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 12:15:00 | 586.12 | 585.16 | 585.13 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 15:15:00 | 582.82 | 584.90 | 585.04 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 09:15:00 | 590.73 | 585.15 | 584.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 09:15:00 | 597.39 | 590.19 | 587.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 13:15:00 | 588.16 | 592.57 | 589.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 13:15:00 | 588.16 | 592.57 | 589.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 588.16 | 592.57 | 589.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:00:00 | 588.16 | 592.57 | 589.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 591.90 | 592.44 | 590.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:00:00 | 591.90 | 592.44 | 590.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 588.69 | 591.69 | 589.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 590.05 | 591.69 | 589.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 581.21 | 589.59 | 589.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:00:00 | 581.21 | 589.59 | 589.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 557.17 | 583.11 | 586.28 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 588.84 | 577.56 | 576.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 593.02 | 580.65 | 578.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 589.62 | 594.42 | 588.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 589.62 | 594.42 | 588.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 589.62 | 594.42 | 588.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:00:00 | 589.62 | 594.42 | 588.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 585.73 | 592.68 | 588.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:45:00 | 585.49 | 592.68 | 588.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 583.79 | 590.90 | 588.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 12:00:00 | 583.79 | 590.90 | 588.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 15:15:00 | 580.87 | 585.81 | 586.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 464.07 | 561.46 | 575.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 404.86 | 397.83 | 446.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 09:45:00 | 407.49 | 397.83 | 446.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 377.67 | 382.10 | 392.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 12:30:00 | 373.10 | 378.54 | 388.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 13:30:00 | 374.02 | 376.94 | 386.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 14:00:00 | 370.57 | 376.94 | 386.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 09:15:00 | 370.72 | 376.10 | 384.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-14 14:15:00 | 355.32 | 367.90 | 376.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 09:15:00 | 354.44 | 364.79 | 373.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 10:15:00 | 352.04 | 362.01 | 371.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 10:15:00 | 352.18 | 362.01 | 371.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-15 15:15:00 | 358.77 | 358.26 | 365.63 | SL hit (close>ema200) qty=0.50 sl=358.26 alert=retest2 |

### Cycle 81 — BUY (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 13:15:00 | 341.63 | 324.47 | 323.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 12:15:00 | 343.47 | 341.04 | 338.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 11:15:00 | 341.34 | 341.45 | 339.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 11:30:00 | 341.34 | 341.45 | 339.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 414.29 | 421.41 | 416.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:30:00 | 413.07 | 421.41 | 416.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 413.99 | 419.93 | 415.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:15:00 | 412.54 | 419.93 | 415.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 406.42 | 417.23 | 415.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 15:00:00 | 406.42 | 417.23 | 415.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 393.79 | 411.10 | 412.53 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 13:15:00 | 420.26 | 413.74 | 413.29 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 15:15:00 | 410.89 | 413.96 | 414.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 401.56 | 409.23 | 411.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 402.43 | 400.71 | 405.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 402.43 | 400.71 | 405.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 402.43 | 400.71 | 405.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:30:00 | 404.23 | 400.71 | 405.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 10:15:00 | 405.45 | 401.66 | 405.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 11:00:00 | 405.45 | 401.66 | 405.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 11:15:00 | 406.56 | 402.64 | 405.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 11:45:00 | 406.81 | 402.64 | 405.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 407.87 | 403.69 | 405.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 13:15:00 | 414.67 | 403.69 | 405.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 409.91 | 406.72 | 406.71 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 15:15:00 | 406.03 | 406.58 | 406.64 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 412.15 | 407.70 | 407.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 12:15:00 | 418.60 | 411.82 | 410.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 13:15:00 | 413.70 | 415.06 | 413.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 13:15:00 | 413.70 | 415.06 | 413.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 413.70 | 415.06 | 413.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 14:00:00 | 413.70 | 415.06 | 413.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 415.00 | 415.17 | 413.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:30:00 | 413.90 | 414.87 | 413.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 412.40 | 414.37 | 413.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:00:00 | 412.40 | 414.37 | 413.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 414.10 | 414.32 | 413.71 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 15:15:00 | 411.90 | 413.47 | 413.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 09:15:00 | 406.95 | 412.17 | 412.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 10:15:00 | 401.50 | 400.14 | 403.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-03 11:00:00 | 401.50 | 400.14 | 403.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 400.55 | 399.63 | 402.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:00:00 | 400.55 | 399.63 | 402.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 402.40 | 400.18 | 402.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 398.35 | 400.18 | 402.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 393.50 | 398.85 | 401.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 12:30:00 | 391.30 | 395.45 | 399.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 14:15:00 | 371.74 | 392.03 | 396.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 401.85 | 393.33 | 396.57 | SL hit (close>ema200) qty=0.50 sl=393.33 alert=retest2 |

### Cycle 89 — BUY (started 2024-05-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 13:15:00 | 399.00 | 388.98 | 388.30 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 14:15:00 | 393.70 | 396.19 | 396.43 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 401.60 | 397.04 | 396.76 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 396.95 | 398.49 | 398.65 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 14:15:00 | 399.95 | 398.80 | 398.73 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 398.00 | 398.64 | 398.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 12:15:00 | 395.60 | 397.89 | 398.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 397.45 | 397.29 | 397.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 397.45 | 397.29 | 397.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 397.45 | 397.29 | 397.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 399.65 | 397.29 | 397.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 397.00 | 397.23 | 397.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:30:00 | 397.50 | 397.23 | 397.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 396.55 | 397.10 | 397.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:15:00 | 396.55 | 397.10 | 397.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 396.00 | 396.88 | 397.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 14:30:00 | 393.95 | 396.20 | 397.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:45:00 | 395.05 | 395.88 | 396.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 15:15:00 | 398.30 | 397.19 | 397.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 15:15:00 | 398.30 | 397.19 | 397.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 10:15:00 | 402.00 | 398.45 | 397.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 14:15:00 | 399.50 | 401.31 | 399.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 399.50 | 401.31 | 399.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 399.50 | 401.31 | 399.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 399.50 | 401.31 | 399.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 399.45 | 400.94 | 399.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 407.95 | 400.94 | 399.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 10:45:00 | 399.80 | 404.73 | 404.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 398.90 | 403.56 | 404.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 398.90 | 403.56 | 404.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 10:15:00 | 395.80 | 400.17 | 401.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 14:15:00 | 401.15 | 400.19 | 401.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 14:15:00 | 401.15 | 400.19 | 401.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 401.15 | 400.19 | 401.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:45:00 | 401.20 | 400.19 | 401.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 400.00 | 400.15 | 401.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 389.45 | 400.15 | 401.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 369.98 | 388.86 | 395.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 350.50 | 386.61 | 393.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 97 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 403.45 | 395.55 | 394.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 411.35 | 400.87 | 397.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 12:15:00 | 479.55 | 482.17 | 466.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 13:00:00 | 479.55 | 482.17 | 466.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 472.50 | 474.99 | 472.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 12:00:00 | 472.50 | 474.99 | 472.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 473.00 | 474.39 | 472.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 13:45:00 | 472.50 | 474.39 | 472.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 472.05 | 473.92 | 472.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 472.05 | 473.92 | 472.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 472.50 | 473.64 | 472.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 476.55 | 473.64 | 472.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 15:15:00 | 466.40 | 473.23 | 473.05 | SL hit (close<static) qty=1.00 sl=471.50 alert=retest2 |

### Cycle 98 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 474.10 | 479.26 | 479.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 472.80 | 477.28 | 478.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 10:15:00 | 482.05 | 477.45 | 478.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 10:15:00 | 482.05 | 477.45 | 478.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 482.05 | 477.45 | 478.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 482.05 | 477.45 | 478.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 485.35 | 479.03 | 478.86 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 12:15:00 | 474.70 | 478.16 | 478.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 13:15:00 | 474.50 | 477.43 | 478.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 15:15:00 | 476.95 | 476.92 | 477.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 15:15:00 | 476.95 | 476.92 | 477.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 476.95 | 476.92 | 477.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 476.10 | 476.92 | 477.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 474.40 | 476.42 | 477.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 472.85 | 476.42 | 477.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:45:00 | 473.75 | 475.68 | 476.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 472.50 | 474.59 | 476.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 472.95 | 474.94 | 475.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 474.85 | 474.86 | 475.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 474.00 | 474.86 | 475.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 472.00 | 472.85 | 474.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 468.45 | 472.55 | 474.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 12:15:00 | 485.90 | 468.90 | 469.59 | SL hit (close>static) qty=1.00 sl=479.75 alert=retest2 |

### Cycle 101 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 485.30 | 472.18 | 471.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 495.50 | 482.38 | 476.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 485.20 | 488.68 | 481.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 14:00:00 | 485.20 | 488.68 | 481.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 476.25 | 486.19 | 481.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 476.25 | 486.19 | 481.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 484.75 | 485.90 | 481.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 485.85 | 486.61 | 482.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-03 11:15:00 | 534.44 | 520.83 | 513.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 515.20 | 519.72 | 520.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 15:15:00 | 509.40 | 516.14 | 518.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 496.90 | 490.51 | 498.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 496.90 | 490.51 | 498.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 496.90 | 490.51 | 498.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 502.25 | 490.51 | 498.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 496.90 | 491.79 | 497.96 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 504.95 | 500.57 | 500.17 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 496.90 | 499.84 | 499.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 12:15:00 | 494.30 | 498.73 | 499.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 13:15:00 | 496.80 | 492.38 | 494.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 13:15:00 | 496.80 | 492.38 | 494.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 496.80 | 492.38 | 494.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 496.80 | 492.38 | 494.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 487.80 | 491.46 | 494.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:30:00 | 487.05 | 488.56 | 491.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 462.70 | 473.88 | 480.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-19 11:15:00 | 473.50 | 472.31 | 478.85 | SL hit (close>ema200) qty=0.50 sl=472.31 alert=retest2 |

### Cycle 105 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 461.75 | 444.47 | 444.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 10:15:00 | 467.30 | 449.04 | 446.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 10:15:00 | 455.65 | 455.72 | 452.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 12:15:00 | 454.10 | 455.12 | 452.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 454.10 | 455.12 | 452.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:45:00 | 454.30 | 455.12 | 452.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 450.60 | 453.91 | 452.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 15:00:00 | 450.60 | 453.91 | 452.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 450.60 | 453.25 | 452.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 421.75 | 453.25 | 452.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 430.75 | 448.75 | 450.18 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 441.45 | 435.29 | 434.85 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 430.15 | 433.76 | 434.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 11:15:00 | 427.25 | 430.14 | 431.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 402.95 | 402.86 | 408.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 402.95 | 402.86 | 408.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 402.95 | 402.86 | 408.35 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 423.65 | 410.66 | 409.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 428.25 | 414.18 | 411.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 462.20 | 465.20 | 459.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:30:00 | 462.70 | 465.20 | 459.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 463.05 | 464.24 | 460.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:45:00 | 460.70 | 464.24 | 460.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 459.65 | 463.00 | 460.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 459.65 | 463.00 | 460.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 458.80 | 462.16 | 460.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 458.90 | 462.16 | 460.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 458.00 | 461.33 | 459.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 474.50 | 461.33 | 459.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:30:00 | 460.75 | 462.08 | 460.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:15:00 | 460.00 | 461.23 | 460.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 458.55 | 460.79 | 460.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 458.55 | 460.79 | 460.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 455.45 | 459.72 | 460.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 450.90 | 450.88 | 454.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 450.90 | 450.88 | 454.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 450.35 | 450.89 | 453.99 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 457.50 | 455.90 | 455.71 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 454.20 | 455.56 | 455.57 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 457.80 | 456.01 | 455.77 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 449.90 | 454.71 | 455.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 446.85 | 453.14 | 454.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 13:15:00 | 460.90 | 453.50 | 454.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 13:15:00 | 460.90 | 453.50 | 454.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 460.90 | 453.50 | 454.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:00:00 | 460.90 | 453.50 | 454.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 457.05 | 454.21 | 454.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:15:00 | 456.50 | 454.21 | 454.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 456.50 | 454.67 | 454.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 461.70 | 454.67 | 454.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 459.50 | 455.63 | 455.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 12:15:00 | 465.20 | 458.50 | 456.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 10:15:00 | 460.90 | 463.29 | 460.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 10:15:00 | 460.90 | 463.29 | 460.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 460.90 | 463.29 | 460.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:30:00 | 460.25 | 463.29 | 460.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 460.60 | 462.61 | 460.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:00:00 | 460.60 | 462.61 | 460.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 458.00 | 461.69 | 460.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:00:00 | 458.00 | 461.69 | 460.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 461.40 | 461.63 | 460.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 15:15:00 | 461.95 | 461.63 | 460.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 10:30:00 | 464.10 | 461.77 | 460.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 11:00:00 | 462.15 | 461.77 | 460.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 12:45:00 | 462.30 | 461.65 | 460.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 460.85 | 461.49 | 460.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:00:00 | 460.85 | 461.49 | 460.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 467.80 | 462.75 | 461.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 15:15:00 | 472.70 | 462.75 | 461.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:00:00 | 471.00 | 469.26 | 465.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 456.30 | 465.79 | 465.18 | SL hit (close<static) qty=1.00 sl=457.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 459.10 | 464.45 | 464.62 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 471.00 | 464.23 | 464.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 13:15:00 | 472.00 | 467.90 | 466.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 466.65 | 467.65 | 466.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 466.65 | 467.65 | 466.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 466.65 | 467.65 | 466.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 466.65 | 467.65 | 466.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 466.50 | 467.42 | 466.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 468.65 | 467.42 | 466.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:45:00 | 470.65 | 467.26 | 466.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 12:00:00 | 470.50 | 467.78 | 467.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-13 14:15:00 | 515.51 | 494.28 | 483.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 490.50 | 506.34 | 508.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 13:15:00 | 484.60 | 501.99 | 506.03 | Break + close below crossover candle low |

### Cycle 119 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 542.05 | 508.59 | 507.89 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 512.10 | 517.43 | 517.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 506.35 | 513.31 | 515.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 467.80 | 467.61 | 476.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 467.80 | 467.61 | 476.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 467.80 | 467.61 | 476.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:30:00 | 460.10 | 464.85 | 471.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:00:00 | 462.50 | 462.14 | 464.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 437.10 | 453.48 | 459.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 439.38 | 453.48 | 459.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 446.60 | 442.88 | 449.80 | SL hit (close>ema200) qty=0.50 sl=442.88 alert=retest2 |

### Cycle 121 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 460.60 | 453.23 | 452.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 464.30 | 456.29 | 454.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 457.10 | 459.38 | 456.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 457.10 | 459.38 | 456.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 454.15 | 458.33 | 456.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 454.15 | 458.33 | 456.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 455.00 | 457.66 | 456.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 469.80 | 457.66 | 456.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 463.00 | 468.98 | 469.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 463.00 | 468.98 | 469.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 460.70 | 465.24 | 467.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 12:15:00 | 418.60 | 413.31 | 421.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 12:15:00 | 418.60 | 413.31 | 421.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 418.60 | 413.31 | 421.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:45:00 | 421.00 | 413.31 | 421.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 425.85 | 415.82 | 422.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:00:00 | 425.85 | 415.82 | 422.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 406.45 | 413.94 | 420.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:15:00 | 402.50 | 413.94 | 420.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 406.35 | 409.76 | 417.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:30:00 | 401.65 | 407.10 | 414.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:15:00 | 405.10 | 407.48 | 414.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 407.20 | 404.12 | 408.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:45:00 | 408.90 | 404.12 | 408.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 408.10 | 404.92 | 408.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:45:00 | 410.90 | 404.92 | 408.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 406.20 | 405.18 | 407.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:30:00 | 405.30 | 405.34 | 407.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 411.00 | 406.11 | 407.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:00:00 | 411.00 | 406.11 | 407.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 413.30 | 407.55 | 408.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:45:00 | 413.30 | 407.55 | 408.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-29 13:15:00 | 420.30 | 410.10 | 409.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 420.30 | 410.10 | 409.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 423.00 | 412.68 | 410.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 15:15:00 | 419.50 | 420.00 | 416.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 11:45:00 | 426.90 | 423.22 | 418.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-01 17:15:00 | 448.25 | 434.25 | 426.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-05 12:15:00 | 444.40 | 445.04 | 439.71 | SL hit (close<ema200) qty=0.50 sl=445.04 alert=retest1 |

### Cycle 124 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 444.95 | 450.58 | 450.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 443.05 | 447.68 | 449.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 12:15:00 | 447.55 | 445.76 | 447.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 12:15:00 | 447.55 | 445.76 | 447.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 447.55 | 445.76 | 447.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:00:00 | 447.55 | 445.76 | 447.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 441.50 | 444.91 | 446.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 439.15 | 444.33 | 446.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:15:00 | 439.75 | 442.62 | 444.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 417.19 | 423.83 | 428.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 417.76 | 423.83 | 428.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 427.25 | 420.93 | 423.96 | SL hit (close>ema200) qty=0.50 sl=420.93 alert=retest2 |

### Cycle 125 — BUY (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 12:15:00 | 413.10 | 412.72 | 412.67 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 412.05 | 412.59 | 412.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 15:15:00 | 410.70 | 412.08 | 412.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 413.80 | 412.43 | 412.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 413.80 | 412.43 | 412.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 413.80 | 412.43 | 412.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 415.40 | 412.43 | 412.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 412.10 | 412.36 | 412.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 411.90 | 412.36 | 412.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 12:15:00 | 416.25 | 413.16 | 412.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 416.25 | 413.16 | 412.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 13:15:00 | 421.85 | 414.90 | 413.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 421.15 | 421.37 | 418.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:30:00 | 421.40 | 421.37 | 418.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 415.00 | 419.57 | 418.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 414.80 | 419.57 | 418.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 415.65 | 418.78 | 418.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:45:00 | 418.15 | 419.07 | 418.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 12:15:00 | 433.80 | 434.40 | 434.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 12:15:00 | 433.80 | 434.40 | 434.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 15:15:00 | 430.80 | 433.33 | 433.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 434.25 | 433.51 | 433.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 434.25 | 433.51 | 433.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 434.25 | 433.51 | 433.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:45:00 | 435.15 | 433.51 | 433.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 437.40 | 434.29 | 434.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 444.15 | 438.12 | 436.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 15:15:00 | 444.25 | 444.38 | 441.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 436.90 | 442.89 | 440.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 436.90 | 442.89 | 440.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 436.90 | 442.89 | 440.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 436.40 | 441.59 | 440.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 436.40 | 441.59 | 440.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 433.35 | 438.73 | 439.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 432.75 | 436.15 | 437.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 433.05 | 430.33 | 434.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 12:15:00 | 433.05 | 430.33 | 434.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 433.05 | 430.33 | 434.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:45:00 | 434.45 | 430.33 | 434.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 435.90 | 431.44 | 434.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 436.00 | 431.44 | 434.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 435.00 | 432.15 | 434.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:45:00 | 436.35 | 432.15 | 434.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 420.80 | 423.99 | 428.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 415.50 | 420.64 | 425.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 12:15:00 | 431.15 | 424.52 | 425.69 | SL hit (close>static) qty=1.00 sl=428.65 alert=retest2 |

### Cycle 131 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 412.45 | 411.81 | 411.77 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 13:15:00 | 411.30 | 411.71 | 411.73 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 412.70 | 411.91 | 411.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 413.20 | 412.33 | 412.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 410.60 | 413.16 | 412.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 13:15:00 | 410.60 | 413.16 | 412.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 410.60 | 413.16 | 412.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:45:00 | 410.60 | 413.16 | 412.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 414.30 | 413.39 | 412.79 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 412.00 | 412.44 | 412.46 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 11:15:00 | 414.75 | 412.90 | 412.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 416.60 | 414.02 | 413.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 14:15:00 | 413.50 | 413.91 | 413.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 14:15:00 | 413.50 | 413.91 | 413.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 413.50 | 413.91 | 413.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 414.65 | 413.91 | 413.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 413.00 | 413.73 | 413.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 419.70 | 413.73 | 413.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 417.45 | 423.35 | 423.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 417.45 | 423.35 | 423.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 409.45 | 420.57 | 422.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 413.60 | 411.34 | 415.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 413.60 | 411.34 | 415.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 413.60 | 411.34 | 415.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 416.55 | 411.34 | 415.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 416.00 | 412.81 | 415.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 416.90 | 412.81 | 415.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 418.00 | 413.85 | 416.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 418.00 | 413.85 | 416.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 421.95 | 415.47 | 416.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 421.95 | 415.47 | 416.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 422.55 | 417.80 | 417.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 09:15:00 | 430.50 | 420.34 | 418.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 12:15:00 | 428.80 | 428.82 | 425.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 12:45:00 | 427.40 | 428.82 | 425.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 427.00 | 428.45 | 425.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 427.00 | 428.45 | 425.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 428.70 | 428.50 | 425.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:15:00 | 427.95 | 428.50 | 425.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 427.95 | 428.39 | 425.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 421.70 | 428.39 | 425.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 418.25 | 426.36 | 425.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 418.25 | 426.36 | 425.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 424.50 | 425.99 | 425.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 421.25 | 425.99 | 425.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 424.60 | 425.73 | 425.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 13:00:00 | 424.60 | 425.73 | 425.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 418.75 | 424.33 | 424.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 415.25 | 422.52 | 423.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 405.60 | 404.52 | 411.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 10:00:00 | 405.60 | 404.52 | 411.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 400.55 | 402.67 | 407.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:45:00 | 396.90 | 400.51 | 405.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 396.80 | 399.22 | 400.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:00:00 | 396.65 | 399.22 | 400.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 15:15:00 | 397.45 | 396.75 | 398.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 397.45 | 396.89 | 398.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 397.00 | 396.89 | 398.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 397.30 | 396.98 | 398.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-20 13:15:00 | 402.65 | 399.82 | 399.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 402.65 | 399.82 | 399.53 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 394.60 | 398.72 | 399.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 390.70 | 396.71 | 398.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 389.10 | 387.22 | 390.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 389.10 | 387.22 | 390.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 389.10 | 387.22 | 390.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 389.10 | 387.22 | 390.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 387.60 | 387.30 | 390.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 391.00 | 387.30 | 390.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 380.50 | 385.68 | 388.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 11:00:00 | 380.05 | 384.55 | 387.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 379.70 | 384.03 | 386.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:15:00 | 379.60 | 384.03 | 386.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 361.05 | 375.87 | 381.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 360.71 | 375.87 | 381.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 360.62 | 375.87 | 381.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 342.05 | 358.37 | 369.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 141 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 359.20 | 355.87 | 355.68 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 354.05 | 355.49 | 355.54 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 356.50 | 355.69 | 355.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 359.95 | 356.58 | 356.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 354.55 | 356.72 | 356.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 354.55 | 356.72 | 356.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 354.55 | 356.72 | 356.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 351.85 | 356.72 | 356.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 350.75 | 355.53 | 355.74 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 359.50 | 356.32 | 356.08 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 344.15 | 354.44 | 355.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 10:15:00 | 342.80 | 347.32 | 350.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 13:15:00 | 348.80 | 346.56 | 349.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 13:15:00 | 348.80 | 346.56 | 349.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 348.80 | 346.56 | 349.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 348.80 | 346.56 | 349.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 353.40 | 347.93 | 349.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 353.40 | 347.93 | 349.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 353.60 | 349.06 | 349.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 358.45 | 349.06 | 349.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 360.60 | 351.37 | 350.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 366.85 | 356.09 | 353.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 12:15:00 | 359.60 | 360.48 | 357.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 12:15:00 | 359.60 | 360.48 | 357.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 359.60 | 360.48 | 357.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 359.60 | 360.48 | 357.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 359.15 | 360.16 | 358.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 359.15 | 360.16 | 358.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 367.45 | 361.51 | 359.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:00:00 | 368.45 | 362.90 | 359.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 368.50 | 363.92 | 360.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 357.80 | 362.43 | 361.46 | SL hit (close<static) qty=1.00 sl=358.30 alert=retest2 |

### Cycle 148 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 358.50 | 360.68 | 360.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 14:15:00 | 356.60 | 359.86 | 360.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 12:15:00 | 341.85 | 336.75 | 341.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 12:15:00 | 341.85 | 336.75 | 341.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 341.85 | 336.75 | 341.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 13:00:00 | 341.85 | 336.75 | 341.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 335.75 | 336.55 | 341.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:15:00 | 333.10 | 336.55 | 341.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 316.44 | 328.49 | 335.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 315.65 | 313.45 | 320.18 | SL hit (close>ema200) qty=0.50 sl=313.45 alert=retest2 |

### Cycle 149 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 316.85 | 313.74 | 313.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 11:15:00 | 319.20 | 314.83 | 314.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 314.60 | 316.78 | 315.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 314.60 | 316.78 | 315.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 314.60 | 316.78 | 315.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 314.60 | 316.78 | 315.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 316.10 | 316.65 | 315.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:45:00 | 315.45 | 316.65 | 315.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 319.50 | 317.22 | 316.02 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 312.65 | 315.66 | 316.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 311.50 | 314.82 | 315.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 315.40 | 314.42 | 315.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 315.40 | 314.42 | 315.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 315.40 | 314.42 | 315.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 317.15 | 314.42 | 315.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 314.00 | 314.34 | 315.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:45:00 | 311.95 | 313.62 | 314.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:15:00 | 310.00 | 313.62 | 314.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 316.50 | 313.62 | 314.33 | SL hit (close>static) qty=1.00 sl=315.45 alert=retest2 |

### Cycle 151 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 317.05 | 315.07 | 314.82 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 309.50 | 313.85 | 314.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 305.20 | 312.12 | 313.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 10:15:00 | 293.65 | 293.10 | 298.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 11:15:00 | 294.15 | 293.10 | 298.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 297.05 | 289.76 | 294.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 297.05 | 289.76 | 294.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 299.45 | 291.70 | 294.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 299.45 | 291.70 | 294.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 305.70 | 297.00 | 296.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 314.60 | 300.52 | 298.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 319.50 | 320.24 | 313.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 319.50 | 320.24 | 313.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 315.00 | 319.59 | 317.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 315.00 | 319.59 | 317.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 312.00 | 318.07 | 317.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 306.00 | 318.07 | 317.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 304.60 | 315.38 | 316.02 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 319.00 | 312.42 | 312.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 321.15 | 314.16 | 313.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 324.25 | 326.87 | 322.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 324.25 | 326.87 | 322.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 333.10 | 336.79 | 334.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 333.10 | 336.79 | 334.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 333.10 | 336.05 | 334.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 331.90 | 336.05 | 334.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 332.90 | 335.42 | 334.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 332.45 | 335.42 | 334.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 329.55 | 334.25 | 333.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 329.55 | 334.25 | 333.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 13:15:00 | 329.35 | 333.27 | 333.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 327.10 | 331.39 | 332.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 334.00 | 331.91 | 332.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 334.00 | 331.91 | 332.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 334.00 | 331.91 | 332.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:30:00 | 329.80 | 331.91 | 332.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 331.40 | 331.81 | 332.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 12:15:00 | 329.85 | 332.04 | 332.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:00:00 | 330.80 | 331.37 | 332.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:45:00 | 329.80 | 330.97 | 331.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 10:45:00 | 330.20 | 330.08 | 331.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 328.10 | 329.33 | 330.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 332.45 | 329.33 | 330.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 335.95 | 330.65 | 331.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 335.95 | 330.65 | 331.02 | SL hit (close>static) qty=1.00 sl=334.30 alert=retest2 |

### Cycle 157 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 338.10 | 332.14 | 331.67 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 11:15:00 | 328.50 | 331.24 | 331.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 327.60 | 330.51 | 331.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 328.95 | 328.74 | 329.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 328.95 | 328.74 | 329.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 328.95 | 328.74 | 329.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 329.25 | 328.74 | 329.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 322.60 | 327.51 | 329.20 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 334.30 | 329.78 | 329.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 335.40 | 331.70 | 330.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 332.70 | 342.48 | 338.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 332.70 | 342.48 | 338.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 332.70 | 342.48 | 338.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 332.70 | 342.48 | 338.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 337.95 | 341.57 | 338.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 335.85 | 341.57 | 338.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 336.90 | 340.64 | 338.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 336.90 | 340.64 | 338.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 336.50 | 339.81 | 338.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:30:00 | 337.60 | 339.81 | 338.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 339.75 | 339.80 | 338.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:00:00 | 340.95 | 340.03 | 338.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 317.00 | 335.30 | 336.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 317.00 | 335.30 | 336.61 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 333.50 | 331.96 | 331.96 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 327.90 | 331.15 | 331.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 10:15:00 | 312.65 | 327.45 | 329.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-09 12:15:00 | 326.75 | 325.80 | 328.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 10:15:00 | 325.90 | 324.95 | 327.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 325.90 | 324.95 | 327.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:45:00 | 327.35 | 324.95 | 327.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 327.85 | 325.53 | 327.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 12:00:00 | 327.85 | 325.53 | 327.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 323.65 | 325.16 | 326.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 13:15:00 | 322.75 | 325.16 | 326.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 333.50 | 325.85 | 326.45 | SL hit (close>static) qty=1.00 sl=330.50 alert=retest2 |

### Cycle 163 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 333.25 | 327.33 | 327.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 337.00 | 332.33 | 330.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 332.65 | 332.72 | 330.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 14:30:00 | 332.40 | 332.72 | 330.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 332.65 | 332.89 | 331.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:15:00 | 339.45 | 332.89 | 331.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:30:00 | 336.60 | 335.15 | 333.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 336.00 | 334.80 | 333.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:45:00 | 337.85 | 335.64 | 333.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 11:15:00 | 373.40 | 353.37 | 345.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 364.00 | 366.83 | 367.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 362.35 | 365.49 | 366.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 362.10 | 361.66 | 363.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 362.10 | 361.66 | 363.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 362.10 | 361.66 | 363.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:00:00 | 349.35 | 355.43 | 357.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 366.50 | 355.69 | 356.27 | SL hit (close>static) qty=1.00 sl=364.30 alert=retest2 |

### Cycle 165 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 366.25 | 357.80 | 357.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 368.70 | 359.98 | 358.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 365.15 | 369.42 | 365.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 365.15 | 369.42 | 365.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 365.15 | 369.42 | 365.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 365.15 | 369.42 | 365.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 361.05 | 367.75 | 364.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 360.85 | 367.75 | 364.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 359.75 | 366.15 | 364.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 364.95 | 366.15 | 364.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 371.55 | 367.93 | 365.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:30:00 | 365.25 | 367.93 | 365.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 411.65 | 411.10 | 408.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 14:30:00 | 413.50 | 411.78 | 409.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:00:00 | 414.50 | 411.86 | 410.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 15:15:00 | 415.60 | 412.04 | 410.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:30:00 | 414.35 | 413.12 | 411.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 410.45 | 412.82 | 411.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 410.45 | 412.82 | 411.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 405.85 | 411.43 | 411.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 405.85 | 411.43 | 411.09 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |

### Cycle 166 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 407.65 | 410.67 | 410.78 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 412.70 | 410.99 | 410.87 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 409.05 | 410.61 | 410.70 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 413.50 | 411.18 | 410.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 414.85 | 412.62 | 411.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 412.40 | 412.58 | 411.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 11:00:00 | 412.40 | 412.58 | 411.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 413.55 | 412.77 | 411.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 412.75 | 412.77 | 411.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 413.65 | 412.95 | 412.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 412.90 | 412.95 | 412.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 412.85 | 412.93 | 412.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:30:00 | 412.55 | 412.93 | 412.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 414.20 | 413.18 | 412.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 412.45 | 413.18 | 412.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 415.65 | 413.77 | 412.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 424.80 | 414.27 | 413.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 418.70 | 416.48 | 414.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 418.00 | 416.59 | 415.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:30:00 | 419.80 | 417.41 | 415.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 417.20 | 418.34 | 416.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:45:00 | 417.15 | 418.34 | 416.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 417.60 | 418.19 | 416.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:45:00 | 418.15 | 418.54 | 417.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 425.60 | 428.11 | 428.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 425.60 | 428.11 | 428.28 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 428.50 | 427.75 | 427.72 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 427.05 | 427.61 | 427.66 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 429.40 | 427.97 | 427.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 432.60 | 429.93 | 428.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 480.20 | 480.69 | 470.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:15:00 | 485.15 | 480.69 | 470.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:45:00 | 483.40 | 484.68 | 478.92 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 11:30:00 | 483.60 | 483.95 | 479.55 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 477.10 | 482.57 | 479.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 477.10 | 482.57 | 479.69 | SL hit (close<ema400) qty=1.00 sl=479.69 alert=retest1 |

### Cycle 174 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 470.25 | 478.27 | 478.28 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 482.65 | 475.33 | 475.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 485.20 | 477.30 | 476.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 486.35 | 486.96 | 483.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 489.05 | 486.96 | 483.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 485.40 | 486.65 | 483.22 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 476.85 | 482.88 | 483.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 472.70 | 480.84 | 482.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 478.20 | 477.00 | 479.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 478.20 | 477.00 | 479.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 477.00 | 477.00 | 479.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 479.10 | 477.00 | 479.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 479.05 | 476.64 | 478.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 479.05 | 476.64 | 478.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 479.20 | 477.15 | 478.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 479.25 | 477.15 | 478.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 481.15 | 477.95 | 478.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 482.25 | 477.95 | 478.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 481.80 | 478.72 | 478.71 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 09:15:00 | 475.80 | 478.41 | 478.60 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 480.05 | 478.70 | 478.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 13:15:00 | 484.00 | 480.11 | 479.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 478.90 | 480.64 | 479.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 478.90 | 480.64 | 479.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 478.90 | 480.64 | 479.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 478.90 | 480.64 | 479.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 475.50 | 479.61 | 479.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 475.50 | 479.61 | 479.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 477.00 | 479.09 | 479.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 470.10 | 473.67 | 475.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 473.50 | 473.26 | 475.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 473.50 | 473.26 | 475.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 473.50 | 473.26 | 475.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 476.50 | 473.26 | 475.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 473.80 | 473.37 | 475.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 474.50 | 473.37 | 475.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 474.00 | 472.93 | 474.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 473.15 | 472.93 | 474.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 470.10 | 472.36 | 473.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 467.05 | 472.09 | 473.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 468.25 | 470.05 | 472.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 466.35 | 468.41 | 471.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 471.75 | 466.52 | 466.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 471.75 | 466.52 | 466.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 488.80 | 473.42 | 469.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 515.45 | 516.09 | 508.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 515.45 | 516.09 | 508.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 524.60 | 526.19 | 524.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 524.00 | 526.19 | 524.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 527.40 | 526.43 | 524.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 529.00 | 526.62 | 524.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 530.65 | 526.63 | 525.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 529.05 | 532.84 | 531.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 530.00 | 531.45 | 531.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 530.00 | 531.45 | 531.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 525.00 | 529.93 | 530.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 530.45 | 529.43 | 530.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 530.45 | 529.43 | 530.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 530.45 | 529.43 | 530.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 530.45 | 529.43 | 530.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 529.40 | 529.42 | 530.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 528.35 | 529.42 | 530.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 530.90 | 529.72 | 530.32 | SL hit (close>static) qty=1.00 sl=530.75 alert=retest2 |

### Cycle 183 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 536.50 | 531.71 | 531.16 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 530.00 | 530.83 | 530.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 517.40 | 527.84 | 529.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 506.40 | 504.04 | 510.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 506.40 | 504.04 | 510.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 513.60 | 505.95 | 510.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 513.60 | 505.95 | 510.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 518.05 | 508.37 | 511.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 513.85 | 508.37 | 511.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 441.15 | 440.67 | 447.42 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 453.95 | 449.69 | 449.23 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 446.00 | 449.60 | 449.72 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 450.00 | 449.23 | 449.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 453.20 | 450.03 | 449.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 450.00 | 450.65 | 450.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 450.00 | 450.65 | 450.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 450.00 | 450.65 | 450.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 450.00 | 450.65 | 450.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 446.85 | 449.89 | 449.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 450.30 | 449.89 | 449.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 448.00 | 449.51 | 449.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 448.00 | 449.51 | 449.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 446.80 | 448.97 | 449.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 447.55 | 446.57 | 447.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 447.55 | 446.57 | 447.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 447.55 | 446.57 | 447.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 453.95 | 446.57 | 447.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 447.00 | 446.66 | 447.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 446.75 | 446.66 | 447.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:00:00 | 445.60 | 446.42 | 447.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:00:00 | 446.70 | 445.58 | 446.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:45:00 | 446.80 | 445.83 | 446.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 450.00 | 446.67 | 446.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:00:00 | 450.00 | 446.67 | 446.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-18 14:15:00 | 449.15 | 447.16 | 446.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 449.15 | 447.16 | 446.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 453.55 | 448.61 | 447.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 467.65 | 469.51 | 464.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 465.85 | 468.37 | 464.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 465.85 | 468.37 | 464.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 465.85 | 468.37 | 464.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 461.85 | 467.06 | 464.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 461.85 | 467.06 | 464.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 463.95 | 466.44 | 464.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 461.95 | 466.44 | 464.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 460.40 | 464.14 | 463.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 461.30 | 464.14 | 463.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 458.35 | 462.43 | 462.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 457.00 | 461.34 | 462.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 457.50 | 456.31 | 458.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 457.50 | 456.31 | 458.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 460.65 | 457.18 | 458.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 460.90 | 457.18 | 458.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 460.70 | 457.88 | 458.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 461.90 | 457.88 | 458.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 456.85 | 457.92 | 458.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 454.45 | 457.92 | 458.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 455.25 | 456.93 | 458.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 15:15:00 | 431.73 | 440.96 | 447.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 15:15:00 | 432.49 | 440.96 | 447.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 439.60 | 438.60 | 445.16 | SL hit (close>ema200) qty=0.50 sl=438.60 alert=retest2 |

### Cycle 191 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 443.60 | 440.55 | 440.30 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 438.45 | 440.15 | 440.20 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 442.80 | 440.26 | 440.21 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 437.80 | 439.76 | 439.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 432.80 | 436.58 | 438.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 437.20 | 434.54 | 435.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 437.20 | 434.54 | 435.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 437.20 | 434.54 | 435.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 437.20 | 434.54 | 435.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 435.15 | 434.66 | 435.86 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 441.40 | 436.63 | 436.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 444.55 | 438.21 | 437.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 443.55 | 444.29 | 441.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 444.15 | 444.29 | 441.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 441.95 | 443.82 | 441.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 441.10 | 443.82 | 441.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 441.80 | 443.42 | 441.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 441.80 | 443.42 | 441.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 443.45 | 443.42 | 441.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 444.30 | 442.26 | 441.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:45:00 | 443.85 | 442.90 | 441.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 446.45 | 445.88 | 444.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 439.90 | 444.64 | 444.05 | SL hit (close<static) qty=1.00 sl=441.25 alert=retest2 |

### Cycle 196 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 438.55 | 442.84 | 443.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 10:15:00 | 437.25 | 440.34 | 441.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 436.70 | 436.62 | 438.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 11:15:00 | 436.70 | 436.62 | 438.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 436.70 | 436.62 | 438.58 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 447.95 | 440.74 | 439.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 451.75 | 442.94 | 441.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 445.75 | 446.03 | 443.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 445.75 | 446.03 | 443.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 445.75 | 446.03 | 443.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 445.55 | 446.03 | 443.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 445.85 | 446.00 | 443.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 448.30 | 445.27 | 444.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 449.20 | 452.03 | 451.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 446.00 | 450.83 | 451.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 446.00 | 450.83 | 451.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 443.80 | 448.48 | 449.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 430.90 | 427.28 | 433.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 430.90 | 427.28 | 433.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 432.95 | 428.42 | 433.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 432.95 | 428.42 | 433.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 429.20 | 428.57 | 433.02 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 448.85 | 436.94 | 435.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 450.20 | 443.55 | 439.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 445.00 | 445.67 | 441.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 10:00:00 | 445.00 | 445.67 | 441.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 485.70 | 490.00 | 487.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 485.15 | 490.00 | 487.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 487.35 | 489.47 | 487.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:30:00 | 488.65 | 489.24 | 487.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:30:00 | 488.45 | 489.23 | 487.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 482.90 | 488.17 | 488.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 482.90 | 488.17 | 488.19 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 495.90 | 489.18 | 488.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 504.20 | 492.18 | 489.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 500.20 | 501.37 | 497.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:30:00 | 500.05 | 501.37 | 497.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 499.55 | 500.81 | 498.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 501.75 | 500.81 | 498.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 495.50 | 499.74 | 497.81 | SL hit (close<static) qty=1.00 sl=498.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 491.75 | 496.95 | 497.21 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 501.85 | 497.74 | 497.48 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 491.75 | 497.37 | 497.59 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 504.30 | 496.01 | 495.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 511.55 | 503.44 | 499.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 508.80 | 510.53 | 505.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 508.80 | 510.53 | 505.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 508.80 | 510.53 | 505.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 11:15:00 | 519.65 | 511.63 | 506.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 517.35 | 512.73 | 507.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:00:00 | 517.40 | 513.66 | 508.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 526.45 | 537.38 | 537.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 526.45 | 537.38 | 537.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 524.35 | 533.42 | 535.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 527.25 | 525.38 | 529.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:45:00 | 526.00 | 525.38 | 529.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 525.15 | 525.51 | 528.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 532.00 | 525.51 | 528.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 526.60 | 525.73 | 528.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 527.80 | 525.73 | 528.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 529.80 | 526.54 | 528.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 533.05 | 526.54 | 528.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 535.35 | 528.30 | 529.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 533.90 | 528.30 | 529.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 542.70 | 531.18 | 530.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 544.15 | 538.68 | 534.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 12:15:00 | 539.25 | 539.74 | 536.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 13:00:00 | 539.25 | 539.74 | 536.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 536.40 | 539.53 | 537.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 536.40 | 539.53 | 537.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 539.45 | 539.51 | 537.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:30:00 | 535.50 | 539.51 | 537.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 541.05 | 539.57 | 537.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:45:00 | 542.40 | 540.63 | 539.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 15:15:00 | 536.45 | 538.91 | 538.78 | SL hit (close<static) qty=1.00 sl=537.25 alert=retest2 |

### Cycle 208 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 547.00 | 555.85 | 556.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 542.70 | 548.93 | 552.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 548.00 | 546.25 | 549.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 14:00:00 | 548.00 | 546.25 | 549.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 546.50 | 544.58 | 547.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 544.40 | 544.58 | 547.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 548.45 | 545.45 | 547.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 549.75 | 545.45 | 547.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 544.40 | 545.24 | 547.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:15:00 | 543.55 | 545.24 | 547.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 554.25 | 547.38 | 546.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 554.25 | 547.38 | 546.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 14:15:00 | 557.05 | 549.32 | 547.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 566.95 | 568.01 | 562.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:45:00 | 567.60 | 568.01 | 562.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 582.80 | 584.41 | 580.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 581.70 | 584.41 | 580.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 581.45 | 583.54 | 580.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:30:00 | 579.70 | 583.54 | 580.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 580.95 | 583.02 | 580.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 576.20 | 583.02 | 580.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 572.85 | 580.99 | 580.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 572.85 | 580.99 | 580.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 570.00 | 578.79 | 579.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 566.45 | 571.93 | 574.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 568.75 | 567.59 | 571.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:00:00 | 568.75 | 567.59 | 571.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 569.60 | 568.00 | 570.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 569.60 | 568.00 | 570.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 558.80 | 566.22 | 569.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:15:00 | 556.30 | 566.22 | 569.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 11:30:00 | 557.20 | 557.40 | 561.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 574.40 | 565.31 | 564.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 574.40 | 565.31 | 564.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 578.50 | 572.12 | 569.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 571.00 | 574.31 | 571.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 571.00 | 574.31 | 571.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 571.00 | 574.31 | 571.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 571.00 | 574.31 | 571.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 572.65 | 573.98 | 571.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:30:00 | 577.55 | 574.72 | 572.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 565.05 | 572.94 | 572.24 | SL hit (close<static) qty=1.00 sl=570.65 alert=retest2 |

### Cycle 212 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 565.65 | 571.48 | 571.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 564.15 | 566.94 | 568.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 569.25 | 565.33 | 566.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 569.25 | 565.33 | 566.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 569.25 | 565.33 | 566.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 569.25 | 565.33 | 566.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 566.15 | 565.49 | 566.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 568.75 | 565.49 | 566.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 561.30 | 564.65 | 566.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 556.65 | 564.65 | 566.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 560.70 | 563.17 | 564.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 560.00 | 561.71 | 563.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:45:00 | 560.70 | 561.61 | 563.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 565.25 | 562.34 | 563.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 565.25 | 562.34 | 563.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 564.40 | 562.75 | 563.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 571.00 | 562.75 | 563.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 572.60 | 564.72 | 564.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 572.60 | 564.72 | 564.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 578.90 | 568.64 | 566.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 569.30 | 571.11 | 568.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 569.30 | 571.11 | 568.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 569.50 | 570.79 | 568.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 572.55 | 570.79 | 568.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 571.80 | 571.50 | 569.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:00:00 | 571.15 | 572.09 | 570.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 572.60 | 572.38 | 570.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 586.10 | 575.22 | 572.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 591.10 | 575.22 | 572.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-02 09:15:00 | 629.80 | 620.29 | 612.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 647.45 | 650.67 | 650.72 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 654.70 | 651.02 | 650.76 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 648.20 | 650.23 | 650.45 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 15:15:00 | 652.30 | 650.58 | 650.56 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 644.00 | 649.26 | 649.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 631.45 | 645.70 | 648.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 644.90 | 644.80 | 647.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 644.90 | 644.80 | 647.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 644.35 | 643.74 | 645.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 637.05 | 641.21 | 644.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:45:00 | 638.55 | 635.51 | 636.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 635.45 | 635.51 | 636.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:15:00 | 639.10 | 636.36 | 636.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 639.25 | 636.94 | 637.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 639.05 | 636.94 | 637.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 645.15 | 638.58 | 637.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 645.15 | 638.58 | 637.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 13:15:00 | 648.50 | 640.57 | 638.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 640.05 | 643.51 | 640.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 640.05 | 643.51 | 640.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 640.05 | 643.51 | 640.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 631.50 | 643.51 | 640.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 637.65 | 642.33 | 640.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 637.65 | 642.33 | 640.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 640.40 | 641.95 | 640.56 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 631.55 | 639.06 | 639.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 629.15 | 637.08 | 638.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 624.90 | 624.27 | 630.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 624.90 | 624.27 | 630.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 624.90 | 624.27 | 630.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 624.90 | 624.27 | 630.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 621.85 | 622.99 | 628.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:45:00 | 623.80 | 622.99 | 628.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 595.30 | 617.50 | 625.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 591.80 | 617.50 | 625.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-22 14:15:00 | 532.62 | 593.18 | 609.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 11:15:00 | 561.60 | 548.61 | 548.11 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 534.50 | 546.05 | 547.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 516.00 | 530.74 | 538.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 494.20 | 490.89 | 504.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 494.20 | 490.89 | 504.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 503.45 | 494.71 | 503.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 494.35 | 493.92 | 502.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 528.00 | 506.93 | 505.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 528.00 | 506.93 | 505.73 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 498.95 | 508.55 | 509.29 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 513.25 | 508.24 | 508.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 516.80 | 511.15 | 509.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 11:15:00 | 525.70 | 525.95 | 522.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:00:00 | 525.70 | 525.95 | 522.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 521.45 | 524.97 | 523.13 | EMA400 retest candle locked (from upside) |

### Cycle 226 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 501.05 | 519.14 | 521.16 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 520.00 | 514.04 | 513.30 | EMA200 above EMA400 |

### Cycle 228 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 500.75 | 511.14 | 512.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 498.00 | 504.70 | 508.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 504.80 | 496.80 | 500.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 504.80 | 496.80 | 500.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 504.80 | 496.80 | 500.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 504.80 | 496.80 | 500.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 498.00 | 497.04 | 500.59 | EMA400 retest candle locked (from downside) |

### Cycle 229 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 515.40 | 503.37 | 502.74 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 501.35 | 503.21 | 503.22 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 507.65 | 503.86 | 503.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 509.40 | 505.86 | 504.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 505.50 | 506.30 | 505.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:15:00 | 503.30 | 506.30 | 505.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 506.00 | 506.24 | 505.18 | EMA400 retest candle locked (from upside) |

### Cycle 232 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 497.45 | 503.31 | 504.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 494.20 | 497.87 | 500.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 10:15:00 | 498.25 | 497.68 | 500.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 11:00:00 | 498.25 | 497.68 | 500.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 489.10 | 495.96 | 499.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:30:00 | 498.80 | 495.96 | 499.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 498.10 | 495.82 | 498.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 498.10 | 495.82 | 498.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 503.00 | 497.26 | 498.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 482.45 | 497.26 | 498.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 458.33 | 482.58 | 486.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 473.70 | 472.87 | 478.02 | SL hit (close>ema200) qty=0.50 sl=472.87 alert=retest2 |

### Cycle 233 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 491.15 | 482.30 | 481.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 492.90 | 484.42 | 482.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 484.00 | 485.01 | 482.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 484.00 | 485.01 | 482.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 484.00 | 485.01 | 482.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 484.45 | 485.01 | 482.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 483.25 | 484.66 | 482.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 483.25 | 484.66 | 482.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 481.00 | 483.93 | 482.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 481.00 | 483.93 | 482.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 483.50 | 483.84 | 482.82 | EMA400 retest candle locked (from upside) |

### Cycle 234 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 475.70 | 481.80 | 482.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 475.05 | 480.45 | 481.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 480.65 | 480.49 | 481.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 480.65 | 480.49 | 481.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 480.65 | 480.49 | 481.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 480.65 | 480.49 | 481.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 481.60 | 480.71 | 481.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 477.60 | 480.91 | 481.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:45:00 | 475.40 | 478.66 | 479.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 15:00:00 | 477.30 | 478.39 | 479.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 483.85 | 474.91 | 474.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 235 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 483.85 | 474.91 | 474.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 487.00 | 477.33 | 475.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 478.25 | 482.01 | 478.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 478.25 | 482.01 | 478.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 478.25 | 482.01 | 478.73 | EMA400 retest candle locked (from upside) |

### Cycle 236 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 467.20 | 476.17 | 476.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 447.70 | 469.10 | 472.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 452.30 | 452.19 | 459.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 450.70 | 452.19 | 459.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 453.45 | 452.44 | 458.53 | EMA400 retest candle locked (from downside) |

### Cycle 237 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 467.65 | 461.07 | 460.75 | EMA200 above EMA400 |

### Cycle 238 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 449.95 | 460.23 | 460.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 447.85 | 456.02 | 458.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 440.50 | 437.48 | 444.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 440.50 | 437.48 | 444.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 440.50 | 437.48 | 444.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 426.70 | 439.87 | 442.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 429.00 | 432.82 | 436.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 430.75 | 432.82 | 436.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 428.75 | 434.97 | 435.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 441.90 | 432.59 | 433.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 448.45 | 435.77 | 434.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 448.45 | 435.77 | 434.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 13:15:00 | 453.25 | 448.75 | 445.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 445.00 | 450.02 | 447.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 445.00 | 450.02 | 447.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 445.00 | 450.02 | 447.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 446.60 | 450.02 | 447.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 15:15:00 | 467.95 | 470.44 | 470.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 240 — SELL (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 15:15:00 | 467.95 | 470.44 | 470.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 447.85 | 465.92 | 468.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 437.25 | 421.38 | 429.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 437.25 | 421.38 | 429.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 437.25 | 421.38 | 429.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 437.25 | 421.38 | 429.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 437.50 | 424.61 | 430.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 435.25 | 424.61 | 430.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 241 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 443.50 | 433.62 | 433.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 449.40 | 439.57 | 436.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 445.50 | 445.75 | 441.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 445.50 | 445.75 | 441.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 447.80 | 456.49 | 453.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 447.80 | 456.49 | 453.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 441.50 | 453.49 | 452.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 440.00 | 453.49 | 452.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 242 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 441.90 | 451.18 | 451.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 440.05 | 445.27 | 448.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 446.50 | 445.52 | 448.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 446.50 | 445.52 | 448.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 446.50 | 445.52 | 448.15 | EMA400 retest candle locked (from downside) |

### Cycle 243 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 454.15 | 449.06 | 448.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 457.00 | 450.64 | 449.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 461.55 | 463.35 | 460.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 461.55 | 463.35 | 460.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 460.10 | 462.70 | 460.09 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-30 13:00:00 | 423.12 | 2023-05-31 14:15:00 | 434.78 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2023-05-31 10:15:00 | 423.03 | 2023-05-31 14:15:00 | 434.78 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2023-05-31 13:15:00 | 423.66 | 2023-05-31 14:15:00 | 434.78 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2023-06-13 09:15:00 | 473.98 | 2023-06-13 09:15:00 | 469.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-06-13 11:45:00 | 479.85 | 2023-06-20 11:15:00 | 486.41 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2023-07-04 09:15:00 | 494.62 | 2023-07-18 09:15:00 | 538.74 | TARGET_HIT | 1.00 | 8.92% |
| BUY | retest2 | 2023-07-04 10:00:00 | 489.76 | 2023-07-18 10:15:00 | 544.08 | TARGET_HIT | 1.00 | 11.09% |
| SELL | retest2 | 2023-08-11 13:30:00 | 554.31 | 2023-08-17 10:15:00 | 559.31 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2023-09-05 09:30:00 | 583.25 | 2023-09-06 13:15:00 | 577.38 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-09-06 10:45:00 | 582.14 | 2023-09-06 13:15:00 | 577.38 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-09-14 12:15:00 | 570.67 | 2023-09-15 10:15:00 | 579.66 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2023-09-14 14:15:00 | 570.33 | 2023-09-15 10:15:00 | 579.66 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-09-14 15:00:00 | 568.29 | 2023-09-15 10:15:00 | 579.66 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2023-09-26 11:30:00 | 563.58 | 2023-09-27 09:15:00 | 576.02 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2023-10-03 09:15:00 | 582.91 | 2023-10-16 09:15:00 | 641.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-03 10:30:00 | 585.15 | 2023-10-16 09:15:00 | 643.67 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-10-27 15:00:00 | 587.19 | 2023-10-30 10:15:00 | 603.26 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2023-10-30 13:15:00 | 588.45 | 2023-10-31 11:15:00 | 600.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-11-07 09:15:00 | 597.78 | 2023-11-08 12:15:00 | 592.87 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-11-07 11:15:00 | 597.29 | 2023-11-08 12:15:00 | 592.87 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-11-07 13:00:00 | 597.29 | 2023-11-08 12:15:00 | 592.87 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-11-08 09:15:00 | 602.20 | 2023-11-08 12:15:00 | 592.87 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2023-11-16 09:15:00 | 617.35 | 2023-11-17 09:15:00 | 601.71 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2023-11-16 13:00:00 | 617.69 | 2023-11-17 09:15:00 | 601.71 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2023-11-23 13:00:00 | 576.02 | 2023-11-23 13:15:00 | 590.10 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2023-11-29 12:15:00 | 579.17 | 2023-11-30 12:15:00 | 588.64 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-11-29 13:30:00 | 579.17 | 2023-11-30 12:15:00 | 588.64 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-11-29 14:00:00 | 578.01 | 2023-11-30 12:15:00 | 588.64 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2023-11-29 15:00:00 | 579.27 | 2023-11-30 12:15:00 | 588.64 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2023-11-30 09:45:00 | 578.25 | 2023-11-30 12:15:00 | 588.64 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2023-11-30 10:45:00 | 577.47 | 2023-11-30 12:15:00 | 588.64 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest1 | 2023-12-04 09:15:00 | 602.68 | 2023-12-08 12:15:00 | 611.57 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2023-12-05 15:15:00 | 609.63 | 2023-12-08 15:15:00 | 610.50 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2023-12-08 14:30:00 | 611.96 | 2023-12-08 15:15:00 | 610.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2023-12-27 14:30:00 | 568.20 | 2023-12-28 14:15:00 | 580.58 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-01-01 11:15:00 | 580.39 | 2024-01-08 15:15:00 | 597.39 | STOP_HIT | 1.00 | 2.93% |
| BUY | retest2 | 2024-01-16 09:15:00 | 646.93 | 2024-01-16 12:15:00 | 630.70 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-02-21 09:15:00 | 591.12 | 2024-02-22 10:15:00 | 583.74 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-02-21 10:30:00 | 585.00 | 2024-02-22 10:15:00 | 583.74 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-02-21 14:30:00 | 586.07 | 2024-02-22 10:15:00 | 583.74 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-02-21 15:00:00 | 586.65 | 2024-02-22 10:15:00 | 583.74 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-03-13 12:30:00 | 373.10 | 2024-03-14 14:15:00 | 355.32 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2024-03-13 13:30:00 | 374.02 | 2024-03-15 09:15:00 | 354.44 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2024-03-13 14:00:00 | 370.57 | 2024-03-15 10:15:00 | 352.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-14 09:15:00 | 370.72 | 2024-03-15 10:15:00 | 352.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-13 12:30:00 | 373.10 | 2024-03-15 15:15:00 | 358.77 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2024-03-13 13:30:00 | 374.02 | 2024-03-15 15:15:00 | 358.77 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2024-03-13 14:00:00 | 370.57 | 2024-03-15 15:15:00 | 358.77 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2024-03-14 09:15:00 | 370.72 | 2024-03-15 15:15:00 | 358.77 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2024-03-22 10:15:00 | 325.79 | 2024-03-26 09:15:00 | 309.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-22 12:45:00 | 325.21 | 2024-03-26 09:15:00 | 308.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-22 10:15:00 | 325.79 | 2024-03-27 12:15:00 | 343.38 | STOP_HIT | 0.50 | -5.40% |
| SELL | retest2 | 2024-03-22 12:45:00 | 325.21 | 2024-03-27 12:15:00 | 343.38 | STOP_HIT | 0.50 | -5.59% |
| SELL | retest2 | 2024-05-06 12:30:00 | 391.30 | 2024-05-06 14:15:00 | 371.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-06 12:30:00 | 391.30 | 2024-05-07 09:15:00 | 401.85 | STOP_HIT | 0.50 | -2.70% |
| SELL | retest2 | 2024-05-07 14:15:00 | 390.45 | 2024-05-10 09:15:00 | 370.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 15:00:00 | 390.95 | 2024-05-10 09:15:00 | 371.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 390.35 | 2024-05-10 09:15:00 | 370.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 14:15:00 | 390.45 | 2024-05-10 10:15:00 | 385.35 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2024-05-07 15:00:00 | 390.95 | 2024-05-10 10:15:00 | 385.35 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2024-05-09 09:15:00 | 390.35 | 2024-05-10 10:15:00 | 385.35 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2024-05-09 10:15:00 | 387.30 | 2024-05-10 12:15:00 | 398.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-05-24 14:30:00 | 393.95 | 2024-05-27 15:15:00 | 398.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-05-27 09:45:00 | 395.05 | 2024-05-27 15:15:00 | 398.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-05-29 09:15:00 | 407.95 | 2024-05-31 11:15:00 | 398.90 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-05-31 10:45:00 | 399.80 | 2024-05-31 11:15:00 | 398.90 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-06-04 09:15:00 | 389.45 | 2024-06-04 11:15:00 | 369.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 389.45 | 2024-06-04 12:15:00 | 350.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-14 09:15:00 | 476.55 | 2024-06-14 15:15:00 | 466.40 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-06-18 09:15:00 | 486.80 | 2024-06-19 12:15:00 | 474.10 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-06-19 10:30:00 | 474.80 | 2024-06-19 12:15:00 | 474.10 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-06-21 10:15:00 | 472.85 | 2024-06-26 12:15:00 | 485.90 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-06-21 11:45:00 | 473.75 | 2024-06-26 12:15:00 | 485.90 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-06-21 12:30:00 | 472.50 | 2024-06-26 12:15:00 | 485.90 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-06-24 09:15:00 | 472.95 | 2024-06-26 12:15:00 | 485.90 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-06-25 11:15:00 | 468.45 | 2024-06-26 12:15:00 | 485.90 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-06-28 09:30:00 | 485.85 | 2024-07-03 11:15:00 | 534.44 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-16 13:30:00 | 487.05 | 2024-07-19 09:15:00 | 462.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 13:30:00 | 487.05 | 2024-07-19 11:15:00 | 473.50 | STOP_HIT | 0.50 | 2.78% |
| BUY | retest2 | 2024-08-26 09:15:00 | 474.50 | 2024-08-28 11:15:00 | 458.55 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-08-26 11:30:00 | 460.75 | 2024-08-28 11:15:00 | 458.55 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-08-26 14:15:00 | 460.00 | 2024-08-28 11:15:00 | 458.55 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-09-04 15:15:00 | 461.95 | 2024-09-09 09:15:00 | 456.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-09-05 10:30:00 | 464.10 | 2024-09-09 09:15:00 | 456.30 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-09-05 11:00:00 | 462.15 | 2024-09-09 09:15:00 | 456.30 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-09-05 12:45:00 | 462.30 | 2024-09-09 09:15:00 | 456.30 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-09-05 15:15:00 | 472.70 | 2024-09-09 09:15:00 | 456.30 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2024-09-06 13:00:00 | 471.00 | 2024-09-09 09:15:00 | 456.30 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-09-12 09:15:00 | 468.65 | 2024-09-13 14:15:00 | 515.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-12 09:45:00 | 470.65 | 2024-09-13 14:15:00 | 517.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-12 12:00:00 | 470.50 | 2024-09-13 14:15:00 | 517.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-03 09:30:00 | 460.10 | 2024-10-07 10:15:00 | 437.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 13:00:00 | 462.50 | 2024-10-07 10:15:00 | 439.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:30:00 | 460.10 | 2024-10-08 10:15:00 | 446.60 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2024-10-04 13:00:00 | 462.50 | 2024-10-08 10:15:00 | 446.60 | STOP_HIT | 0.50 | 3.44% |
| BUY | retest2 | 2024-10-11 09:15:00 | 469.80 | 2024-10-17 09:15:00 | 463.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-10-24 15:15:00 | 402.50 | 2024-10-29 13:15:00 | 420.30 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2024-10-25 09:30:00 | 406.35 | 2024-10-29 13:15:00 | 420.30 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-10-25 11:30:00 | 401.65 | 2024-10-29 13:15:00 | 420.30 | STOP_HIT | 1.00 | -4.64% |
| SELL | retest2 | 2024-10-25 13:15:00 | 405.10 | 2024-10-29 13:15:00 | 420.30 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest1 | 2024-10-31 11:45:00 | 426.90 | 2024-11-01 17:15:00 | 448.25 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-31 11:45:00 | 426.90 | 2024-11-05 12:15:00 | 444.40 | STOP_HIT | 0.50 | 4.10% |
| BUY | retest2 | 2024-11-07 15:00:00 | 454.00 | 2024-11-08 11:15:00 | 444.95 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-11-11 15:15:00 | 439.15 | 2024-11-18 09:15:00 | 417.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:15:00 | 439.75 | 2024-11-18 09:15:00 | 417.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:15:00 | 439.15 | 2024-11-19 09:15:00 | 427.25 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2024-11-12 12:15:00 | 439.75 | 2024-11-19 09:15:00 | 427.25 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2024-11-27 11:15:00 | 411.90 | 2024-11-27 12:15:00 | 416.25 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-11-29 13:45:00 | 418.15 | 2024-12-09 12:15:00 | 433.80 | STOP_HIT | 1.00 | 3.74% |
| SELL | retest2 | 2024-12-18 09:15:00 | 415.50 | 2024-12-18 12:15:00 | 431.15 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2024-12-20 12:45:00 | 415.90 | 2024-12-27 12:15:00 | 412.45 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2025-01-01 09:15:00 | 419.70 | 2025-01-06 09:15:00 | 417.45 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-01-15 11:45:00 | 396.90 | 2025-01-20 13:15:00 | 402.65 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-01-17 09:30:00 | 396.80 | 2025-01-20 13:15:00 | 402.65 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-01-17 10:00:00 | 396.65 | 2025-01-20 13:15:00 | 402.65 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-01-17 15:15:00 | 397.45 | 2025-01-20 13:15:00 | 402.65 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-01-24 11:00:00 | 380.05 | 2025-01-27 09:15:00 | 361.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 379.70 | 2025-01-27 09:15:00 | 360.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:15:00 | 379.60 | 2025-01-27 09:15:00 | 360.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 11:00:00 | 380.05 | 2025-01-28 09:15:00 | 342.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 379.70 | 2025-01-28 09:15:00 | 341.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 13:15:00 | 379.60 | 2025-01-28 09:15:00 | 341.64 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-07 11:00:00 | 368.45 | 2025-02-10 10:15:00 | 357.80 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-02-07 12:15:00 | 368.50 | 2025-02-10 10:15:00 | 357.80 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-02-13 14:15:00 | 333.10 | 2025-02-14 10:15:00 | 316.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:15:00 | 333.10 | 2025-02-17 14:15:00 | 315.65 | STOP_HIT | 0.50 | 5.24% |
| SELL | retest2 | 2025-02-25 14:45:00 | 311.95 | 2025-02-27 09:15:00 | 316.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-02-25 15:15:00 | 310.00 | 2025-02-27 09:15:00 | 316.50 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-02-27 12:15:00 | 312.00 | 2025-02-27 12:15:00 | 315.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-03-26 12:15:00 | 329.85 | 2025-03-27 14:15:00 | 335.95 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-03-26 14:00:00 | 330.80 | 2025-03-27 14:15:00 | 335.95 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-03-26 14:45:00 | 329.80 | 2025-03-27 14:15:00 | 335.95 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-03-27 10:45:00 | 330.20 | 2025-03-27 14:15:00 | 335.95 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-04-04 15:00:00 | 340.95 | 2025-04-07 09:15:00 | 317.00 | STOP_HIT | 1.00 | -7.02% |
| SELL | retest2 | 2025-04-11 13:15:00 | 322.75 | 2025-04-15 09:15:00 | 333.50 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2025-04-17 10:15:00 | 339.45 | 2025-04-22 11:15:00 | 373.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 13:30:00 | 336.60 | 2025-04-22 11:15:00 | 370.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:15:00 | 336.00 | 2025-04-22 11:15:00 | 369.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:45:00 | 337.85 | 2025-04-22 11:15:00 | 371.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-28 11:30:00 | 372.40 | 2025-04-29 11:15:00 | 364.20 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-04-29 09:15:00 | 374.70 | 2025-04-29 11:15:00 | 364.20 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-05-06 14:00:00 | 349.35 | 2025-05-07 11:15:00 | 366.50 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2025-05-16 14:30:00 | 413.50 | 2025-05-20 13:15:00 | 405.85 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-05-19 13:00:00 | 414.50 | 2025-05-20 13:15:00 | 405.85 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-05-19 15:15:00 | 415.60 | 2025-05-20 13:15:00 | 405.85 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-05-20 10:30:00 | 414.35 | 2025-05-20 13:15:00 | 405.85 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-05-26 09:15:00 | 424.80 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-05-26 12:00:00 | 418.70 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-05-26 13:15:00 | 418.00 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2025-05-27 10:30:00 | 419.80 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2025-05-28 09:45:00 | 418.15 | 2025-06-03 12:15:00 | 425.60 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest1 | 2025-06-11 09:15:00 | 485.15 | 2025-06-12 13:15:00 | 477.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2025-06-12 09:45:00 | 483.40 | 2025-06-12 13:15:00 | 477.10 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2025-06-12 11:30:00 | 483.60 | 2025-06-12 13:15:00 | 477.10 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-07-01 13:15:00 | 467.05 | 2025-07-04 13:15:00 | 471.75 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-01 15:00:00 | 468.25 | 2025-07-04 13:15:00 | 471.75 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-02 09:30:00 | 466.35 | 2025-07-04 13:15:00 | 471.75 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-17 11:30:00 | 529.00 | 2025-07-22 14:15:00 | 530.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-07-18 09:15:00 | 530.65 | 2025-07-22 14:15:00 | 530.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-07-22 12:15:00 | 529.05 | 2025-07-22 14:15:00 | 530.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-07-23 13:15:00 | 528.35 | 2025-07-23 13:15:00 | 530.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-08-13 09:15:00 | 450.30 | 2025-08-13 09:15:00 | 448.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-08-14 11:15:00 | 446.75 | 2025-08-18 14:15:00 | 449.15 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-08-14 13:00:00 | 445.60 | 2025-08-18 14:15:00 | 449.15 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-08-18 11:00:00 | 446.70 | 2025-08-18 14:15:00 | 449.15 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-08-18 12:45:00 | 446.80 | 2025-08-18 14:15:00 | 449.15 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-08-26 09:15:00 | 454.45 | 2025-08-28 15:15:00 | 431.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 11:00:00 | 455.25 | 2025-08-28 15:15:00 | 432.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 454.45 | 2025-08-29 10:15:00 | 439.60 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-08-26 11:00:00 | 455.25 | 2025-08-29 10:15:00 | 439.60 | STOP_HIT | 0.50 | 3.44% |
| BUY | retest2 | 2025-09-10 09:15:00 | 444.30 | 2025-09-11 11:15:00 | 439.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-10 09:45:00 | 443.85 | 2025-09-11 11:15:00 | 439.90 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-11 10:15:00 | 446.45 | 2025-09-11 11:15:00 | 439.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-18 09:30:00 | 448.30 | 2025-09-23 10:15:00 | 446.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-09-23 10:00:00 | 449.20 | 2025-09-23 10:15:00 | 446.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-13 11:30:00 | 488.65 | 2025-10-14 12:15:00 | 482.90 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-13 12:30:00 | 488.45 | 2025-10-14 12:15:00 | 482.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-17 09:15:00 | 501.75 | 2025-10-17 09:15:00 | 495.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-10-29 11:15:00 | 519.65 | 2025-11-06 09:15:00 | 526.45 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-10-29 12:15:00 | 517.35 | 2025-11-06 09:15:00 | 526.45 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-10-29 13:00:00 | 517.40 | 2025-11-06 09:15:00 | 526.45 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2025-11-13 11:45:00 | 542.40 | 2025-11-13 15:15:00 | 536.45 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-14 09:15:00 | 551.00 | 2025-11-20 10:15:00 | 547.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-24 13:15:00 | 543.55 | 2025-11-25 13:15:00 | 554.25 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-12-08 10:15:00 | 556.30 | 2025-12-10 09:15:00 | 574.40 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-12-09 11:30:00 | 557.20 | 2025-12-10 09:15:00 | 574.40 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-12-15 13:30:00 | 577.55 | 2025-12-16 09:15:00 | 565.05 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-12-18 14:15:00 | 556.65 | 2025-12-22 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-12-19 11:15:00 | 560.70 | 2025-12-22 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-12-19 12:45:00 | 560.00 | 2025-12-22 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-12-19 13:45:00 | 560.70 | 2025-12-22 09:15:00 | 572.60 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-23 09:15:00 | 572.55 | 2026-01-02 09:15:00 | 629.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-23 09:45:00 | 571.80 | 2026-01-02 09:15:00 | 628.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-23 13:00:00 | 571.15 | 2026-01-02 09:15:00 | 628.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-23 14:45:00 | 572.60 | 2026-01-02 09:15:00 | 629.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-24 10:15:00 | 591.10 | 2026-01-06 09:15:00 | 650.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-13 11:45:00 | 637.05 | 2026-01-19 12:15:00 | 645.15 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-19 09:45:00 | 638.55 | 2026-01-19 12:15:00 | 645.15 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-19 10:15:00 | 635.45 | 2026-01-19 12:15:00 | 645.15 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-19 11:15:00 | 639.10 | 2026-01-19 12:15:00 | 645.15 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-22 10:15:00 | 591.80 | 2026-01-22 14:15:00 | 532.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-22 14:30:00 | 560.60 | 2026-01-22 15:15:00 | 532.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 14:30:00 | 560.60 | 2026-01-27 15:15:00 | 528.15 | STOP_HIT | 0.50 | 5.79% |
| SELL | retest2 | 2026-02-03 10:45:00 | 494.35 | 2026-02-04 09:15:00 | 528.00 | STOP_HIT | 1.00 | -6.81% |
| SELL | retest2 | 2026-03-04 09:15:00 | 482.45 | 2026-03-09 09:15:00 | 458.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 482.45 | 2026-03-10 09:15:00 | 473.70 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2026-03-13 09:15:00 | 477.60 | 2026-03-18 11:15:00 | 483.85 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-03-13 13:45:00 | 475.40 | 2026-03-18 11:15:00 | 483.85 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-13 15:00:00 | 477.30 | 2026-03-18 11:15:00 | 483.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-04-02 09:15:00 | 426.70 | 2026-04-08 10:15:00 | 448.45 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2026-04-06 09:15:00 | 429.00 | 2026-04-08 10:15:00 | 448.45 | STOP_HIT | 1.00 | -4.53% |
| SELL | retest2 | 2026-04-06 09:45:00 | 430.75 | 2026-04-08 10:15:00 | 448.45 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2026-04-07 09:30:00 | 428.75 | 2026-04-08 10:15:00 | 448.45 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2026-04-13 10:15:00 | 446.60 | 2026-04-22 15:15:00 | 467.95 | STOP_HIT | 1.00 | 4.78% |
